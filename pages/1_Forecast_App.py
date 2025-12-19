import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, bindparam
import altair as alt 
import numpy as np
import json 
# Importaciones para Análisis Técnico
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from db import get_engine
from sidebar import render_sidebar


# --- BLOQUEO DE SEGURIDAD ---
if st.session_state.get('logged_in', False) == False:
    st.error("Por favor, inicie sesión primero para acceder a esta página.")
    st.page_link("app.py", label="Ir a Login", icon="🏠")
    st.stop()
# --- FIN DE BLOQUEO ---

render_sidebar()


import forecasting_engine  # importamos el módulo completo

from forecasting_engine import (
    project_future_drivers,
    run_model_competition, 
    run_final_forecast,
)
from db import get_engine



# --- Constantes del Negocio ---
DUMMY_VARS_LIST = [
    'CUP_IMPORT', 'CAMBIO_IVA', 'ELEC_PRESIDENCIALES', 'PARO', 'UTILIDADES',
    'DUMMY_COVID_LOCKDOWN' # <-- Dummy de COVID añadida
]
CORR_THRESHOLD = 0.5 

# --- Funciones de Cache ---
@st.cache_resource
def get_db_engine():
    try:
        engine = get_engine()
        return engine
    except Exception as e: st.error(f"Error (engine): {e}"); st.stop()

@st.cache_data
def load_drivers_data(_engine):
    """Carga y cachea solo los drivers."""
    df_drivers = pd.read_sql("SELECT * FROM historical_drivers", _engine, parse_dates=['Fecha'])
    df_drivers.columns = [
        col.replace(' ', '_').replace('/', '_').replace('.', '_')
        for col in df_drivers.columns
    ]
    df_drivers.columns = [col.upper() if col.upper() in DUMMY_VARS_LIST else col for col in df_drivers.columns]
    return df_drivers

@st.cache_data
def get_filter_options(_engine):
    """
    Obtiene los valores únicos para los filtros desde la BD granular.
    """
    print("Cargando opciones de filtros desde la BD...")
    options = {}
    with _engine.connect() as conn:
        options['segmento'] = pd.read_sql("SELECT DISTINCT segmento FROM sales_granular", conn)['segmento'].tolist()
        options['marca'] = pd.read_sql("SELECT DISTINCT marca FROM sales_granular", conn)['marca'].tolist()
        options['modelo'] = pd.read_sql("SELECT DISTINCT modelo FROM sales_granular", conn)['modelo'].tolist()
        options['provincia'] = pd.read_sql("SELECT DISTINCT provincia FROM sales_granular", conn)['provincia'].tolist()
        options['tipo_combustible'] = pd.read_sql("SELECT DISTINCT tipo_combustible FROM sales_granular", conn)['tipo_combustible'].tolist()
        options['origen'] = pd.read_sql("SELECT DISTINCT origen FROM sales_granular", conn)['origen'].tolist()
        options['tipo_hibridacion'] = pd.read_sql("SELECT DISTINCT tipo_hibridacion FROM sales_granular", conn)['tipo_hibridacion'].tolist()
        
        min_date, max_date = conn.execute(
            text("SELECT MIN(fecha_proceso), MAX(fecha_proceso) FROM sales_granular")
        ).fetchone()
        
        if min_date is None or max_date is None:
            print("ADVERTENCIA: La tabla 'sales_granular' está vacía o no tiene fechas.")
            today = pd.Timestamp.now()
            options['min_date'] = (today - pd.DateOffset(years=1)).to_pydatetime()
            options['max_date'] = today.to_pydatetime()
            options['is_empty'] = True
        else:
            options['min_date'] = min_date
            options['max_date'] = max_date
            options['is_empty'] = False
            
    print("Opciones de filtros cargadas.")
    return options

@st.cache_data
def load_filtered_sales_data(_engine, start_date, end_date, **filters):
    """
    Consulta SQL dinámica (Postgres) agregando por mes y segmento.
    Soporta filtros multi-select usando IN + expanding params.
    Devuelve columnas: Fecha, segmento, Ventas_Industria
    """
    print("Ejecutando consulta SQL dinámica para ventas...")

    base_sql = """
    SELECT
        date_trunc('month', fecha_proceso)::date AS "Fecha",
        segmento,
        SUM(unidades) AS "Ventas_Industria"
    FROM sales_granular
    WHERE fecha_proceso BETWEEN :start_date AND :end_date
    """

    params = {"start_date": start_date, "end_date": end_date}
    conditions = []
    bind_params = []

    # filtros posibles = keys que vienen en filters (segmento, marca, modelo, provincia, etc.)
    for col, values in filters.items():
        if values:  # lista no vacía
            conditions.append(f"{col} IN :{col}")
            params[col] = list(values)
            bind_params.append(bindparam(col, expanding=True))

    if conditions:
        base_sql += "\n AND " + "\n AND ".join(conditions)

    base_sql += """
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    stmt = text(base_sql)
    if bind_params:
        stmt = stmt.bindparams(*bind_params)

    df_sales = pd.read_sql_query(stmt, _engine, params=params, parse_dates=["Fecha"])
    print(f"Consulta SQL dinámica completada. {len(df_sales)} filas agregadas devueltas.")
    return df_sales

@st.cache_data
def get_segment_correlation(_engine, _segmento, _df_drivers_hist, _start_date, _end_date):
    """
    Calcula la correlación entre un segmento de ventas (o Total) y los drivers.

    Mejora:
    - Calcula correlación en NIVEL y en CRECIMIENTO INTERANUAL (YoY).
    - Excluye Dummies y devuelve un DataFrame con ambas métricas y un 'Score'
      que prioriza la correlación YoY (cuando existe).
    """
    print(f"Calculando correlación para el segmento: {_segmento}")

    # 1) Ventas agregadas del segmento
    if _segmento == "Total Industria":
        segment_filter = []
    else:
        segment_filter = [_segmento]

    df_sales_segment = load_filtered_sales_data(
        _engine,
        _start_date,
        _end_date,
        segmento=segment_filter
    )

    df_total_segment = df_sales_segment.groupby('Fecha')['Ventas_Industria'].sum().reset_index()

    if df_total_segment.empty:
        return None, f"No se encontraron ventas para '{_segmento}' en el rango de fechas."

    # 2) Unir ventas con drivers por Fecha
    df_corr_base = pd.merge(df_total_segment, _df_drivers_hist, on='Fecha', how='inner')
    if df_corr_base.empty:
        return None, "No se encontró intersección entre ventas y drivers en el rango seleccionado."

    try:
        # --- Correlación en nivel (como antes) ---
        df_corr_base_numeric = df_corr_base.drop(columns=['Fecha'], errors='ignore')
        corr_matrix_level = df_corr_base_numeric.corr(method='pearson')
        corr_level = corr_matrix_level['Ventas_Industria'].drop('Ventas_Industria', errors='ignore')

        # --- Correlación en crecimiento interanual (YoY) ---
        df_tmp = df_corr_base.set_index('Fecha').sort_index()
        # pct_change(12) ≈ crecimiento interanual
        df_yoy = df_tmp.pct_change(12).dropna(how='all')

        if 'Ventas_Industria' in df_yoy.columns:
            corr_matrix_yoy = df_yoy.corr(method='pearson')
            corr_yoy = corr_matrix_yoy['Ventas_Industria'].drop('Ventas_Industria', errors='ignore')
        else:
            corr_yoy = pd.Series(dtype=float)

        # Excluir dummies
        corr_level = corr_level.drop(labels=DUMMY_VARS_LIST, errors='ignore')
        corr_yoy = corr_yoy.drop(labels=DUMMY_VARS_LIST, errors='ignore')

        # --- Unificar variables y construir DataFrame de salida ---
        all_vars = sorted(set(corr_level.index).union(set(corr_yoy.index)))
        df_corr = pd.DataFrame({
            'Variable': all_vars,
            'Corr_Nivel': [corr_level.get(v, np.nan) for v in all_vars],
            'Corr_YoY': [corr_yoy.get(v, np.nan) for v in all_vars],
        })

        # Score: prioriza la correlación YoY; si no hay, usa la de nivel
        df_corr['Score'] = df_corr['Corr_YoY'].abs()
        df_corr['Score'] = df_corr['Score'].fillna(df_corr['Corr_Nivel'].abs())
        df_corr = df_corr.sort_values('Score', ascending=False).reset_index(drop=True)

        return df_corr, f"Correlación (Nivel y YoY) calculada para '{_segmento}'."

    except Exception as e:
        return None, f"Error al calcular correlación: {e}"

# --- FIN DE FUNCIONES ---

# --- Cargar datos iniciales ---
try:
    engine = get_db_engine()
    df_drivers_hist = load_drivers_data(engine)
    filter_options = get_filter_options(engine)
except Exception as e:
    st.error(f"Error al cargar datos históricos de la BD: {e}")
    st.exception(e)
    st.stop()

# =================================================================
# --- INICIO DE LA APLICACIÓN DE FORECAST ---
# =================================================================

st.title("Aplicación de Forecast S&OP")

tab1, tab2, tab3 = st.tabs([
    "📊 Verificación de Datos Históricos", 
    "⚙️ Paso 1: Escenarios de Drivers", 
    "📈 Paso 2: Pronóstico de Ventas"
])

# --- Pestaña 1: Verificación de Datos Históricos ---
with tab1:
    st.header("Verificación de Datos Históricos")
    st.markdown("Analiza el comportamiento pasado de las ventas y sus segmentos.")
    
    with st.expander("Filtros y Rango de Datos", expanded=True):
        
        if filter_options.get('is_empty', True):
            st.error("ERROR: No se encontraron datos de ventas en la tabla 'sales_granular'.")
            st.warning(f"Por favor, asegúrate de que el archivo 'data/industria.xlsx' tenga datos y vuelve a ejecutar el script 'load_initial_data.py'.")
            st.stop()
        
        min_val = filter_options['min_date'] 
        max_val = filter_options['max_date']
        
        default_start_val = (pd.to_datetime(max_val) - pd.DateOffset(years=4)).date()
        if default_start_val < min_val: default_start_val = min_val
        
        date_range_verif = st.slider(
            "Seleccionar Rango de Fechas:",
            min_value=min_val,  
            max_value=max_val,  
            value=(default_start_val, max_val), 
            format="YYYY-MM",
            key="date_slider_verif_tab1"
        )
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            segmentos_seleccionados = st.multiselect(
                "segmentos:",
                options=sorted(filter_options['segmento']),
                default=sorted(filter_options['segmento'])
            )
        with col_f2:
            marcas_seleccionadas = st.multiselect("Marcas:", options=sorted(filter_options['marca']), default=[])
        with col_f3:
            modelos_seleccionados = st.multiselect("Modelos:", options=sorted(filter_options['modelo']), default=[])

        col_f4, col_f5, col_f6 = st.columns(3)
        with col_f4:
            provincias_seleccionadas = st.multiselect("Provincias:", options=sorted(filter_options['provincia']), default=[])
        with col_f5:
            combustibles_seleccionados = st.multiselect("Tipo de Combustible:", options=sorted(filter_options['tipo_combustible']), default=[])
        with col_f6:
            origenes_seleccionados = st.multiselect("Origen (País):", options=sorted(filter_options['origen']), default=[])
        
        if st.button("Aplicar Filtros y Actualizar KPIs", type="primary"):
            with st.spinner("Cargando y agregando datos desde la BD..."):
                filters_dict = {
                    "segmento": segmentos_seleccionados,
                    "marca": marcas_seleccionadas,
                    "modelo": modelos_seleccionados,
                    "provincia": provincias_seleccionadas,
                    "tipo_combustible": combustibles_seleccionados,
                    "origen": origenes_seleccionados
                }
                
                df_sales_filtered = load_filtered_sales_data(
                    engine,
                    start_date=date_range_verif[0],
                    end_date=date_range_verif[1],
                    **filters_dict 
                )
                st.session_state.df_sales_filtered = df_sales_filtered
                st.session_state.date_range_verif = date_range_verif 
                st.success("¡Dashboard actualizado con los filtros seleccionados!")

    if 'df_sales_filtered' in st.session_state:
        df_sales_filtered = st.session_state.df_sales_filtered
        
        if df_sales_filtered.empty:
            st.warning("La combinación de filtros seleccionada no arrojó resultados (0 ventas). Por favor, ajusta los filtros.")
            st.stop()
            
        df_total_filtered = df_sales_filtered.groupby('Fecha')['Ventas_Industria'].sum().reset_index()
        df_segments_filtered = df_sales_filtered[df_sales_filtered['segmento'] != 'Total'].copy()

        # --- 🔧 INICIO POPOVER (v6.4) ---
        col_header_kpi, col_info_kpi = st.columns([0.9, 0.1])
        with col_header_kpi:
            st.subheader("KPIs Principales (Datos Filtrados)")
        with col_info_kpi:
            with st.popover("ℹ️ Info", use_container_width=False, help="Click para ver la explicación de los KPIs"):
                st.markdown("""
                **Interpretación de KPIs:**
                * **Promedio Mensual (L12M):** El promedio de ventas de los últimos 12 meses. Útil para ver la tendencia reciente.
                * **Mejor Mes:** El mes con más ventas en el rango filtrado.
                * **Desv. Estándar (L12M):** Mide la **volatilidad**. Un número alto significa que las ventas son muy variables mes a mes.
                * **Crecimiento YTD:** Compara las ventas acumuladas del año actual (ej. Ene-Oct 2025) contra el mismo período del año anterior (Ene-Oct 2024).
                """)
        # --- 🔧 FIN POPOVER (v6.4) ---
        
        total_sales = df_total_filtered['Ventas_Industria'].sum()
        max_date_in_filter = df_total_filtered['Fecha'].max()
        l12m_start_date = max_date_in_filter - pd.DateOffset(months=11)
        df_l12m = df_total_filtered[df_total_filtered['Fecha'] >= l12m_start_date]
        avg_monthly_sales_l12m = df_l12m['Ventas_Industria'].mean()
        std_dev_monthly_l12m = df_l12m['Ventas_Industria'].std()
        if not df_total_filtered.empty:
            best_month_sales = df_total_filtered['Ventas_Industria'].max()
            best_month_date = df_total_filtered[df_total_filtered['Ventas_Industria'] == best_month_sales]['Fecha'].dt.strftime('%Y-%m').values[0]
        else:
            best_month_sales = 0; best_month_date = "N/A"
        current_year = max_date_in_filter.year
        current_month = max_date_in_filter.month
        previous_year = current_year - 1
        current_ytd_sales = df_total_filtered[(df_total_filtered['Fecha'].dt.year == current_year) & (df_total_filtered['Fecha'].dt.month <= current_month)]['Ventas_Industria'].sum()
        previous_ytd_sales = df_total_filtered[(df_total_filtered['Fecha'].dt.year == previous_year) & (df_total_filtered['Fecha'].dt.month <= current_month)]['Ventas_Industria'].sum()
        ytd_growth = 0.0
        if previous_ytd_sales > 0: ytd_growth = (current_ytd_sales / previous_ytd_sales) - 1
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Ventas Totales (Rango)", f"{total_sales:,.0f} un.")
        col2.metric("Promedio Mensual (L12M)", f"{avg_monthly_sales_l12m:,.0f} un.")
        col3.metric("Mejor Mes (Rango)", f"{best_month_sales:,.0f} un.", delta=f"({best_month_date})")
        col4.metric("Desv. Estándar (L12M)", f"{std_dev_monthly_l12m:,.0f} un.")
        col5.metric(f"Crecimiento YTD ({current_year} vs {previous_year})", f"{ytd_growth:.2%}")
        st.divider()

        st.subheader("Evolución Temporal: Total (Línea) vs. segmentos (Barras Apiladas)")
        add_ma = st.checkbox("Añadir Media Móvil (3 meses) y ocultar barras")
        base = alt.Chart(df_segments_filtered).encode(x=alt.X('Fecha', title='Fecha'))
        chart_bars = base.mark_bar().encode(y=alt.Y('Ventas_Industria', title='Ventas (Unidades)', stack='zero'), color=alt.Color('segmento', title='segmento'), tooltip=['Fecha', 'segmento', 'Ventas_Industria'])
        chart_line = alt.Chart(df_total_filtered).mark_line(color='black', strokeWidth=3).encode(x='Fecha', y=alt.Y('Ventas_Industria', title='Ventas (Unidades)'), tooltip=[alt.Tooltip('Fecha', format='%Y-%m'), 'Ventas_Industria'])
        text_labels = chart_line.mark_text(align='center', dy=-10, fontSize=11).encode(text=alt.Text('Ventas_Industria', format='~s'), color=alt.value('black'))
        if add_ma:
            df_total_filtered['Ventas_MA_3M'] = df_total_filtered['Ventas_Industria'].rolling(3, center=True).mean()
            chart_ma = alt.Chart(df_total_filtered).mark_line(color='red', strokeWidth=2, strokeDash=[5,5]).encode(x='Fecha', y=alt.Y('Ventas_MA_3M', title='Media Móvil 3M'), tooltip=['Fecha', 'Ventas_MA_3M'])
            final_chart_layers = [chart_line, text_labels, chart_ma]
        else:
            final_chart_layers = [chart_bars, chart_line, text_labels]
        final_chart = alt.layer(*final_chart_layers).interactive()
        st.altair_chart(final_chart, use_container_width=True) 
        st.divider()
        
        col_pie, col_share_time = st.columns(2)
        with col_pie:
            st.subheader(f"Composición del Mercado (YTD {current_year})")
            df_segments_ytd = df_segments_filtered[(df_segments_filtered['Fecha'].dt.year == current_year) & (df_segments_filtered['Fecha'].dt.month <= current_month)]
            df_share = df_segments_ytd.groupby('segmento')['Ventas_Industria'].sum().reset_index()
            df_share['Share'] = df_share['Ventas_Industria'] / df_share['Ventas_Industria'].sum()
            base_pie = alt.Chart(df_share).encode(theta=alt.Theta("Ventas_Industria", stack=True))
            pie = base_pie.mark_arc(outerRadius=120, innerRadius=80).encode(color=alt.Color("segmento"), order=alt.Order("Ventas_Industria", sort="descending"), tooltip=["segmento", alt.Tooltip("Ventas_Industria", title="Total Unidades", format=","), alt.Tooltip("Share", title="Share", format=".1%")])
            text_pie = base_pie.mark_text(radius=140).encode(text=alt.Text("Share", format=".1%"), order=alt.Order("Ventas_Industria", sort="descending"), color=alt.value("black"))
            st.altair_chart(pie + text_pie, use_container_width=True) 
        with col_share_time:
            st.subheader("Evolución del Market Share (Rango)")
            df_share_over_time = df_segments_filtered.pivot_table(index='Fecha', columns='segmento', values='Ventas_Industria', aggfunc='sum').fillna(0)
            df_share_over_time = df_share_over_time.apply(lambda x: x / x.sum(), axis=1)
            df_share_over_time = df_share_over_time.reset_index().melt('Fecha', var_name='segmento', value_name='Share')
            chart_share_time = alt.Chart(df_share_over_time).mark_area().encode(x='Fecha', y=alt.Y('Share', stack='normalize', axis=alt.Axis(format='%')), color='segmento', tooltip=['Fecha', 'segmento', alt.Tooltip('Share', format='.1%')]).interactive()
            st.altair_chart(chart_share_time, use_container_width=True) 
        st.divider()

        # --- 🔧 INICIO POPOVER (v6.4) ---
        col_header_tech, col_info_tech = st.columns([0.9, 0.1])
        with col_header_tech:
            st.subheader("Análisis Técnico Adicional (Total Filtrado)")
        with col_info_tech:
            with st.popover("ℹ️ Info", use_container_width=True, help="Click para ver la explicación de los gráficos"):
                st.markdown("""
                **Interpretación de Gráficos Técnicos:**
                * **Estacionalidad (Boxplot):** Muestra el patrón de ventas típico para cada mes. Ayuda a identificar visualmente los meses pico (ej. Diciembre) y valles (ej. Enero/Febrero).
                * **Crecimiento (YoY):** Compara cada mes con el mismo mes del año anterior. Es la métrica de "salud" del mercado, ya que elimina la estacionalidad.
                * **Descomposición (STL):** Separa la serie en sus 3 componentes: **Tendencia** (largo plazo), **Estacionalidad** (patrón anual) y **Residual** (ruido o *shocks* como Paros/COVID).
                * **Autocorrelación (ACF/PACF):** Gráficos estadísticos que confirman la estacionalidad (picos en lag 12, 24 en ACF) y qué lags pasados son buenos predictores (pico en lag 1 en PACF).
                """)
        # --- 🔧 FIN POPOVER (v6.4) ---
        
        df_ts_analysis = df_total_filtered.set_index('Fecha')['Ventas_Industria'].asfreq('MS')
        df_ts_analysis = df_ts_analysis.fillna(method='ffill')
        if df_ts_analysis.empty or len(df_ts_analysis) < 24:
            st.warning("No hay suficientes datos (>24 meses) en el rango seleccionado para el análisis técnico.")
        else:
            tech_tab1, tech_tab2, tech_tab3, tech_tab4 = st.tabs(["Estacionalidad (Boxplot)", "Crecimiento (YoY / MoM)", "Descomposición (STL)", "Autocorrelación (ACF/PACF)"])
            with tech_tab1:
                df_season = df_total_filtered.copy()
                df_season['Mes'] = df_season['Fecha'].dt.strftime('%m-%b')
                chart_season = alt.Chart(df_season).mark_boxplot().encode(x=alt.X('Mes', sort='ascending'), y=alt.Y('Ventas_Industria', title='Ventas'), tooltip=['Mes', 'Ventas_Industria']).interactive()
                st.altair_chart(chart_season, use_container_width=True) 
            with tech_tab2:
                df_growth = df_total_filtered.set_index('Fecha').copy()
                df_growth['YoY (%)'] = df_growth['Ventas_Industria'].pct_change(12) * 100
                df_growth['MoM (%)'] = df_growth['Ventas_Industria'].pct_change(1) * 100
                st.markdown("##### Crecimiento Interanual (YoY)")
                chart_yoy = alt.Chart(df_growth.reset_index()).mark_area(color="lightblue", line=alt.OverlayMarkDef(color="blue")).encode(x='Fecha', y='YoY (%)', tooltip=['Fecha', alt.Tooltip('YoY (%)', format='.2f')]).interactive()
                st.altair_chart(chart_yoy, use_container_width=True) 
            with tech_tab3:
                try:
                    result = seasonal_decompose(df_ts_analysis.dropna(), model='additive', period=12)
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
                    result.observed.plot(ax=ax1, legend=False); ax1.set_ylabel('Observado')
                    result.trend.plot(ax=ax2, legend=False); ax2.set_ylabel('Tendencia')
                    result.seasonal.plot(ax=ax3, legend=False); ax3.set_ylabel('Estacional')
                    result.resid.plot(ax=ax4, legend=False); ax4.set_ylabel('Residual')
                    plt.tight_layout(); st.pyplot(fig) 
                    plt.close(fig) 
                except Exception as e: st.error(f"Error al calcular la descomposición: {e}")
            with tech_tab4:
                try:
                    data_for_plots = df_ts_analysis.dropna()
                    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                    plot_acf(data_for_plots, ax=ax_acf, lags=36); ax_acf.set_title("Autocorrelación (ACF)")
                    st.pyplot(fig_acf)
                    plt.close(fig_acf)
                    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                    plot_pacf(data_for_plots, ax=ax_pacf, lags=36, method='ols'); ax_pacf.set_title("Autocorrelación Parcial (PACF)")
                    st.pyplot(fig_pacf)
                    plt.close(fig_pacf)
                except Exception as e: st.error(f"Error al calcular ACF/PACF: {e}")
        
        st.divider()
        st.subheader("Exportar Datos")
        @st.cache_data
        def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
        csv = convert_df_to_csv(df_segments_filtered)
        st.download_button(label="Descargar Datos Filtrados (segmentos) como CSV", data=csv, file_name="datos_segmentos_filtrados.csv", mime="text/csv")
        csv_total = convert_df_to_csv(df_total_filtered)
        st.download_button(label="Descargar Datos Filtrados (Total) como CSV", data=csv_total, file_name="datos_total_filtrados.csv", mime="text/csv")

    else:
        st.info("Presiona 'Aplicar Filtros y Actualizar KPIs' para cargar el dashboard.")


# --- Pestaña 2: Proyección de Drivers ---
with tab2:
    # --- 🔧 INICIO POPOVER (v6.4) ---
    col_header_p2, col_info_p2 = st.columns([0.9, 0.1])
    with col_header_p2:
        st.header("Paso 1: Generar y Ajustar Escenario de Drivers")
    with col_info_p2:
        with st.popover("ℹ️ Info", use_container_width=True, help="Click para ver la guía de esta pestaña"):
            st.markdown("""
            **Objetivo:** Crear los supuestos de negocio (drivers) que alimentarán el modelo de ventas.
            
            **Metodología "Ciencia + Arte":**
            
            1.  **Configuración (Ciencia):** Selecciona el horizonte (ej. 14 meses) y el segmento (ej. `Total Industria`) para el análisis de correlación.
            2.  **Selección (Ciencia):** La app te sugiere *drivers* basado en su correlación histórica.
            3.  **Generación (Ciencia):** El modelo `SARIMAX` proyecta estos *drivers* basándose en su tendencia pasada.
            4.  **Ajuste (Arte):** ¡El paso más importante! Revisa las proyecciones estadísticas.
                * Usa el **Asistente de Proyección Anual %** para *drivers* que siguen una meta (ej. `PIB` al 2%).
                * Usa el **Editor de Escenarios** de abajo para ajustar manualmente *drivers* volátiles (ej. `Riesgo_Pais` a 400 puntos).
            5.  **Guardar:** Guarda tus supuestos (Ciencia + Arte) como un nuevo "Escenario".
            """)
    # --- 🔧 FIN POPOVER (v6.4) ---
    
    st.markdown("Este proceso ahora se basa en el segmento. Las proyecciones se guardarán en la base de datos.")
    
    st.subheader("1. Configuración del Análisis")
    
    if 'date_range_verif' not in st.session_state:
        st.error("Por favor, ve a la Pestaña 1 y presiona 'Aplicar Filtros' primero.")
        st.stop()
        
    date_range_corr = st.session_state.date_range_verif

    col1, col2 = st.columns(2)
    with col1:
        horizonte = st.slider("Meses a Proyectar", min_value=6, max_value=36, value=18, step=1, key="horizonte_slider_tab2")
    with col2:
        segmento_options = ["Total Industria"] + sorted(filter_options['segmento'])
        segmento_base_tab2 = st.selectbox(
            "Segmento Base para Correlación",
            options=segmento_options,
            index=0, 
            key="segmento_base_tab2"
        )

    st.subheader(f"2. Selección de Variables (Correlación vs. Ventas de '{segmento_base_tab2}')")

    
    with st.spinner(f"Calculando correlación para '{segmento_base_tab2}'..."):
        df_corr, corr_msg = get_segment_correlation(
            engine,
            segmento_base_tab2,
            df_drivers_hist,   # <-- así
            date_range_corr[0],
            date_range_corr[1]
        )

    if df_corr is None:
        st.error(corr_msg)
        st.stop()
    else:
        st.success(corr_msg)
        
        # Preparar tabla para el usuario: mostramos correlación en nivel y YoY
        df_corr_display = df_corr.copy()
        df_corr_display['Incluir'] = df_corr_display['Score'].abs() > CORR_THRESHOLD

        # Regla de negocio: no auto-seleccionar IMP_CBU (muy cercano a la propia venta)
        if 'IMP_CBU' in df_corr_display['Variable'].values:
            df_corr_display.loc[df_corr_display['Variable'] == 'IMP_CBU', 'Incluir'] = False

        st.info(
            f"Utilice la tabla para seleccionar qué variables se usarán como DRIVERS. "
            f"El *Score* prioriza la correlación en crecimiento interanual (YoY). "
            f"Se sugieren automáticamente las variables con Score fuerte (> {CORR_THRESHOLD})."
        )
        
        edited_corr_df = st.data_editor(
            df_corr_display,
            column_config={
                "Variable": st.column_config.TextColumn("Variable", disabled=True),
                "Corr_Nivel": st.column_config.NumberColumn("Corr. Nivel", format="%.3f", disabled=True),
                "Corr_YoY": st.column_config.NumberColumn("Corr. YoY", format="%.3f", disabled=True),
                "Score": st.column_config.NumberColumn("Score", format="%.3f", disabled=True),
                "Incluir": st.column_config.CheckboxColumn("Incluir?", default=False)
            },
            hide_index=True,
            use_container_width=True, 
            key="editor_corr_tab2"
        )
        
        drivers_seleccionados = edited_corr_df[edited_corr_df['Incluir']]['Variable'].tolist()
        st.session_state.drivers_seleccionados_tab2 = drivers_seleccionados  
        
    st.subheader("3. Generación de Proyección (Ciencia)")
    
    if st.button("Generar Proyección Estadística de Drivers Seleccionados", type="primary"):
        drivers_seleccionados = st.session_state.get('drivers_seleccionados_tab2', [])
        
        if not drivers_seleccionados:
            st.warning("No se seleccionó ningún driver. Por favor, seleccione al menos uno en la tabla de correlación.")
        else:
            with st.spinner(f"Corriendo modelos SARIMAX/AutoARIMA para {len(drivers_seleccionados)} drivers..."):
                
                df_sales_completa_sql = text("""
                SELECT
                date_trunc('month', fecha_proceso)::date AS "Fecha",
                segmento,
                SUM(unidades) AS "Ventas_Industria"
                FROM sales_granular
                GROUP BY 1, 2
                ORDER BY 1, 2
                """)
                df_sales_completa = pd.read_sql(df_sales_completa_sql, engine, parse_dates=['Fecha'])

                df_master_hist_engine = pd.merge(
                    df_sales_completa, 
                    df_drivers_hist, 
                    on='Fecha', 
                    how='inner'
                )

                drivers_a_mantener = drivers_seleccionados + DUMMY_VARS_LIST + ['Fecha', 'segmento', 'Ventas_Industria']
                cols_en_df = [col for col in drivers_a_mantener if col in df_master_hist_engine.columns]
                
                df_master_hist_filtrado = df_master_hist_engine[cols_en_df].set_index('Fecha')
                
                df_proyectado_scenarios = project_future_drivers(df_master_hist_filtrado, horizonte)
                
                numeric_cols = df_proyectado_scenarios.select_dtypes(include=np.number).columns
                df_proyectado_scenarios[numeric_cols] = df_proyectado_scenarios[numeric_cols].clip(lower=0) 
                
                st.session_state.projected_drivers_scenarios_tab2 = df_proyectado_scenarios
                
                base_vars = [c.replace('_normal', '') for c in df_proyectado_scenarios.filter(like='_normal').columns]
                st.session_state.uncertainty_map_upper = {var: 100 for var in base_vars}
                st.session_state.uncertainty_map_lower = {var: 100 for var in base_vars}

                st.success(f"Proyección generada para {len(drivers_seleccionados)} drivers (+ dummies) para {horizonte} meses.")

    if 'projected_drivers_scenarios_tab2' in st.session_state:
        
        df_proj_base = st.session_state.projected_drivers_scenarios_tab2
        
        base_vars_all = list(st.session_state.uncertainty_map_upper.keys())
        
        indep_vars = sorted([v for v in base_vars_all if v.upper() not in DUMMY_VARS_LIST])
        dummy_vars = sorted([v for v in base_vars_all if v.upper() in DUMMY_VARS_LIST])
        base_vars_sorted = indep_vars + dummy_vars
        
        def format_var_name(var):
            if var.upper() in DUMMY_VARS_LIST:
                return f"🔵 [DUMMY] {var}"
            else:
                return f"🟢 [INDEP] {var}"
        
        def extract_var_name(formatted_var):
            if formatted_var.startswith("🔵 [DUMMY] "):
                return formatted_var[10:]
            elif formatted_var.startswith("🟢 [INDEP] "):
                return formatted_var[10:]
            return formatted_var

        formatted_vars = [format_var_name(var) for var in base_vars_sorted]
        
        if not formatted_vars:
            st.warning("No se proyectaron variables. Revise la selección.")
            st.stop()
        
        # --- 🔧 INICIO NUEVA SECCIÓN: Asistente de Proyección (v6.1) ---
        with st.expander("Asistente de Proyección (Override Anual %)", expanded=False):
            st.info("Use esta herramienta para sobrescribir la proyección estadística con un supuesto de crecimiento anual (YoY).")
            
            col_assist_1, col_assist_2 = st.columns([2,1])
            with col_assist_1:
                selected_var_for_growth = st.selectbox(
                    "Seleccione la variable para aplicar crecimiento:",
                    options=indep_vars, # Solo variables independientes
                    key="assist_var_select"
                )
            with col_assist_2:
                growth_rate = st.number_input(
                    "Tasa de Crecimiento Anual (%):",
                    min_value=-50.0, max_value=50.0, value=2.0, step=0.1,
                    key="assist_growth_rate"
                )
            
            if st.button("Aplicar Crecimiento Anual a Escenarios"):
                try:
                    var = selected_var_for_growth
                    rate = growth_rate / 100.0
                    df_hist = df_drivers_hist.set_index('Fecha') # Asegurar que el histórico tenga índice de fecha
                    df_proj = st.session_state.projected_drivers_scenarios_tab2

                    new_normal = []
                    new_optimista = []
                    new_pesimista = []
                    
                    for date in df_proj.index:
                        last_year_date = date - pd.DateOffset(years=1)
                        if last_year_date in df_hist.index:
                            last_year_val = df_hist.loc[last_year_date, var]
                            new_val = last_year_val * (1 + rate)
                            new_normal.append(new_val)
                            
                            # Asumir bandas de +/- 5% sobre el nuevo valor
                            new_optimista.append(new_val * 1.05)
                            new_pesimista.append(new_val * 0.95)
                        else:
                            # Fallback si no hay historia (ej. primer año)
                            new_normal.append(df_proj.loc[date, f"{var}_normal"])
                            new_optimista.append(df_proj.loc[date, f"{var}_optimista"])
                            new_pesimista.append(df_proj.loc[date, f"{var}_pesimista"])

                    # Actualizar el DataFrame en session_state
                    df_proj[f"{var}_normal"] = new_normal
                    df_proj[f"{var}_optimista"] = new_optimista
                    df_proj[f"{var}_pesimista"] = new_pesimista
                    
                    st.session_state.projected_drivers_scenarios_tab2 = df_proj
                    st.success(f"Proyección de {var} actualizada con {growth_rate}% YoY. Revise el gráfico y la tabla de abajo.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al aplicar crecimiento: {e}")

        # --- 🔧 FIN NUEVA SECCIÓN ---


        st.subheader("4. Visualizar y Ajustar Proyecciones (Ajuste Macro Asimétrico)")
        st.info("Use el dropdown para seleccionar una variable. Luego, ajuste sus bandas de incertidumbre Superior (Optimista) e Inferior (Pesimista).")

        selected_formatted_var = st.selectbox(
            "Seleccione Variable para Visualizar y Ajustar:", 
            options=formatted_vars,
            key="viz_var_tab2"
        )
        
        var_to_plot = extract_var_name(selected_formatted_var)
        
        is_dummy = var_to_plot.upper() in DUMMY_VARS_LIST
        
        col1_viz, col2_viz = st.columns(2)
        
        current_upper = st.session_state.uncertainty_map_upper.get(var_to_plot, 100)
        current_lower = st.session_state.uncertainty_map_lower.get(var_to_plot, 100)

        with col1_viz:
            new_upper = st.slider(
                f"Ajuste Superior (Optimista) para {var_to_plot}", 
                min_value=0, max_value=200, value=current_upper, step=10, 
                format="%d%%",
                key=f"slider_upper_{var_to_plot}",
                help="Ajusta la banda superior. 100% = sin cambios. 0% = igual a la base.",
                disabled=is_dummy 
            )
        with col2_viz:
            new_lower = st.slider(
                f"Ajuste Inferior (Pesimista) para {var_to_plot}", 
                min_value=0, max_value=200, value=current_lower, step=10, 
                format="%d%%",
                key=f"slider_lower_{var_to_plot}",
                help="Ajusta la banda inferior. 100% = sin cambios. 0% = igual a la base.",
                disabled=is_dummy 
            )
        
        if is_dummy:
            st.caption("ℹ️ Los ajustes de incertidumbre no aplican a variables Dummies. Edite los valores 'base' directamente en la tabla del Paso 5.")
            new_upper = 100
            new_lower = 100

        st.session_state.uncertainty_map_upper[var_to_plot] = new_upper
        st.session_state.uncertainty_map_lower[var_to_plot] = new_lower
        
        upper_multiplier = new_upper / 100.0
        lower_multiplier = new_lower / 100.0
        
        df_chart_adj = pd.DataFrame(index=df_proj_base.index)
        df_chart_adj['base'] = df_proj_base[f'{var_to_plot}_normal']
        
        chart_layers = []
        
        if f'{var_to_plot}_optimista' in df_proj_base.columns:
            base_margin_upper = df_proj_base[f'{var_to_plot}_optimista'] - df_proj_base[f'{var_to_plot}_normal']
            base_margin_lower = df_proj_base[f'{var_to_plot}_normal'] - df_proj_base[f'{var_to_plot}_pesimista']
            
            df_chart_adj['optimista_adj'] = df_chart_adj['base'] + (base_margin_upper * upper_multiplier)
            df_chart_adj['pesimista_adj'] = df_chart_adj['base'] - (base_margin_lower * lower_multiplier)
            df_chart_adj['pesimista_adj'] = df_chart_adj['pesimista_adj'].clip(lower=0)
        else:
            df_chart_adj['optimista_adj'] = df_chart_adj['base']
            df_chart_adj['pesimista_adj'] = df_chart_adj['base']

        df_chart_adj = df_chart_adj.reset_index().rename(columns={'index': 'Fecha'})

        if f'{var_to_plot}_optimista' in df_proj_base.columns and not is_dummy:
            banda_conf = alt.Chart(df_chart_adj).mark_area(opacity=0.3, color='#85C1E9').encode(
                x=alt.X('Fecha:T', title='Fecha'), 
                y=alt.Y('pesimista_adj', title=var_to_plot), 
                y2=alt.Y2('optimista_adj'),
                tooltip=[
                    alt.Tooltip('Fecha:T'), 
                    alt.Tooltip('pesimista_adj', title='Pesimista Ajustado'), 
                    alt.Tooltip('optimista_adj', title='Optimista Ajustado')
                ]
            )
            chart_layers.append(banda_conf)

        df_hist_chart = df_drivers_hist.set_index('Fecha') # Asegurar índice de fecha
        df_hist_chart = df_hist_chart[df_hist_chart.index >= '2020-01-01'][[var_to_plot]].copy()
        df_hist_chart = df_hist_chart.rename(columns={var_to_plot: 'valor'}).reset_index()
        
        linea_real = alt.Chart(df_hist_chart).mark_line(point=False, color='black').encode(
            x=alt.X('Fecha:T', title='Fecha'), 
            y=alt.Y('valor', title=var_to_plot), 
            tooltip=[alt.Tooltip('Fecha:T'), 'valor'] 
        )
        
        linea_proy = alt.Chart(df_chart_adj).mark_line(point=True, color='blue', strokeDash=[5,5]).encode(
            x=alt.X('Fecha:T', title='Fecha'), 
            y=alt.Y('base', title=var_to_plot), 
            tooltip=[
                alt.Tooltip('Fecha:T'), 
                alt.Tooltip('base', title='Proyección Base')
            ]
        )
        
        chart_layers.insert(0, linea_real)
        chart_layers.append(linea_proy)
        
        st.altair_chart(alt.layer(*chart_layers).interactive(), use_container_width=True) 

        st.subheader("5. Editor de Escenarios (Override de Negocio)")
        st.warning(f"La tabla de abajo se ha actualizado con todos sus ajustes. Aún puede sobrescribir manualmente cualquier celda.")
        
        df_adjusted_for_editor = df_proj_base.copy()
        
        for var in base_vars_all: 
            if var.upper() not in DUMMY_VARS_LIST and f'{var}_optimista' in df_adjusted_for_editor.columns:
                upper_mult = st.session_state.uncertainty_map_upper.get(var, 100) / 100.0
                lower_mult = st.session_state.uncertainty_map_lower.get(var, 100) / 100.0
                
                base_margin_upper = df_proj_base[f'{var}_optimista'] - df_proj_base[f'{var}_normal']
                base_margin_lower = df_proj_base[f'{var}_normal'] - df_proj_base[f'{var}_pesimista']
                
                df_adjusted_for_editor[f'{var}_optimista'] = df_proj_base[f'{var}_normal'] + (base_margin_upper * upper_mult)
                df_adjusted_for_editor[f'{var}_pesimista'] = df_proj_base[f'{var}_normal'] - (base_margin_lower * lower_mult)
                df_adjusted_for_editor[f'{var}_pesimista'] = df_adjusted_for_editor[f'{var}_pesimista'].clip(lower=0)
            
            elif f'{var}_optimista' in df_adjusted_for_editor.columns: 
                df_adjusted_for_editor[f'{var}_optimista'] = df_proj_base[f'{var}_normal']
                df_adjusted_for_editor[f'{var}_pesimista'] = df_proj_base[f'{var}_normal']

        edited_drivers_df = st.data_editor(
            df_adjusted_for_editor, 
            height=400, 
            key="editor_drivers_tab2",
            use_container_width=True 
        )
        
        st.subheader("6. Guardar Escenario en Base de Datos")
        
        scenario_name = st.text_input(
            "Nombre del Escenario:", 
            placeholder=f"Ej: Plan Base {segmento_base_tab2} {horizonte}m",
            key="scenario_name_tab2"
        )
        
        if st.button("Guardar Escenario", type="primary"):
            if not scenario_name:
                st.error("Por favor, ingrese un nombre para el escenario.")
            else:
                try:
                    df_to_save = edited_drivers_df.copy()
                    
                    df_melted = df_to_save.reset_index().melt(
                        id_vars='index', 
                        var_name='col_name', 
                        value_name='valor_proyectado'
                    )
                    
                    df_melted_split = df_melted['col_name'].str.rsplit('_', n=1, expand=True)
                    df_melted['variable'] = df_melted_split[0]
                    df_melted['escenario'] = df_melted_split[1]

                    df_melted['scenario_name'] = scenario_name
                    df_melted['segmento_base'] = segmento_base_tab2
                    
                    df_final_insert = df_melted.rename(columns={'index': 'fecha'})
                    df_final_insert = df_final_insert[['scenario_name', 'segmento_base', 'fecha', 'variable', 'escenario', 'valor_proyectado']]
                    df_final_insert['fecha'] = pd.to_datetime(df_final_insert['fecha']).dt.date
                    
                    df_final_insert['escenario'] = df_final_insert['escenario'].replace({'base': 'normal'})

                    with engine.connect() as conn:
                        with conn.begin(): 
                            sql_delete = text("DELETE FROM forecast_driver_scenarios WHERE scenario_name = :name")
                            conn.execute(sql_delete, {"name": scenario_name})
                            
                            df_final_insert.to_sql(
                                'forecast_driver_scenarios',
                                conn,
                                if_exists='append',
                                index=False,
                                chunksize=1000
                            )
                    
                    st.success(f"¡Escenario '{scenario_name}' guardado exitosamente en la BD con {len(df_final_insert)} registros!")
                    
                except Exception as e:
                    st.error(f"Error al guardar: {e}")
                    st.exception(e)

# =================================================================
# --- Pestaña 3: Pronóstico de Ventas (REESCRITA v6.4) ---
# =================================================================
with tab3:
    # --- 🔧 INICIO POPOVER (v6.4) ---
    col_header_p3, col_info_p3 = st.columns([0.9, 0.1])
    with col_header_p3:
        st.header(f"Paso 2: Ejecutar Forecast de Ventas")
    with col_info_p3:
        with st.popover("ℹ️ Info", use_container_width=True, help="Click para ver la guía de esta pestaña"):
            st.markdown("""
            **Objetivo:** Generar el pronóstico de ventas final combinando la Ciencia y el Arte.
            
            **Flujo de Trabajo:**
            
            1.  **Cargar Escenario:** Selecciona el escenario de *drivers* que guardaste en el Paso 1.
            2.  **Correr Competencia:** Presiona el botón para ejecutar el *backtest*. Esto simula qué tan bien cada modelo (XGBoost, SARIMAX...) habría predicho los datos históricos.
            3.  **Seleccionar Modelo:** Revisa el "Leaderboard" (tabla de errores) y selecciona el modelo ganador (usualmente el de menor **MAPE**).
            4.  **Ajuste Final (Arte):** El modelo generará el pronóstico. Si ves un número que tu intuición de negocio rechaza (ej. una caída ilógica en Noviembre), **sobrescríbelo** directamente en el **Editor de Pronóstico**.
            5.  **Validar (Arte):** Revisa el gráfico "Análisis de Crecimiento (YoY)" para comparar tu pronóstico (ya ajustado) con los años anteriores.
            6.  **Guardar:** Una vez satisfecho, ponle un nombre y guarda tu proyección final en la base de datos.
            """)
    # --- 🔧 FIN POPOVER (v6.4) ---

    st.subheader("1. Cargar Escenario de Drivers")
    
    if 'df_sales_filtered' not in st.session_state:
        st.error("Por favor, ve a la Pestaña 1 y presiona 'Aplicar Filtros' primero.")
        st.stop()
        
    try:
        scenarios_list_df = pd.read_sql("SELECT DISTINCT scenario_name, segmento_base, MAX(created_at) as created_at FROM forecast_driver_scenarios GROUP BY 1, 2 ORDER BY 3 DESC", engine)
        
        if scenarios_list_df.empty:
            st.error("No se han guardado escenarios de drivers en la Base de Datos. Por favor, complete el Paso 1 (Pestaña 2) primero.")
            st.stop()
            
        options_list = [f"{row['scenario_name']} (Base: {row['segmento_base']})" for index, row in scenarios_list_df.iterrows()]
        scenario_key_map = {f"{row['scenario_name']} (Base: {row['segmento_base']})": row['scenario_name'] for index, row in scenarios_list_df.iterrows()}

        scenario_display = st.selectbox(
            "Seleccione el Escenario de Drivers a utilizar:",
            options=options_list,
            key="scenario_load_tab3"
        )
        
        scenario_to_load = scenario_key_map[scenario_display]
        
    except Exception as e:
        st.error(f"Error al cargar lista de escenarios: {e}")
        st.stop()

    st.subheader("2. Configuración del Pronóstico")
    
    # --- 🔧 INICIO CORRECCIÓN GRÁFICO HISTÓRICO (v6.1) ---
    df_sales_completa_sql = text("""
    SELECT
    date_trunc('month', fecha_proceso)::date AS "Fecha",
    segmento,
    SUM(unidades) AS "Ventas_Industria"
    FROM sales_granular
    GROUP BY 1, 2
    ORDER BY 1, 2
    """)
    df_sales_completa = pd.read_sql(df_sales_completa_sql, engine, parse_dates=['Fecha'])
    
    df_master_hist_engine_full = pd.merge(
        df_sales_completa, 
        df_drivers_hist, 
        on='Fecha', how='inner'
    ).set_index('Fecha')
    
    df_total_forecast = df_sales_completa.groupby('Fecha')['Ventas_Industria'].sum().reset_index()
    df_total_forecast['segmento'] = 'Total_Industria' 
    
    df_master_hist_engine_total_full = pd.merge(
        df_total_forecast,
        df_drivers_hist,
        on='Fecha'
    ).set_index('Fecha')
    
    df_master_hist_train = pd.merge(
        st.session_state.df_sales_filtered, 
        df_drivers_hist, 
        on='Fecha', how='inner'
    ).set_index('Fecha')
    
    df_total_train = st.session_state.df_sales_filtered.groupby('Fecha')['Ventas_Industria'].sum().reset_index()
    df_total_train['segmento'] = 'Total_Filtrado'
    
    df_master_hist_total_train = pd.merge(
        df_total_train,
        df_drivers_hist,
        on='Fecha'
    ).set_index('Fecha')
    
    df_master_hist_train_final = pd.concat([
        df_master_hist_train,
        df_master_hist_total_train
    ])
    
    df_master_hist_graph_final = pd.concat([
        df_master_hist_engine_full,
        df_master_hist_engine_total_full
    ])
    # --- FIN CORRECCIÓN GRÁFICO HISTÓRICO ---
    
    segmentos_forecast_disponibles = df_master_hist_train_final['segmento'].unique()
    
    if 'Total_Filtrado' in segmentos_forecast_disponibles:
        default_segment_index = list(segmentos_forecast_disponibles).index('Total_Filtrado')
    elif 'Total_Industria' in segmentos_forecast_disponibles:
        default_segment_index = list(segmentos_forecast_disponibles).index('Total_Industria')
    else:
        default_segment_index = 0


    segmento_forecast_tab3 = st.selectbox(
        "Segmento a Pronosticar (basado en tus filtros de Pestaña 1):",
        options=segmentos_forecast_disponibles,
        index=default_segment_index,
        key="segmento_a_pronosticar_tab3"
    )
    
    st.subheader("3. Correr Competencia de Modelos (Backtest)")
    
    if st.button("Correr Competencia de Modelos", type="primary"):
        if not scenario_to_load:
            st.error("¡Error! Por favor, seleccione un escenario de drivers de la lista.")
        else:
            with st.spinner(f"Cargando escenario y ejecutando competencia para '{segmento_forecast_tab3}'..."):
                try:
                    sql_load = text("SELECT fecha, variable, escenario, valor_proyectado FROM forecast_driver_scenarios WHERE scenario_name = :name")
                    df_long = pd.read_sql(sql_load, engine, params={"name": scenario_to_load})
                    
                    df_wide_pivot = df_long.pivot_table(index='fecha', columns=['variable', 'escenario'], values='valor_proyectado')
                    df_wide_pivot.columns = ['_'.join(col) for col in df_wide_pivot.columns]
                    df_wide_pivot.index = pd.to_datetime(df_wide_pivot.index)
                    
                    st.session_state.loaded_drivers_for_forecast = df_wide_pivot
                    print(f"Escenario '{scenario_to_load}' cargado y pivotado.")
                    
                    future_driver_cols = df_long[df_long['escenario'] == 'normal']['variable'].unique()
                    
                    cols_to_keep = list(future_driver_cols) + ['Ventas_Industria', 'segmento']
                    cols_to_keep_safe = [col for col in cols_to_keep if col in df_master_hist_train_final.columns]
                    
                    df_master_hist_engine_filtered = df_master_hist_train_final[cols_to_keep_safe].copy()
                    
                    st.session_state.df_master_hist_engine_final_filtered = df_master_hist_engine_filtered
                    
                    cols_to_keep_graph = [col for col in cols_to_keep if col in df_master_hist_graph_final.columns]
                    st.session_state.df_master_hist_graph_final_filtered = df_master_hist_graph_final[cols_to_keep_graph].copy()

                    print(f"Historial filtrado para incluir solo {len(future_driver_cols)} drivers del escenario.")

                    # >>> NUEVO: definir horizonte y nombre para guardar métricas en BD <<<
                    horizon_months = st.session_state.loaded_drivers_for_forecast.index.nunique()
                    # Por ahora usamos el nombre del escenario + segmento como "projection_name" de métricas
                    projection_name_metrics = f"{scenario_to_load}__{segmento_forecast_tab3}"
                    scope_text = segmento_forecast_tab3  # puedes mapearlo a "Total Industria", "SUV", etc. si quieres

                    metrics_dict, backtest_forecasts_dict, winner_rmse, winner_backtest_data, cv_metadata = run_model_competition(
                        df_master_hist=st.session_state.df_master_hist_engine_final_filtered,
                        segment_name=segmento_forecast_tab3
                    )

                    st.session_state.metrics_dict = metrics_dict
                    st.session_state.backtest_forecasts_dict = backtest_forecasts_dict
                    st.session_state.winner_rmse_for_ml = winner_rmse
                    st.session_state.winner_backtest_data_for_ml = winner_backtest_data
                    st.session_state.segmento_forecast_run = segmento_forecast_tab3
                    st.session_state.cv_metadata_for_ml = cv_metadata

                    
                    print("Competencia de modelos completada.")
                    st.success("¡Competencia de modelos completada!")

                except Exception as e:
                    st.error(f"Error en la competencia de modelos: {e}"); st.exception(e);

    if 'metrics_dict' in st.session_state:
        
        st.subheader("Leaderboard de Modelos (Resultados del Backtest)")
        metrics_df = pd.DataFrame.from_dict(st.session_state.metrics_dict, orient='index')
        metrics_df = metrics_df[['MAPE', 'MAE', 'RMSE']] 
        metrics_df['MAPE'] = (metrics_df['MAPE'] * 100).round(2).astype(str) + ' %'
        metrics_df['MAE'] = metrics_df['MAE'].round(0).astype(int)
        metrics_df['RMSE'] = metrics_df['RMSE'].round(0).astype(int)
        st.dataframe(metrics_df.sort_values(by="MAPE"), use_container_width=True) 
        
        st.divider()
        st.subheader("4. Visualizar Pronóstico por Modelo")
        
        model_to_display = st.selectbox(
            "Seleccione un modelo para visualizar el pronóstico final:",
            options=st.session_state.metrics_dict.keys(),
            key="model_select_tab3"
        )
        
        # Esta función ya no puede ser cacheada
        def get_final_forecast_for_model(_model_name, _segment_name):
            try:
                # 1. Cargar el DF completo de escenarios
                df_all_future_scenarios = st.session_state.loaded_drivers_for_forecast.copy()
                
                df_all_future_scenarios.columns = df_all_future_scenarios.columns.str.replace('_base', '_normal', regex=False)
                
                # 2. Correr Escenario NORMAL (Base)
                with st.spinner(f"Ejecutando escenario 'Normal' para {_model_name}..."):
                    df_forecast_normal, final_model, X_train_final, Y_train_final = run_final_forecast(
                        df_master_hist=st.session_state.df_master_hist_engine_final_filtered, 
                        segment_name=_segment_name,
                        df_future_drivers=df_all_future_scenarios, 
                        model_name=_model_name,
                        winner_rmse=st.session_state.winner_rmse_for_ml,
                        scenario_suffix='_normal' 
                    )
                
                if df_forecast_normal is None:
                    st.error("Falló la ejecución del escenario 'Normal'.")
                    return None, None, None, None

                # 3. Correr Escenario OPTIMISTA
                with st.spinner(f"Ejecutando escenario 'Optimista' para {_model_name}..."):
                    df_forecast_optimista, _, _, _ = run_final_forecast(
                        df_master_hist=st.session_state.df_master_hist_engine_final_filtered, 
                        segment_name=_segment_name,
                        df_future_drivers=df_all_future_scenarios, 
                        model_name=_model_name,
                        winner_rmse=st.session_state.winner_rmse_for_ml,
                        scenario_suffix='_optimista' 
                    )
                
                # 4. Correr Escenario PESIMISTA
                with st.spinner(f"Ejecutando escenario 'Pesimista' para {_model_name}..."):
                    df_forecast_pesimista, _, _, _ = run_final_forecast(
                        df_master_hist=st.session_state.df_master_hist_engine_final_filtered, 
                        segment_name=_segment_name,
                        df_future_drivers=df_all_future_scenarios, 
                        model_name=_model_name,
                        winner_rmse=st.session_state.winner_rmse_for_ml,
                        scenario_suffix='_pesimista' 
                    )

                # 5. Combinar resultados
                df_forecast_final = df_forecast_normal.rename(columns={
                    'Ventas_Proyectadas': 'Ventas_Normal',
                    'Ventas_Minimo': 'Ventas_Minimo',
                    'Ventas_Maximo': 'Ventas_Maximo'
                })
                
                if df_forecast_optimista is not None:
                    df_forecast_final['Ventas_Optimista'] = df_forecast_optimista['Ventas_Proyectadas']
                
                if df_forecast_pesimista is not None:
                    df_forecast_final['Ventas_Pesimista'] = df_forecast_pesimista['Ventas_Proyectadas']

                return df_forecast_final, final_model, X_train_final, Y_train_final
            
            except Exception as e:
                st.error(f"Error al generar los escenarios de pronóstico: {e}")
                st.exception(e)
                return None, None, None, None
        
        segmento_run = st.session_state.segmento_forecast_run
        
        spinner_placeholder = st.empty()
        with spinner_placeholder.status(f"Generando todos los escenarios para {model_to_display}...", expanded=True):
            df_forecast_final, final_model, X_train_final, Y_train_final = get_final_forecast_for_model(
                model_to_display, 
                segmento_run
            )
        spinner_placeholder.empty()


        if df_forecast_final is not None and not df_forecast_final.empty: 
            
            # --- 🔧 INICIO NUEVA SECCIÓN: OVERRIDE MANUAL (v6.1) ---
            st.subheader("5. Editor de Pronóstico (Override Manual)")
            st.warning("Usa esta tabla para aplicar tu 'Arte' y sobrescribir manualmente cualquier predicción del modelo (ej. subir Noviembre a 11,500).")
            
            edited_final_forecast = st.data_editor(
                df_forecast_final, 
                height=400, 
                key="final_forecast_editor",
                use_container_width=True
            )
            # --- FIN NUEVA SECCIÓN ---

            # --- 1. Datos Históricos (Usando el DF de GRÁFICO) ---
            # --- 🔧 FIX v6.4: Corregido el filtro para que solo muestre el segmento seleccionado ---
            # Nota: para el Total, el histórico se guardó como 'Total_Industria',
            # mientras que el segmento que selecciona el usuario es 'Total_Filtrado'.
            segment_for_graph = segmento_run
            if segmento_run in ["Total_Filtrado", "Total_Industria"]:
                # Si existe 'Total_Industria' en el DF, usamos ese para el gráfico
                segmentos_hist = st.session_state.df_master_hist_graph_final_filtered['segmento'].unique()
                if "Total_Industria" in segmentos_hist:
                    segment_for_graph = "Total_Industria"

            df_hist_chart = st.session_state.df_master_hist_graph_final_filtered[
                st.session_state.df_master_hist_graph_final_filtered['segmento'] == segment_for_graph
            ][['Ventas_Industria']].reset_index()

            df_hist_chart = df_hist_chart.rename(
                columns={'Ventas_Industria': 'Ventas', 'index': 'Fecha', 'fecha': 'Fecha'}
            )
            df_hist_chart['Tipo'] = 'Real'

            # --- 2. Datos del Backtest ---
            df_backtest_chart = st.session_state.backtest_forecasts_dict[model_to_display].reset_index()
            df_backtest_chart.columns = ['Fecha', 'Ventas']
            df_backtest_chart['Tipo'] = 'Backtest'

            # --- 3. Datos del Pronóstico Futuro (Usando el DF EDITADO) ---
            df_future_chart = edited_final_forecast.reset_index().rename(columns={'index': 'Fecha', 'fecha': 'Fecha'})

            df_future_plot = df_future_chart.melt(
                id_vars=['Fecha', 'Ventas_Minimo', 'Ventas_Maximo'],
                value_vars=['Ventas_Normal', 'Ventas_Optimista', 'Ventas_Pesimista'],
                var_name='Escenario',
                value_name='Ventas'
            )
            df_future_plot = df_future_plot.dropna(subset=['Ventas'])

            # --- 4. Unir Historial + Backtest ---
            df_full_chart = pd.concat([df_hist_chart, df_backtest_chart])
            df_full_chart['Fecha'] = pd.to_datetime(df_full_chart['Fecha'])

            # Usamos una copia limpia para el chart
            df_plot_final = df_full_chart.copy()

            st.subheader(f"Pronóstico de Ventas para [{segmento_run}] con {model_to_display}")

            # Fecha de inicio del forecast (primer mes proyectado)
            forecast_start = df_future_plot['Fecha'].min()

            # --- 5. Capas del gráfico ---

            # 5.1 Histórico real
            chart_real = (
                alt.Chart(df_plot_final[df_plot_final['Tipo'] == 'Real'])
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X('Fecha:T', title='Fecha'),
                    y=alt.Y('Ventas:Q', title=f'Ventas ({segmento_run})'),
                    color=alt.value('black'),
                    tooltip=[
                        alt.Tooltip('Fecha:T', title='Fecha'),
                        alt.Tooltip('Ventas:Q', title='Ventas reales', format=',.0f'),
                    ],
                )
            )

            # 5.2 Backtest (validación histórica)
            chart_backtest = (
                alt.Chart(df_plot_final[df_plot_final['Tipo'] == 'Backtest'])
                .mark_line(strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    x=alt.X('Fecha:T', title='Fecha'),
                    y=alt.Y('Ventas:Q', title=f'Ventas ({segmento_run})'),
                    color=alt.value('#1f77b4'),
                    tooltip=[
                        alt.Tooltip('Fecha:T', title='Fecha'),
                        alt.Tooltip('Ventas:Q', title='Backtest', format=',.0f'),
                    ],
                )
            )

            # 5.3 Escenarios futuros (Normal / Optimista / Pesimista)
            base_future = (
                alt.Chart(df_future_plot)
                .encode(
                    x=alt.X('Fecha:T', title='Fecha'),
                    tooltip=[
                        alt.Tooltip('Fecha:T', title='Fecha'),
                        alt.Tooltip('Escenario:N', title='Escenario'),
                        alt.Tooltip('Ventas:Q', title='Ventas', format=',.0f'),
                    ],
                )
            )

            color_scale = alt.Scale(
                domain=['Ventas_Normal', 'Ventas_Optimista', 'Ventas_Pesimista'],
                range=['#1f77b4', '#2ca02c', '#d62728'],
            )

            chart_scenarios = (
                base_future.mark_line(strokeWidth=2)
                .encode(
                    y=alt.Y('Ventas:Q', title=f'Ventas ({segmento_run})'),
                    color=alt.Color('Escenario:N', scale=color_scale, legend=alt.Legend(title='Escenarios')),
                )
            )

            # 5.4 Banda de incertidumbre (Min–Max)
            df_banda = df_future_chart[['Fecha', 'Ventas_Minimo', 'Ventas_Maximo']].copy()
            banda = (
                alt.Chart(df_banda)
                .mark_area(opacity=0.2)
                .encode(
                    x=alt.X('Fecha:T', title='Fecha'),
                    y=alt.Y('Ventas_Minimo:Q', title=f'Ventas ({segmento_run})'),
                    y2=alt.Y2('Ventas_Maximo:Q'),
                    tooltip=[
                        alt.Tooltip('Fecha:T', title='Fecha'),
                        alt.Tooltip('Ventas_Minimo:Q', title='Mínimo', format=',.0f'),
                        alt.Tooltip('Ventas_Maximo:Q', title='Máximo', format=',.0f'),
                    ],
                )
            )

            # 5.5 Línea vertical que marca el inicio del forecast
            cut_line = (
                alt.Chart(pd.DataFrame({'Fecha': [forecast_start]}))
                .mark_rule(strokeDash=[4, 4])
                .encode(x='Fecha:T')
            )

            # Dibujamos primero backtest y luego histórico, para que la línea real quede encima
            final_chart = (chart_backtest + chart_real + chart_scenarios + banda + cut_line).interactive()

            st.altair_chart(final_chart, use_container_width=True)

            st.divider()


            # --- 🔧 INICIO NUEVA SECCIÓN: ANÁLISIS YoY (v6.4) ---
            st.subheader("Análisis de Crecimiento (Año vs. Año)")
            st.markdown(
                "Comparación de las ventas anuales proyectadas (Escenario Normal) "
                "vs. las ventas reales de años anteriores. "
                "Para el año de inicio del forecast se suma **real + proyección**; "
                "los años posteriores son solo proyección."
            )

            try:
                # 1. Históricos del segmento (solo Total_Filtrado / segmento_run)
                # Histórico para el análisis anual.
                # Igual que en el gráfico, mapeamos Total_Filtrado -> Total_Industria.
                segment_for_graph = segmento_run
                if segmento_run in ["Total_Filtrado", "Total_Industria"]:
                    segmentos_hist = st.session_state.df_master_hist_graph_final_filtered['segmento'].unique()
                    if "Total_Industria" in segmentos_hist:
                        segment_for_graph = "Total_Industria"

                df_hist_full = (
                    st.session_state.df_master_hist_graph_final_filtered[
                        st.session_state.df_master_hist_graph_final_filtered['segmento'] == segment_for_graph
                    ][['Ventas_Industria']]
                    .reset_index()
                    .rename(columns={'Ventas_Industria': 'Ventas', 'index': 'Fecha', 'fecha': 'Fecha'})
                )

                if df_hist_full.empty:
                    st.warning("No hay datos históricos para este segmento.")
                else:
                    # 2. Proyección (Escenario Normal) usando la tabla editada
                    df_projected = (
                        edited_final_forecast[['Ventas_Normal']]
                        .reset_index()
                        .rename(columns={'Ventas_Normal': 'Ventas', 'index': 'Fecha', 'fecha': 'Fecha'})
                    )

                    if df_projected.empty:
                        st.warning("No hay datos de proyección para el análisis de crecimiento anual.")
                    else:
                        df_hist_full['Tipo'] = 'Real'
                        df_projected['Tipo'] = 'Proyectado'

                        df_all = pd.concat([df_hist_full, df_projected], ignore_index=True)
                        df_all['Fecha'] = pd.to_datetime(df_all['Fecha'])
                        df_all['Año'] = df_all['Fecha'].dt.year

                        # Fecha y año de inicio del forecast (primer mes proyectado)
                        forecast_start_date = df_projected['Fecha'].min()
                        forecast_start_year = forecast_start_date.year

                        # Último año real disponible (independiente del forecast)
                        years_real = sorted(df_hist_full['Fecha'].dt.year.unique())
                        last_real_year = years_real[-1]

                        # 3. Agregados anuales por tipo (Real / Proyectado)
                        agg = df_all.groupby(['Año', 'Tipo'], as_index=False)['Ventas'].sum()
                        pivot = agg.pivot(index='Año', columns='Tipo', values='Ventas').fillna(0.0)
                        pivot = pivot.sort_index()

                        # 4. Total anual mezclando real + proyección según el año
                        def compute_total(row):
                            year = row.name
                            real = row.get('Real', 0.0)
                            proy = row.get('Proyectado', 0.0)
                            if year < forecast_start_year:
                                return real
                            elif year == forecast_start_year:
                                # Año mixto: real + proyección (ej. 2025 = real + nov-dic proyectado)
                                return real + proy
                            else:
                                # Años completos solo proyectados (ej. 2026)
                                return proy

                        pivot['Total'] = pivot.apply(compute_total, axis=1)

                        # 5. Clasificar tipo de año para color
                        def tipo_anio(row):
                            year = row.name
                            if year < forecast_start_year:
                                return "Solo real"
                            elif year == forecast_start_year:
                                return "Real + proyección"
                            else:
                                return "Solo proyección"

                        pivot['Tipo_Año'] = pivot.apply(tipo_anio, axis=1)

                        # 6. Últimos 3 años reales + todos los años proyectados posteriores
                        ultimos_reales = years_real[-3:]
                        futuros = [y for y in pivot.index if y > last_real_year]
                        years_to_keep = sorted(set(ultimos_reales + futuros))
                        pivot = pivot.loc[years_to_keep]

                        # 7. Crecimiento vs año anterior usando el Total anual
                        pivot['Crecimiento %'] = pivot['Total'].pct_change()

                        # 8. Mostrar tabla
                        st.dataframe(
                            pivot[['Real', 'Proyectado', 'Total', 'Crecimiento %']]
                            .fillna(0.0)
                            .style.format(
                                {
                                    'Real': "{:,.0f}",
                                    'Proyectado': "{:,.0f}",
                                    'Total': "{:,.0f}",
                                    'Crecimiento %': "{:.1%}",
                                }
                            ),
                            use_container_width=True,
                        )

                        # 9. Gráfico de barras por año
                        df_plot_years = pivot.reset_index().rename(columns={'Año': 'Year'})
                        df_plot_years['YearLabel'] = df_plot_years['Year'].astype(str)

                        bars = (
                            alt.Chart(df_plot_years)
                            .mark_bar()
                            .encode(
                                x=alt.X('YearLabel:N', title='Año'),
                                y=alt.Y('Total:Q', title='Ventas anuales'),
                                color=alt.Color('Tipo_Año:N', title='Tipo de año'),
                                tooltip=[
                                    alt.Tooltip('Year:N', title='Año'),
                                    alt.Tooltip('Total:Q', title='Ventas anuales', format=',.0f'),
                                    alt.Tooltip(
                                        'Crecimiento %:Q',
                                        title='Crecimiento vs año anterior',
                                        format='.1%',
                                    ),
                                ],
                            )
                        )

                        labels = (
                            bars.mark_text(dy=-8, color='black')
                            .encode(text=alt.Text('Total:Q', format=',.0f'))
                        )

                        st.altair_chart(bars + labels, use_container_width=True)

            except Exception as e:
                st.error(f"No se pudo generar el gráfico de comparación Año vs Año: {e}")
                st.exception(e)


            # --- FIN NUEVA SECCIÓN ---
            
            st.divider()

            # --- 🔧 INICIO NUEVA SECCIÓN: GUARDAR PROYECCIÓN (v6.1) ---
            st.subheader("6. Guardar Proyección Final")
            
            col_save_1, col_save_2 = st.columns([2,1])
            with col_save_1:
                projection_name = st.text_input(
                    "Nombre para esta Proyección:",
                    placeholder=f"Forecast {segmento_run} - {model_to_display} - {pd.Timestamp.now().strftime('%Y-%m')}"
                )
            
            with col_save_2:
                st.write("") # Espaciador
                st.write("") # Espaciador
                if st.button("Guardar Proyección Final en BD", type="primary"):
                    if not projection_name:
                        st.error("Por favor, ingrese un nombre para la proyección.")
                    else:
                        with st.spinner("Guardando proyección..."):
                            try:
                                # 1. Usar los datos de la tabla EDITADA
                                df_to_save = edited_final_forecast.copy()
                                df_to_save = df_to_save[['Ventas_Normal', 'Ventas_Optimista', 'Ventas_Pesimista']]
                                
                                # 2. Fundir (Melt) de 'wide' a 'long'
                                df_melted = df_to_save.reset_index().rename(columns={'index':'fecha', 'fecha':'fecha'}).melt(
                                    id_vars='fecha',
                                    var_name='tipo_escenario',
                                    value_name='valor_proyectado'
                                )
                                
                                # 3. Limpiar nombres de escenario
                                df_melted['tipo_escenario'] = df_melted['tipo_escenario'].str.replace('Ventas_', '')
                                
                                # 4. Añadir las columnas de metadata
                                df_melted['projection_name'] = projection_name
                                df_melted['segmento_base'] = segmento_run
                                df_melted['modelo_usado'] = model_to_display
                                df_melted['scenario_name'] = scenario_to_load   # 👈 NUEVO: enlaza proyección con escenario
                                df_melted['fecha'] = pd.to_datetime(df_melted['fecha']).dt.date

                                # 5. Reordenar para la BD (incluyendo scenario_name)
                                df_final_insert = df_melted[[
                                    'projection_name', 'segmento_base', 'modelo_usado', 'scenario_name',
                                    'fecha', 'tipo_escenario', 'valor_proyectado'
                                ]]
                                
                                # 6. Insertar en la BD
                                with engine.connect() as conn:
                                    with conn.begin(): 
                                        # Borrar si ya existe (para permitir sobrescribir)
                                        sql_delete = text("DELETE FROM forecast_final_projections WHERE projection_name = :name")
                                        conn.execute(sql_delete, {"name": projection_name})
                                        
                                        df_final_insert.to_sql(
                                            'forecast_final_projections',
                                            conn,
                                            if_exists='append',
                                            index=False,
                                            chunksize=1000
                                        )
                                st.success(f"¡Proyección '{projection_name}' guardada exitosamente!")

                                # --- Nuevo: guardar métricas de backtest asociadas a esta proyección ---
                                cv_meta = st.session_state.get("cv_metadata_for_ml")
                                metrics_dict = st.session_state.get("metrics_dict")

                                if cv_meta is None or metrics_dict is None:
                                    st.warning(
                                        "La proyección se guardó, pero no se encontraron métricas de backtest "
                                        "en memoria para registrar en forecast_runs / forecast_model_metrics."
                                    )
                                else:
                                    try:
                                        # Horizonte = número de meses proyectados (mismo índice que los drivers futuros)
                                        horizon_months = st.session_state.loaded_drivers_for_forecast.index.nunique()

                                        forecasting_engine.save_backtest_results_to_db(
                                            engine=engine,
                                            projection_name=projection_name,           # 👈 mismo nombre que guardas en forecast_final_projections
                                            scope=segmento_run,                        # ej. 'Total_Filtrado' o segmento concreto
                                            horizon_months=horizon_months,
                                            train_start=cv_meta["train_start"],
                                            train_end=cv_meta["train_end"],
                                            test_start=cv_meta["test_start"],
                                            test_end=cv_meta["test_end"],
                                            leaderboard_dict=metrics_dict,             # dict con MAPE/MAE/RMSE por modelo
                                        )
                                        st.info("Métricas de backtest asociadas a esta proyección guardadas en forecast_runs/forecast_model_metrics.")
                                    except Exception as e_metrics:
                                        st.error(
                                            f"La proyección se guardó, pero hubo un error al guardar las métricas de modelo: {e_metrics}"
                                        )
                                        st.exception(e_metrics)

                            except Exception as e:
                                st.error(f"Error al guardar: {e}")
                                st.exception(e)
            # --- 🔧 FIN NUEVA SECCIÓN ---
            
            with st.expander("Ver Datos de Proyección y Drivers (Debug)", expanded=False):
                col_data_1, col_data_2 = st.columns(2)
                
                with col_data_1:
                    st.markdown(f"##### Proyección de Ventas (Editada)")
                    st.dataframe(edited_final_forecast, height=300) 
                    
                    @st.cache_data
                    def convert_df_to_csv_indexed(df):
                        return df.to_csv(index=True).encode('utf-8')
                    csv = convert_df_to_csv_indexed(edited_final_forecast) 
                    st.download_button(
                        label=f"Descargar Forecast ({model_to_display})",
                        data=csv, file_name=f"forecast_{segmento_run}_{model_to_display}_all_scenarios.csv", mime="text/csv"
                    )

                with col_data_2:
                    st.markdown("##### Drivers Usados (Escenario 'normal')")
                    df_drivers_display = st.session_state.loaded_drivers_for_forecast.filter(like="_normal")
                    if df_drivers_display.empty: 
                         df_drivers_display = st.session_state.loaded_drivers_for_forecast.filter(like="_base")
                         df_drivers_display.columns = df_drivers_display.columns.str.replace('_base', '', regex=False)
                    else:
                         df_drivers_display.columns = df_drivers_display.columns.str.replace('_normal', '', regex=False)
                    st.dataframe(df_drivers_display, height=300)
        
        elif df_forecast_final is not None and df_forecast_final.empty:
             st.error(f"El modelo {model_to_display} no pudo generar un pronóstico. (data_to_forecast estaba vacío).")

        if 'run_forecast' in st.session_state:
             del st.session_state.run_forecast