import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import base64
from sqlalchemy import create_engine, text
from datetime import datetime
from sklearn.linear_model import LinearRegression
from math import isnan
from auth import login_user  # Tu función de login
from db import get_engine
from sidebar import render_sidebar


engine = get_engine()

# =============================
# CATÁLOGO DE DRIVERS (Ficha Técnica)
# =============================

def normalize_driver_key(v: str) -> str:
    """Normaliza el nombre de variable para matchear con el catálogo."""
    if v is None:
        return ""
    k = str(v).strip().upper()
    k = k.replace(" ", "_").replace(".", "_")
    return k

DRIVER_CATALOG = {
    "IVA": {
        "tipo": "Fiscal / Demanda agregada",
        "porque": "Proxy de consumo formal y actividad comercial; suele moverse con ventas de bienes durables y servicios asociados (talleres, repuestos).",
        "fuente": "SRI",
    },
    "WTI": {
        "tipo": "Commodities / Energía",
        "porque": "Afecta costos logísticos y expectativas macro; en Ecuador también incide en ingresos petroleros → gasto/actividad y liquidez.",
        "fuente": "https://es.investing.com/commodities/crude-oil",
    },
    "euro_usd": {
        "tipo": "Tipo de cambio externo",
        "porque": "Influye en el costo de importación desde Europa (vehículos/partes) y en precios relativos de proveedores.",
        "fuente": "https://es.investing.com/currencies/eur-usd",
    },
    "PVP_ORO_1_OZ": {
        "tipo": "Commodities / Financiero",
        "porque": "Indicador de aversión al riesgo global; cuando sube fuerte suele coincidir con menor apetito por crédito/consumo durable.",
        "fuente": "https://es.investing.com/commodities/gold",
    },
    "PIB_DEM_CAD_BRU": {
        "tipo": "Actividad real",
        "porque": "Captura el ciclo económico: en expansión suben ingresos, empleo y confianza → más compras de vehículos.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/cuentasnacionales/ix_cuentasnacionalestrimestrales.html",
    },
    "DEMINT_CORR_AJUS": {
        "tipo": "Demanda interna",
        "porque": "Mide el pulso de consumo + inversión doméstica, que son motores directos de ventas de autos (particulares y flotas).",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/cuentasnacionales/ix_cuentasnacionalestrimestrales.html",
    },
    "PIB_CAD_AJUS": {
        "tipo": "Actividad real",
        "porque": "Serie “limpia” (ajustada) para modelar tendencia/ciclo sin ruido estacional; mejora forecast de demanda automotriz.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/cuentasnacionales/ix_cuentasnacionalestrimestrales.html",
    },
    "ING_PETROLEO": {
        "tipo": "Fiscal / Petrolero",
        "porque": "Más ingresos petroleros suelen traducirse en más gasto público/liquidez, mejorando empleo y consumo, impactando ventas.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/cuentasnacionales/ix_cuentasnacionalestrimestrales.html",
    },
    "PETRO_PROD": {
        "tipo": "Real / Petrolero",
        "porque": "Anticipa ingresos del sector y dinamismo macro; también incide en transporte/actividad productiva y consumo.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/cuentasnacionales/ix_cuentasnacionalestrimestrales.html",
    },
    "IPC": {
        "tipo": "Precios",
        "porque": "Inflación alta erosiona poder adquisitivo, encarece operación (servicios) y puede frenar compras de durables.",
        "fuente": "https://app.powerbi.com/view?r=eyJrIjoiMjM5MmZiYTUtYWJhOC00ZjQ3LTg3YWYtM2I4YTBhMmRiYmRhIiwidCI6ImYxNThhMmU4LWNhZWMtNDQwNi1iMGFiLWY1ZTI1OWJkYTExMiJ9",
    },
    "INFLACIONMENSUAL": {
        "tipo": "Precios (derivada)",
        "porque": "Detecta shocks recientes de precios; útil para ajustar timing del consumo y sensibilidad al financiamiento.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/indicadores/real/Inflacion.html",
    },
    "CREDITOSECTORPRIVADO": {
        "tipo": "Financiera / Crédito",
        "porque": "La compra de autos depende mucho del financiamiento; más crédito disponible suele elevar ventas.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_MonetariasFinancierasPrin.html",
    },
    "DEPOSITOSALAVISTA": {
        "tipo": "Liquidez",
        "porque": "Indica liquidez del sistema y capacidad de colocación de crédito; también refleja confianza/ingreso corriente.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_MonetariasFinancierasPrin.html",
    },
    "ICC": {
        "tipo": "Expectativas",
        "porque": "La decisión de comprar auto es sensible a confianza; ICC captura disposición a gastar y percepción de futuro.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/indicadores/real/IndiceConfianzaConsumidor.html",
    },
    "EMPLEOADECUADOPLENO": {
        "tipo": "Laboral",
        "porque": "Mejor empleo → más ingreso estable → mayor aprobación de crédito y más demanda de vehículos.",
        "fuente": "https://app.powerbi.com/view?r=eyJrIjoiNDY3MjljMTYtNGI5Yy00ZWM4LWE4OTYtNjlhYWYxYzcwZTAxIiwidCI6ImYxNThhMmU4LWNhZWMtNDQwNi1iMGFiLWY1ZTI1OWJkYTExMiJ9",
    },
    "IMP_CBU": {
        "tipo": "Comercio exterior",
        "porque": "Vehículos suelen llegar como CBU; es un leading indicator de oferta/ventas futuras (importas hoy, vendes después).",
        "fuente": "Veritrade",
    },
    "RIESGO_PAIS": {
        "tipo": "Riesgo / Mercado",
        "porque": "Afecta costo de fondeo, tasas, disponibilidad de crédito y confianza; alto riesgo suele frenar inversión/consumo durable.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/PublicacionesGenerales/ix_PublicacionesGeneralesPrin.html",
    },
    "TASA_DE_INTERES_ACTIVA": {
        "tipo": "Tasas",
        "porque": "Determina cuota y costo total de financiamiento; subidas de tasa reducen demanda de vehículos a crédito.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_MonetariasFinancierasPrin.html",
    },
    "CARTERA_CREDITOS_DE_CONSUMO": {
        "tipo": "Crédito (segmento)",
        "porque": "Proxy directo de dinámica de crédito al hogar; correlaciona con ventas de autos y motos.",
        "fuente": "https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_MonetariasFinancierasPrin.html",
    },
    "UTILIDADES": {"tipo": "Resultados / Ingreso", "porque": "(Según definición) utilidades sectoriales o financieras reflejan capacidad de inversión (flotas) y solvencia del sistema para prestar.", "fuente": "Dummy"},
    "PARO": {"tipo": "Laboral", "porque": "Mayor desempleo reduce intención de compra y aumenta riesgo crediticio → menor colocación de créditos de auto.", "fuente": "Dummy"},
    "CUP_IMPORT": {"tipo": "Comercio exterior (costo)", "porque": "Aproxima costo unitario importado; subidas anticipan alzas de precios de vehículos/repuestos y posibles caídas de volumen.", "fuente": "Dummy"},
    "ELEC_PRESIDENCIALES": {"tipo": "Evento / Política", "porque": "Periodos electorales suelen mover expectativas e inversión; puede haber postergación de compras o shocks de demanda.", "fuente": "Dummy"},
    "CAMBIO_IVA": {"tipo": "Política fiscal", "porque": "Cambios de IVA alteran el precio final (vehículo, servicios y repuestos) y pueden generar adelanto/atraso de compras.", "fuente": "Dummy"},
    "DUMMY_COVID_LOCKDOWN": {"tipo": "Shock exógeno", "porque": "Captura quiebres estructurales: cierres reducen ventas, importaciones, empleo y crédito; evita que el modelo “aprenda mal”.", "fuente": "Dummy"},
}

# =============================
# CONFIGURACIÓN INICIAL
# =============================

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False
if 'email' not in st.session_state:
    st.session_state['email'] = ""
if 'df_desagregado' not in st.session_state:
    st.session_state['df_desagregado'] = None
if 'meses_share_modelo' not in st.session_state:
    st.session_state['meses_share_modelo'] = 3
if 'selected_forecast_name' not in st.session_state:
    st.session_state['selected_forecast_name'] = None
if 'df_total_forecast' not in st.session_state:
    st.session_state['df_total_forecast'] = None

# Config de página según login
if not st.session_state['logged_in']:
    st.set_page_config(layout="centered", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    
# Lista de dummies para clasificar drivers en la ficha técnica
DUMMY_VARS_LIST = [
    'CUP_IMPORT', 'CAMBIO_IVA', 'ELEC_PRESIDENCIALES', 'PARO', 'UTILIDADES',
    'DUMMY_COVID_LOCKDOWN'
]


# Descripciones y fuentes sugeridas para algunos drivers macro
DRIVER_DESCRIPTIONS = {
    "PIB": "Crecimiento real del Producto Interno Bruto de Ecuador.",
    "RIESGO_PAIS": "Índice de riesgo país (spread EMBI u otro índice equivalente).",
    "TASA_DESEMPLEO": "Tasa de desempleo nacional.",
    "TC_PROMEDIO": "Tipo de cambio promedio utilizado en las importaciones.",
    "TASA_INTERES": "Tasa de interés referencial / crédito de consumo.",
}

DRIVER_SOURCES = {
    "PIB": "Banco Central del Ecuador (BCE).",
    "RIESGO_PAIS": "Fuentes internacionales (p. ej. JP Morgan EMBI) procesadas por Maresa.",
    "TASA_DESEMPLEO": "INEC / publicaciones oficiales.",
    "TC_PROMEDIO": "BCE / sistema financiero.",
    "TASA_INTERES": "BCE / Superintendencia de Bancos.",
}


@st.cache_resource
def get_db_engine():
    try:
        engine = get_engine()
        return engine
    except Exception as e:
        st.error(f"Error al crear conexión a BD: {e}")
        st.stop()


@st.cache_data(ttl=600)
def load_all_forecast_names():
    """Nombres de proyecciones guardadas en forecast_final_projections."""
    try:
        engine = get_db_engine()
        sql = """
            SELECT projection_name, MAX(created_at) as last_created 
            FROM forecast_final_projections 
            GROUP BY projection_name 
            ORDER BY last_created DESC
        """
        df = pd.read_sql(sql, engine)
        return df['projection_name'].tolist()
    except Exception as e:
        st.error(f"Error al cargar nombres de forecast: {e}")
        return []


@st.cache_data(ttl=600)
def load_selected_forecast(projection_name):
    """
    Carga la proyección TOTAL (industria) de forecast_final_projections para un projection_name.

    Devuelve un DF pivot:
      index = fecha
      columns = ['Ventas_Normal', 'Ventas_Optimista', 'Ventas_Pesimista']
    """
    try:
        engine = get_db_engine()
        sql = text("""
            SELECT fecha, tipo_escenario, valor_proyectado 
            FROM forecast_final_projections 
            WHERE projection_name = :name
        """)
        df = pd.read_sql(sql, engine, params={"name": projection_name}, parse_dates=['fecha'])

        df_pivot = df.pivot_table(
            index='fecha',
            columns='tipo_escenario',
            values='valor_proyectado'
        ).rename(columns={
            'Normal': 'Ventas_Normal',
            'Optimista': 'Ventas_Optimista',
            'Pesimista': 'Ventas_Pesimista'
        })
        return df_pivot
    except Exception as e:
        st.error(f"Error al cargar forecast seleccionado: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_all_granular_sales():
    """
    Carga ventas históricas granulares desde sales_granular.
    Agrupa a nivel (mes, segmento, marca, modelo, provincia).
    """
    try:
        engine = get_db_engine()
        sql = """
            SELECT 
                date_trunc('month', fecha_proceso)::date as "Fecha",
                segmento,
                marca,
                familia,
                modelo,
                provincia,
                SUM(unidades) as "Unidades"
            FROM sales_granular
            GROUP BY 1, 2, 3, 4, 5, 6
        """
        df = pd.read_sql(sql, engine, parse_dates=['Fecha'])
        return df
    except Exception as e:
        st.error(f"Error al cargar ventas granulares: {e}")
        return pd.DataFrame()
    
# =============================
# FUNCIONES PARA FICHA TÉCNICA
# =============================

@st.cache_data
def load_projection_catalog(_engine):
    """
    Catálogo de proyecciones guardadas:
    una fila por (projection_name, segmento_base, modelo_usado),
    usando el scenario_name NO NULO cuando exista.
    """
    query = text("""
        SELECT 
            projection_name,
            segmento_base,
            modelo_usado,
            MAX(scenario_name) AS scenario_name
        FROM forecast_final_projections
        GROUP BY projection_name, segmento_base, modelo_usado
        ORDER BY projection_name
    """)
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df



@st.cache_data(ttl=600)
def load_projection_metrics(_engine, projection_name):
    """
    Devuelve:
      - df_run: una fila de forecast_runs (scope, best_model, etc.)
      - df_metrics: tabla de métricas por modelo desde forecast_model_metrics
    """
    with _engine.connect() as conn:
        # Resumen de la corrida (forecast_runs)
        df_run = pd.read_sql(
            text("""
                SELECT projection_name, scope, horizon_months,
                       train_start, train_end, test_start, test_end,
                       best_model, best_mape, best_mae, best_rmse
                FROM forecast_runs
                WHERE projection_name = :p
                ORDER BY train_start DESC
                LIMIT 1
            """),
            conn,
            params={"p": projection_name},
            parse_dates=['train_start', 'train_end', 'test_start', 'test_end']
        )

        # Leaderboard por modelo (forecast_model_metrics) -> AQUÍ NO hay 'scope'
        df_metrics = pd.read_sql(
            text("""
                SELECT projection_name, model_name, mape, mae, rmse
                FROM forecast_model_metrics
                WHERE projection_name = :p
                ORDER BY mape ASC
            """),
            conn,
            params={"p": projection_name},
        )

    return df_run, df_metrics


@st.cache_data
def load_projection_variables(_engine, scenario_name: str):
    """
    Carga las variables (drivers) usadas en el escenario de drivers asociado
    a la proyección.
    """
    if not scenario_name or (isinstance(scenario_name, float) and np.isnan(scenario_name)):
        return pd.DataFrame()

    with _engine.connect() as conn:
        df_vars = pd.read_sql(
            text("""
                SELECT DISTINCT variable
                FROM forecast_driver_scenarios
                WHERE scenario_name = :s
                ORDER BY variable
            """),
            conn,
            params={"s": scenario_name},
        )
    return df_vars


@st.cache_data(ttl=600)
def load_historical_drivers_for_ft(_engine):
    """Carga la tabla historical_drivers para la ficha técnica."""
    with _engine.connect() as conn:
        try:
            df_hist = pd.read_sql("SELECT * FROM historical_drivers", conn, parse_dates=["Fecha"])
        except Exception:
            # Si la tabla no existe o tiene un esquema distinto, devolvemos DF vacío
            return pd.DataFrame()
    return df_hist


@st.cache_data(ttl=600)
def build_driver_summary_for_scenario(_engine, scenario_name: str):
    """Devuelve un resumen por variable con último valor real, último valor proyectado
    y variación porcentual para el escenario indicado."""
    if not scenario_name or (isinstance(scenario_name, float) and np.isnan(scenario_name)):
        return pd.DataFrame()

    # Variables del escenario (lista base)
    df_vars = load_projection_variables(_engine, scenario_name)
    if df_vars.empty:
        return df_vars

    # Histórico de drivers (ancho -> largo)
    df_hist = load_historical_drivers_for_ft(_engine)
    if df_hist.empty or "Fecha" not in df_hist.columns:
        df_vars["valor_real"] = np.nan
        df_vars["valor_proyectado"] = np.nan
        df_vars["Variacion_pct"] = np.nan
        df_vars["Descripcion"] = df_vars["variable"].apply(
            lambda v: DRIVER_DESCRIPTIONS.get(str(v).upper(), "Driver macro / de negocio.")
        )
        df_vars["Fuente"] = df_vars["variable"].apply(
            lambda v: DRIVER_SOURCES.get(str(v).upper(), "Maresa / fuentes internas y públicas.")
        )
        return df_vars

    hist_melt = df_hist.melt(id_vars="Fecha", var_name="variable", value_name="valor_real")
    # Último valor real por variable
    idx_hr = hist_melt.groupby("variable")["Fecha"].idxmax()
    hist_latest = hist_melt.loc[idx_hr, ["variable", "valor_real"]]

    # Proyecciones del escenario (tomamos escenario = 'normal' para referencia)
    with _engine.connect() as conn:
        df_proj_long = pd.read_sql(
            text(
                """
                SELECT fecha, variable, escenario, valor_proyectado
                FROM forecast_driver_scenarios
                WHERE scenario_name = :s
                """
            ),
            conn,
            params={"s": scenario_name},
            parse_dates=["fecha"],
        )

    df_proj_normal = df_proj_long[df_proj_long["escenario"].str.lower() == "normal"].copy()
    if df_proj_normal.empty:
        proj_latest = pd.DataFrame(columns=["variable", "valor_proyectado"])
    else:
        idx_p = df_proj_normal.groupby("variable")["fecha"].idxmax()
        proj_latest = df_proj_normal.loc[idx_p, ["variable", "valor_proyectado"]]

    df = df_vars.merge(hist_latest, on="variable", how="left").merge(
        proj_latest, on="variable", how="left"
    )

    # Enriquecimiento con catálogo (Tipo / Por qué / Fuente sugerida)
    def _get_cat(v, field, default):
        k = normalize_driver_key(v)
        return DRIVER_CATALOG.get(k, {}).get(field, default)

    df["Tipo"] = df["variable"].apply(lambda v: _get_cat(v, "tipo", "Driver macro / de negocio"))
    df["Por_que"] = df["variable"].apply(lambda v: _get_cat(v, "porque", "Driver relevante para el forecast automotriz."))
    df["Fuente_sugerida"] = df["variable"].apply(lambda v: _get_cat(v, "fuente", "Maresa / fuentes internas y públicas."))

    # (Opcional) si ya usas "Descripcion" y "Fuente" en la UI, puedes mapear:
    df["Descripcion"] = df["Por_que"]
    df["Fuente"] = df["Fuente_sugerida"]

    # Variación porcentual entre el último valor real y el proyectado
    df["Variacion_pct"] = np.where(
        df["valor_real"].notna()
        & (df["valor_real"] != 0)
        & df["valor_proyectado"].notna(),
        (df["valor_proyectado"] / df["valor_real"] - 1) * 100.0,
        np.nan,
    )

    return df


# =============================
# DESAGREGACIÓN JERÁRQUICA
# =============================

@st.cache_data
def calculate_desagregado(df_total_forecast, df_real_granular, meses_share=3):
    """
    Desagrega el forecast total por segmento/marca/modelo/provincia (top-down):

      - Share de segmento: usando TODO el histórico antes del inicio del forecast.
      - Share de modelo dentro de segmento: usando solo los últimos `meses_share` meses reales.

    df_total_forecast:
        index = fecha, columnas ['Ventas_Normal','Ventas_Optimista','Ventas_Pesimista']
    df_real_granular:
        columnas ['Fecha','segmento','marca','modelo','provincia','Unidades']
    """
    if df_total_forecast.empty or df_real_granular.empty:
        return pd.DataFrame()

    forecast_start_date = df_total_forecast.index.min()
    df_hist_for_share = df_real_granular[df_real_granular['Fecha'] < forecast_start_date].copy()

    if df_hist_for_share.empty:
        st.error("No hay histórico antes del inicio del forecast para calcular shares.")
        return pd.DataFrame()

    last_real_date = df_hist_for_share['Fecha'].max()

    # 1) Share por segmento (toda la historia)
    seg_hist = (
        df_hist_for_share
        .groupby('segmento')['Unidades']
        .sum()
        .rename('Unidades_hist_segmento')
        .reset_index()
    )

    total_hist = seg_hist['Unidades_hist_segmento'].sum()
    if total_hist == 0:
        st.error("Histórico vacío para share por segmento.")
        return pd.DataFrame()

    seg_hist['Share_segmento_hist'] = seg_hist['Unidades_hist_segmento'] / total_hist

    # 2) Share por modelo últimos N meses
    share_start_date = last_real_date - pd.DateOffset(months=meses_share - 1)
    df_share_period = df_hist_for_share[
        (df_hist_for_share['Fecha'] >= share_start_date) &
        (df_hist_for_share['Fecha'] <= last_real_date)
    ].copy()

    if df_share_period.empty:
        st.error(
            f"No se encontraron ventas en los últimos {meses_share} meses "
            f"para calcular el share por modelo."
        )
        return pd.DataFrame()

    granular_cols = ['segmento', 'marca', 'modelo', 'provincia']

    model_3m = (
        df_share_period
        .groupby(granular_cols)['Unidades']
        .sum()
        .rename('Unidades_3m_modelo')
        .reset_index()
    )

    seg_3m = (
        df_share_period
        .groupby('segmento')['Unidades']
        .sum()
        .rename('Unidades_3m_segmento')
        .reset_index()
    )

    model_3m = model_3m.merge(seg_3m, on='segmento', how='left')
    model_3m['Share_model_3m'] = model_3m['Unidades_3m_modelo'] / model_3m['Unidades_3m_segmento']
    model_3m['Share_model_3m'] = model_3m['Share_model_3m'].fillna(0).clip(lower=0)
    model_3m['sum_seg'] = model_3m.groupby('segmento')['Share_model_3m'].transform('sum')
    model_3m['Share_model_3m_norm'] = np.where(
        model_3m['sum_seg'] > 0,
        model_3m['Share_model_3m'] / model_3m['sum_seg'],
        0
    )

    df_shares = model_3m.merge(
        seg_hist[['segmento', 'Share_segmento_hist']],
        on='segmento',
        how='left'
    )
    df_shares['Share'] = df_shares['Share_segmento_hist'] * df_shares['Share_model_3m_norm']
    df_shares = df_shares[granular_cols + ['Share']]

    # 3) Expandir forecast total x shares
    df_shares['key'] = 1
    df_total_forecast_temp = df_total_forecast.reset_index()
    df_total_forecast_temp['key'] = 1

    df_mega = df_total_forecast_temp.merge(df_shares, on='key').drop(columns=['key'])

    # Map columnas a nombres de escenario
    escenarios_map = {
        'Ventas_Normal': 'Normal',
        'Ventas_Optimista': 'Optimista',
        'Ventas_Pesimista': 'Pesimista'
    }
    df_mega = df_mega.rename(columns=escenarios_map)
    escenarios_presentes = [v for v in escenarios_map.values() if v in df_mega.columns]

    dfs_final = []
    for esc in escenarios_presentes:
        df_esc = df_mega.copy()
        df_esc['Unidades_Decimal'] = df_esc[esc] * df_esc['Share']
        df_esc['Unidades'] = np.round(df_esc['Unidades_Decimal'], 0)

        # Ajuste de redondeo por mes
        df_esc_list = []
        for f in df_esc['fecha'].unique():
            df_mes = df_esc[df_esc['fecha'] == f].copy()
            total_proy = df_mes[esc].iloc[0]
            total_red = df_mes['Unidades'].sum()
            dif = total_proy - total_red
            if dif != 0 and not df_mes.empty:
                idx_max = df_mes['Unidades_Decimal'].idxmax()
                df_mes.loc[idx_max, 'Unidades'] += dif
            df_esc_list.append(df_mes)

        if not df_esc_list:
            continue

        df_esc_final = pd.concat(df_esc_list)
        df_esc_final['Escenario'] = esc
        df_esc_final = df_esc_final.drop(
            columns=[c for c in escenarios_presentes if c in df_esc_final.columns] +
                    ['Share', 'Unidades_Decimal']
        )
        dfs_final.append(df_esc_final)

    if not dfs_final:
        return pd.DataFrame()

    df_desagregado = pd.concat(dfs_final, ignore_index=True)
    df_desagregado = df_desagregado.rename(columns={'fecha': 'Fecha'})
    df_desagregado['Tipo'] = 'Proyectado'
    return df_desagregado


# =============================
# LOGIN
# =============================

if not st.session_state['logged_in']:
    st.title("Bienvenido a la Plataforma de Forecast")

    with st.form("login_form"):
        st.subheader("Inicio de Sesión")
        email = st.text_input("Correo Electrónico (@corpmaresa.com.ec)")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar")

        if submitted:
            success, message, is_admin = login_user(email, password)
            if success:
                st.session_state['logged_in'] = True
                st.session_state['is_admin'] = is_admin
                st.session_state['email'] = email
                st.rerun()
            else:
                st.error(message)

    st.info("¿No tienes cuenta? Regístrate aquí:")

    if st.button("Ir a Registro"):
        st.switch_page("pages/3_Register.py")

    st.stop()


# =============================
# APP PRINCIPAL (LOGUEADO)
# =============================

render_sidebar()

engine = get_db_engine()
st.title("📈 Forecast Industria Automotriz – Ecuador")

# 1) Cargar datos básicos
with st.spinner("Cargando datos..."):
    forecast_names = load_all_forecast_names()
    df_real_granular = load_all_granular_sales()

if not forecast_names:
    st.error("No hay proyecciones guardadas en forecast_final_projections.")
    st.stop()

if df_real_granular.empty:
    st.error("No hay ventas históricas en sales_granular.")
    st.stop()

# 2) Seleccionar Proyección Maestra (esto aplica a TODA la app)
st.subheader("Selección de Proyección Maestra")
col_f1, col_f2 = st.columns([2, 3])
with col_f1:
    selected_forecast_name = st.selectbox(
        "Proyección guardada:",
        options=forecast_names,
        index=0
    )
    st.session_state['selected_forecast_name'] = selected_forecast_name

# Cargar forecast total (industria) para esa proyección
df_total_forecast = load_selected_forecast(selected_forecast_name)
st.session_state['df_total_forecast'] = df_total_forecast

if df_total_forecast.empty:
    st.error("La proyección seleccionada no tiene datos.")
    st.stop()

# Preparamos DF de forecast melt para facilidad
df_proy_total = df_total_forecast.reset_index().melt(
    id_vars='fecha',
    value_vars=['Ventas_Normal', 'Ventas_Optimista', 'Ventas_Pesimista'],
    var_name='Escenario_col',
    value_name='Unidades'
)
df_proy_total['Escenario'] = df_proy_total['Escenario_col'].map({
    'Ventas_Normal': 'Normal',
    'Ventas_Optimista': 'Optimista',
    'Ventas_Pesimista': 'Pesimista'
})
df_proy_total = df_proy_total.drop(columns=['Escenario_col'])
df_proy_total = df_proy_total.rename(columns={'fecha': 'Fecha'})

# Real total por fecha
df_real_total = df_real_granular.groupby('Fecha')['Unidades'].sum().reset_index()

# =============================
# TABS PRINCIPALES
# =============================

tab_overview, tab_modelos = st.tabs([
    "Visión General Proyectada",
    "Detalle por modelo / versión"
])


# -------------------------------------------------
# TAB 1: VISIÓN GENERAL PROYECTADA
# -------------------------------------------------
with tab_overview:
    st.header("Visión General Proyectada")

    # Filtro de escenario SOLO para esta pestaña
    escenarios_disponibles = sorted(df_proy_total['Escenario'].unique())
    escenario_overview = st.radio(
        "Escenario para análisis (solo afecta algunas gráficas):",
        options=escenarios_disponibles,
        index=0,
        horizontal=True
    )

    # 1) FICHA TÉCNICA DE LA PROYECCIÓN
    with st.expander("📑 Ficha Técnica de la Proyección", expanded=False):

        df_proj_catalog = load_projection_catalog(engine)

        if df_proj_catalog.empty:
            st.info("No se encontraron proyecciones guardadas en la tabla 'forecast_final_projections'.")
        else:
            # Usar directamente la proyección maestra seleccionada arriba
            selected_projection = st.session_state.get("selected_forecast_name")
            df_sel = df_proj_catalog[df_proj_catalog["projection_name"] == selected_projection]

            if df_sel.empty:
                st.warning(
                    "La proyección seleccionada en la parte superior no tiene ficha técnica registrada. "
                    "Se mostrará la primera proyección disponible."
                )
                row_sel = df_proj_catalog.iloc[0]
            else:
                row_sel = df_sel.iloc[0]

            projection_name_ft = row_sel["projection_name"]
            segmento_base_ft_raw = row_sel["segmento_base"]
            # Mostrar 'Total Industria' en lugar de 'Total_Filtrado'
            segmento_base_ft = "Total Industria" if segmento_base_ft_raw == "Total_Filtrado" else segmento_base_ft_raw
            modelo_usado_ft = row_sel["modelo_usado"]
            scenario_name_ft = row_sel["scenario_name"]

            col_info_1, col_info_2 = st.columns([0.6, 0.4])
            with col_info_1:
                st.markdown(f"**Proyección seleccionada:** `{projection_name_ft}`")
                st.markdown(f"- Segmento base: **{segmento_base_ft}**")
                st.markdown(f"- Modelo usado en la proyección final: **{modelo_usado_ft}**")

                if scenario_name_ft is None or (isinstance(scenario_name_ft, float) and np.isnan(scenario_name_ft)):
                    st.markdown("- Escenario de drivers: `No asociado (proyección antigua)`")
                else:
                    st.markdown(f"- Escenario de drivers: `{scenario_name_ft}`")

            # Variables (drivers) utilizados
            st.markdown("### 🔧 Variables (Drivers) utilizados en el escenario")

            if scenario_name_ft is None or (isinstance(scenario_name_ft, float) and np.isnan(scenario_name_ft)):
                st.info(
                    "Esta proyección no tiene un escenario de drivers asociado (scenario_name = NULL). "
                    "Probablemente fue generada antes de actualizar el flujo. "
                    "Si quieres ver la ficha completa, vuelve a correr el forecast y guarda la proyección de nuevo."
                )
            else:
                df_vars_summary = build_driver_summary_for_scenario(engine, scenario_name_ft)

                if df_vars_summary.empty:
                    st.warning("No se encontraron variables asociadas a este escenario de drivers.")
                else:
                    def _cat(v, field, default="-"):
                        k = normalize_driver_key(v)
                        return DRIVER_CATALOG.get(k, {}).get(field, default)

                    df_vars_summary["Tipo"] = df_vars_summary["variable"].apply(lambda v: _cat(v, "tipo", "Driver macro / de negocio"))
                    df_vars_summary["Por_que"] = df_vars_summary["variable"].apply(lambda v: _cat(v, "porque", "Driver relevante para forecast automotriz."))
                    df_vars_summary["Fuente_sugerida"] = df_vars_summary["variable"].apply(lambda v: _cat(v, "fuente", "Dummy"))


                    df_display = df_vars_summary.copy()
                    df_display = df_display.rename(
                        columns={
                            "variable": "Variable",
                            "Por_que": "Por qué se considera para forecast automotriz",
                            "valor_real": "Último valor real",
                            "valor_proyectado": "Último valor proyectado",
                            "Variacion_pct": "Variación %",
                            "Fuente_sugerida": "Fuente sugerida",
                        }
                    )

                    # Formateos (miles con punto y % con 1 decimal)
                    def fmt_miles(x):
                        if pd.isna(x):
                            return "-"
                        try:
                            x = float(x)
                        except Exception:
                            return str(x)
                        # 22.000 (punto miles, sin decimales)
                        return f"{x:,.0f}".replace(",", ".")

                    def fmt_pct(x):
                        if pd.isna(x):
                            return "-"
                        try:
                            x = float(x)
                        except Exception:
                            return str(x)
                        # 1 decimal máximo en porcentaje
                        return f"{x:.1f}%".replace(".", ",")

                    if "Último valor real" in df_display.columns:
                        df_display["Último valor real"] = df_display["Último valor real"].apply(fmt_miles)

                    if "Último valor proyectado" in df_display.columns:
                        df_display["Último valor proyectado"] = df_display["Último valor proyectado"].apply(fmt_miles)

                    if "Variación %" in df_display.columns:
                        df_display["Variación %"] = df_display["Variación %"].apply(fmt_pct)


                    columnas_orden = [
                        "Variable",
                        "Tipo",
                        "Por qué se considera para forecast automotriz",
                        "Último valor real",
                        "Último valor proyectado",
                        "Variación %",
                        "Fuente sugerida",
                    ]
                    columnas_orden = [c for c in columnas_orden if c in df_display.columns]


                    st.dataframe(
                        df_display[columnas_orden],
                        use_container_width=True,
                        hide_index=True,
                    )

                    n_total = len(df_display)
                    n_dummy = df_display["Variable"].astype(str).str.upper().isin(DUMMY_VARS_LIST).sum()
                    n_cont = n_total - n_dummy

                    st.caption(
                        f"Esta proyección utiliza **{n_total} drivers**: "
                        f"{n_cont} cuantitativos (PIB, Riesgo País, tasas, etc.) y "
                        f"{n_dummy} dummies de eventos (Paros, IVA, COVID, elecciones, utilidades...)."
                    )

            st.markdown("---")

            # Métricas de modelos y modelo ganador
            col_head_m, col_pop_m = st.columns([0.7, 0.3])
            with col_head_m:
                st.markdown("### 🧠 Modelos evaluados y errores (Backtest)")
            with col_pop_m:
                with st.popover("ℹ️ Modelos y métricas", use_container_width=True):
                    st.markdown(
                        """
                        **Modelos incluidos (ejemplos):**
                        - **RandomForest:** bosque de árboles de decisión, captura relaciones no lineales y es robusto a outliers.
                        - **XGBoost:** algoritmo de *gradient boosting* muy potente para datos tabulares y relaciones complejas.
                        - **SARIMAX:** modelo clásico de series de tiempo con estacionalidad y variables explicativas (drivers).
                        - **Prophet:** modelo aditivo (tendencia + estacionalidad + festivos/eventos) orientado a series de negocio.

                        **Métricas (MAPE / MAE / RMSE):**
                        - **MAPE (%):** mide el error *relativo* promedio.  
                          Ej: un MAPE del 6% significa que, en promedio, el modelo se equivoca un 6% sobre las ventas reales.
                        - **MAE:** error absoluto promedio en **unidades**.  
                          Ej: MAE = 500 → en promedio el modelo se equivoca en 500 vehículos por mes.
                        - **RMSE:** error cuadrático medio. Penaliza más los errores grandes
                          y es útil para medir el riesgo de desviaciones fuertes.
                        """
                    )

            df_run_ft, df_metrics_ft = load_projection_metrics(engine, projection_name_ft)

            if df_metrics_ft.empty:
                st.warning(
                    "No se encontraron métricas de backtest para esta proyección en "
                    "'forecast_runs' / 'forecast_model_metrics'."
                )
            else:
                df_metrics_show = df_metrics_ft.copy()
                df_metrics_show["MAPE (%)"] = (df_metrics_show["mape"]).round(2)
                df_metrics_show["MAE"] = df_metrics_show["mae"].round(0).astype(int)
                df_metrics_show["RMSE"] = df_metrics_show["rmse"].round(0).astype(int)

                df_metrics_show = df_metrics_show[["model_name", "MAPE (%)", "MAE", "RMSE"]]

                st.dataframe(
                    df_metrics_show.rename(columns={"model_name": "Modelo"}).sort_values("MAPE (%)"),
                    use_container_width=True,
                    hide_index=True,
                )

                if not df_run_ft.empty:
                    row_run = df_run_ft.iloc[0]
                    best_model = row_run["best_model"]
                    best_mape = row_run["best_mape"]
                    best_mae = row_run["best_mae"]
                    best_rmse = row_run["best_rmse"]
                    train_start = pd.to_datetime(row_run["train_start"]).date()
                    train_end = pd.to_datetime(row_run["train_end"]).date()
                    test_start = pd.to_datetime(row_run["test_start"]).date()
                    test_end = pd.to_datetime(row_run["test_end"]).date()
                    horizon = int(row_run["horizon_months"])

                    st.markdown("### 🏆 Modelo Ganador y Justificación")

                    col_w1, col_w2 = st.columns([0.4, 0.6])

                    with col_w1:
                        st.metric("Modelo Ganador", best_model)
                        st.metric("MAPE Backtest", f"{best_mape:.2f} %")
                        st.metric("RMSE", f"{best_rmse:,.0f}")

                    with col_w2:
                        st.markdown(
                            f"""
                            **¿Por qué ganó `{best_model}`?**

                            - Se ejecutó un **backtest walk-forward** sobre el histórico de ventas del segmento **{segmento_base_ft}**.
                            - Período de entrenamiento: **{train_start} → {train_end}**  
                            - Período de prueba (validación): **{test_start} → {test_end}**  
                            - Horizonte de prueba por corrida: **{horizon} meses**.
                            - El modelo ganador obtuvo el **menor MAPE** (error porcentual medio absoluto) en las pruebas,
                              comparado contra los otros modelos (RandomForest, XGBoost, SARIMAX, Prophet).
                            - Además, el RMSE de `{best_model}` ({best_rmse:,.0f} unidades) indica que su error típico en unidades
                              es coherente con la escala del mercado.
                            """
                        )
                else:
                    st.info(
                        "Hay métricas por modelo en 'forecast_model_metrics', pero no un resumen en 'forecast_runs'. "
                        "Revisa si se guardó correctamente la corrida."
                    )

    # 2) Pronóstico Total (Real + TODOS los escenarios)
    st.subheader("Pronóstico de Ventas (Total Industria)")
    st.markdown(
        "Proyección mensual de ventas de la industria automotriz. "
        "Se muestra el histórico (real) y los tres escenarios de forecast (Normal, Optimista y Pesimista). "
        "Este gráfico **no** se ve afectado por el filtro de escenario."
    )

    df_real_chart = df_real_total.copy()
    df_real_chart['Escenario'] = 'Real'
    df_real_chart = df_real_chart.rename(columns={'Fecha': 'Fecha', 'Unidades': 'Ventas'})

    df_proy_chart = df_proy_total.copy().rename(columns={'Unidades': 'Ventas'})
    df_forecast_chart = pd.concat([df_real_chart, df_proy_chart], ignore_index=True)

    chart_forecast = (
        alt.Chart(df_forecast_chart)
        .mark_line(point=True)
        .encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Ventas:Q', title='Unidades'),
            color=alt.Color('Escenario:N', title='Escenario'),
            tooltip=['Fecha:T', 'Escenario:N', alt.Tooltip('Ventas:Q', format=',.0f')]
        )
        .interactive()
    )
    st.altair_chart(chart_forecast, use_container_width=True)

    # 2) Crecimiento Anual (según escenario seleccionado)
    st.subheader("Crecimiento Anual de la Industria")
    st.markdown(
        "Comparación de las ventas anuales reales vs. las ventas proyectadas bajo el "
        f"**escenario {escenario_overview}**. Para el año donde empieza el forecast se suma "
        "**real + proyección**; los años posteriores son solo proyección."
    )

    try:
        # --- 2.1 Real total ---
        df_real_anual = df_real_total.copy()
        df_real_anual['Fecha'] = pd.to_datetime(df_real_anual['Fecha'])
        df_real_anual['Year'] = df_real_anual['Fecha'].dt.year
        df_real_group = df_real_anual.groupby('Year')['Unidades'].sum().rename('Real')

        last_real_date = df_real_anual['Fecha'].max()

        # --- 2.2 Forecast solo del escenario seleccionado ---
        df_proj_sel = df_proy_total[df_proy_total['Escenario'] == escenario_overview].copy()
        df_proj_sel['Fecha'] = pd.to_datetime(df_proj_sel['Fecha'])

        if not df_proj_sel.empty:
            df_proj_future = df_proj_sel[df_proj_sel['Fecha'] > last_real_date].copy()
            if not df_proj_future.empty:
                df_proj_future['Year'] = df_proj_future['Fecha'].dt.year
                df_proj_group = df_proj_future.groupby('Year')['Unidades'].sum().rename('Proyectado')
            else:
                df_proj_group = pd.Series(dtype=float, name='Proyectado')
        else:
            df_proj_group = pd.Series(dtype=float, name='Proyectado')

        # --- 2.3 Unir real + proyección por año ---
        df_growth = pd.concat([df_real_group, df_proj_group], axis=1).fillna(0.0)
        df_growth.index.name = 'Año'
        df_growth = df_growth.reset_index()

        df_growth['Total'] = df_growth['Real'] + df_growth['Proyectado']

        def clasificar_tipo(row):
            y = row['Año']
            if row['Proyectado'] == 0 and row['Real'] > 0:
                return 'Solo real'
            if row['Real'] > 0 and row['Proyectado'] > 0:
                return 'Real + proyección'
            if row['Real'] == 0 and row['Proyectado'] > 0:
                return 'Solo proyección'
            return 'Sin datos'

        df_growth['Tipo'] = df_growth.apply(clasificar_tipo, axis=1)

        df_growth = df_growth.sort_values('Año').reset_index(drop=True)
        df_growth['Crecimiento %'] = 0.0
        for i in range(1, len(df_growth)):
            prev = df_growth.loc[i - 1, 'Total']
            curr = df_growth.loc[i, 'Total']
            if prev > 0:
                df_growth.loc[i, 'Crecimiento %'] = (curr / prev - 1) * 100

        df_growth_display = df_growth.copy()
        df_growth_display['Real'] = df_growth_display['Real'].round(0).astype(int)
        df_growth_display['Proyectado'] = df_growth_display['Proyectado'].round(0).astype(int)
        df_growth_display['Total'] = df_growth_display['Total'].round(0).astype(int)
        df_growth_display['Crecimiento %'] = df_growth_display['Crecimiento %'].map(lambda x: f"{x:.1f}%")

        st.dataframe(df_growth_display, use_container_width=True, hide_index=True)

        df_growth_chart = df_growth.copy()
        df_growth_chart['Label'] = df_growth_chart['Total'].map(lambda x: f"{x:,.0f}")

        chart_growth = (
            alt.Chart(df_growth_chart)
            .mark_bar()
            .encode(
                x=alt.X('Año:O', title='Año'),
                y=alt.Y('Total:Q', title='Ventas anuales (Total)'),
                color=alt.Color('Tipo:N', title='Tipo de año'),
                tooltip=[
                    alt.Tooltip('Año:O', title='Año'),
                    alt.Tooltip('Real:Q', format=',.0f', title='Real'),
                    alt.Tooltip('Proyectado:Q', format=',.0f', title='Proyectado'),
                    alt.Tooltip('Total:Q', format=',.0f', title='Total'),
                    alt.Tooltip('Crecimiento %:Q', format='.1f', title='Crecimiento %')
                ]
            )
        )

        text_growth = chart_growth.mark_text(
            dy=-5,
            color='black'
        ).encode(text='Label:N')

        st.altair_chart(chart_growth + text_growth, use_container_width=True)

    except Exception as e:
        st.error(f"Error en gráfico de Crecimiento Anual: {e}")
        st.exception(e)

    # 3) Análisis de Estacionalidad
    st.subheader("Análisis de Estacionalidad")
    st.markdown(
        "Gráfico de barras mensualizado por año. En el eje X se muestran los meses, "
        "en el eje Y las unidades vendidas y en la leyenda los **años**. "
        "Se toman los **últimos 3 años**, combinando datos reales y proyectados "
        f"del escenario **{escenario_overview}**. Sobre cada barra se muestra la "
        "variación porcentual vs. el mismo mes del año anterior."
    )

    # --- 3.1 Combinar real + proyectado (solo escenario seleccionado) ---
    df_real_est = df_real_total.copy()
    df_real_est['Fecha'] = pd.to_datetime(df_real_est['Fecha'])

    df_proy_est = df_proy_total[
        df_proy_total['Escenario'] == escenario_overview
    ].copy()
    df_proy_est['Fecha'] = pd.to_datetime(df_proy_est['Fecha'])

    last_real_date = df_real_est['Fecha'].max()

    df_hist = df_real_est.copy()
    df_fut = df_proy_est[df_proy_est['Fecha'] > last_real_date].copy()
    df_fut = df_fut.rename(columns={'Unidades': 'Unidades'})

    df_comb = pd.concat(
        [df_hist[['Fecha', 'Unidades']], df_fut[['Fecha', 'Unidades']]],
        ignore_index=True
    )

    df_comb['Year'] = df_comb['Fecha'].dt.year
    df_comb['MesNum'] = df_comb['Fecha'].dt.month
    df_comb['Mes'] = df_comb['MesNum'].map({
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    })

    df_est_all = (
        df_comb
        .groupby(['Mes', 'MesNum', 'Year'])['Unidades']
        .sum()
        .reset_index()
    )

    df_est_all = df_est_all.sort_values(['MesNum', 'Year']).reset_index(drop=True)
    df_est_all['VarPct'] = np.nan

    for mes in df_est_all['MesNum'].unique():
        sub_idx = df_est_all['MesNum'] == mes
        sub = df_est_all.loc[sub_idx].copy().sort_values('Year')
        prev_units = None
        for i, row in sub.iterrows():
            if prev_units is not None and prev_units > 0:
                df_est_all.loc[i, 'VarPct'] = (row['Unidades'] / prev_units - 1) * 100
            prev_units = row['Unidades']

    years_available = sorted(df_est_all['Year'].unique())
    if len(years_available) >= 3:
        last3_years = years_available[-3:]
    else:
        last3_years = years_available

    df_est_group = df_est_all[df_est_all['Year'].isin(last3_years)].copy()

    def format_varpct(x):
        if pd.isna(x):
            return ""
        return f"{x:.1f}%"

    df_est_group['Label'] = df_est_group['VarPct'].apply(format_varpct)

    mes_order = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # --- justo antes de base_est = alt.Chart(df_est_group) ---
    df_est_group["YearStr"] = df_est_group["Year"].astype(str)
    year_domain = [str(y) for y in sorted(df_est_group["Year"].unique())]

    color_scale_year = alt.Scale(
        domain=year_domain,
        scheme="tableau10"   # paleta categórica (colores bien distintos)
    )

    base_est = alt.Chart(df_est_group)

    chart_est = (
        base_est
        .mark_bar()
        .encode(
            x=alt.X('Mes:N', sort=mes_order, title='Mes'),
            xOffset=alt.XOffset('YearStr:N'),
            y=alt.Y('Unidades:Q', title='Unidades'),
            color=alt.Color('YearStr:N', title='Año', scale=color_scale_year, sort=year_domain),
            tooltip=[
                alt.Tooltip('YearStr:N', title='Año'),
                alt.Tooltip('Mes:N', title='Mes'),
                alt.Tooltip('Unidades:Q', format=',.0f', title='Unidades'),
                alt.Tooltip('VarPct:Q', format='.1f', title='Var. vs año anterior (%)')
            ]
        )
    )

    text_est = (
        base_est
        .mark_text(dy=-5, size=10)
        .encode(
            x=alt.X('Mes:N', sort=mes_order),
            xOffset=alt.XOffset('YearStr:N'),
            y=alt.Y('Unidades:Q'),
            text='Label:N'
        )
    )

    st.altair_chart(chart_est + text_est, use_container_width=True)

    # 4) Evolución de Share por Segmento
    st.subheader("Evolución de Share por Segmento")
    st.markdown(
        "Share mensual de cada segmento sobre el total industria. "
        "Las líneas continuas son históricas; las punteadas son proyecciones "
        "estimadas a partir de la tendencia y estacionalidad del share real "
        "usando el horizonte del forecast seleccionado."
    )

    try:
        df_share_real = (
            df_real_granular
            .groupby(['Fecha', 'segmento'])['Unidades']
            .sum()
            .reset_index()
        )
        df_share_real['Total_mes'] = df_share_real.groupby('Fecha')['Unidades'].transform('sum')
        df_share_real = df_share_real[df_share_real['Total_mes'] > 0]
        df_share_real['Share'] = df_share_real['Unidades'] / df_share_real['Total_mes']
        df_share_real['Tipo'] = 'Real'

        future_dates = list(df_total_forecast.index.unique())
        df_share_proy = pd.DataFrame()

        if future_dates:
            rows = []
            h = len(future_dates)
            eps = 1e-6

            for seg in df_share_real['segmento'].unique():
                df_seg = (
                    df_share_real[df_share_real['segmento'] == seg]
                    .sort_values('Fecha')
                    .dropna(subset=['Share'])
                    .copy()
                )
                if df_seg.empty:
                    continue

                df_seg['mes'] = df_seg['Fecha'].dt.month

                cutoff_season = df_seg['Fecha'].max() - pd.DateOffset(years=4)
                df_season = df_seg[df_seg['Fecha'] >= cutoff_season].copy()
                if df_season.empty:
                    df_season = df_seg.copy()

                mean_global = df_season['Share'].mean()
                if mean_global <= 0:
                    continue

                monthly_mean = df_season.groupby('mes')['Share'].mean()
                factores = (monthly_mean / mean_global).reindex(range(1, 13), fill_value=1.0)

                df_seg['Factor_mes'] = df_seg['mes'].map(factores).fillna(1.0)
                df_seg['Share_deseason'] = df_seg['Share'] / df_seg['Factor_mes']

                df_seg['t'] = np.arange(len(df_seg))
                y = df_seg['Share_deseason'].clip(eps, 1 - eps).values

                if len(df_seg) >= 6:
                    y_logit = np.log(y / (1 - y))
                    X = df_seg[['t']].values
                    model = LinearRegression()
                    model.fit(X, y_logit)

                    t_future = np.arange(len(df_seg), len(df_seg) + h).reshape(-1, 1)
                    y_logit_future = model.predict(t_future)
                    y_future_deseason = 1 / (1 + np.exp(-y_logit_future))
                else:
                    last_n = min(12, len(df_seg))
                    y_future_deseason = np.repeat(
                        df_seg.tail(last_n)['Share_deseason'].mean(),
                        h
                    )

                for fecha_f, y_des in zip(future_dates, y_future_deseason):
                    mes_f = pd.to_datetime(fecha_f).month
                    factor_mes_f = float(factores.get(mes_f, 1.0))
                    share_f = float(np.clip(y_des * factor_mes_f, eps, 1 - eps))

                    rows.append({
                        'Fecha': pd.to_datetime(fecha_f),
                        'segmento': seg,
                        'Share': share_f,
                        'Tipo': 'Proyectado'
                    })

            df_share_proy = pd.DataFrame(rows)
            if not df_share_proy.empty:
                df_share_proy['sum_mes'] = df_share_proy.groupby('Fecha')['Share'].transform('sum')
                df_share_proy['Share'] = np.where(
                    df_share_proy['sum_mes'] > 0,
                    df_share_proy['Share'] / df_share_proy['sum_mes'],
                    df_share_proy['Share']
                )

        df_share_all = pd.concat([df_share_real, df_share_proy], ignore_index=True)

        chart_share = (
            alt.Chart(df_share_all)
            .mark_line()
            .encode(
                x=alt.X('Fecha:T', title='Fecha'),
                y=alt.Y('Share:Q', title='Share segmento', axis=alt.Axis(format='%')),
                color=alt.Color('segmento:N', title='Segmento'),
                strokeDash=alt.StrokeDash('Tipo:N', title='Tipo'),
                tooltip=[
                    alt.Tooltip('Fecha:T', title='Fecha'),
                    alt.Tooltip('segmento:N', title='Segmento'),
                    alt.Tooltip('Tipo:N', title='Tipo'),
                    alt.Tooltip('Share:Q', title='Share', format='.1%')
                ]
            )
            .interactive()
        )
        st.altair_chart(chart_share, use_container_width=True)

        if 'df_share_proy' in locals() and not df_share_proy.empty:
            df_share_proy_seg = df_share_proy[['Fecha', 'segmento', 'Share']].copy()
        else:
            df_share_proy_seg = pd.DataFrame()

    except Exception as e:
        st.error("Error al generar la evolución de share por segmento.")
        st.exception(e)

    # 5) Unidades por segmento (Real + Proyección del escenario seleccionado)
    st.subheader("Unidades por Segmento (Real + Proyección)")
    st.markdown(
        "Distribución de ventas por segmento a lo largo del tiempo. "
        "Incluye histórico real y forecast del escenario seleccionado: "
        f"**{escenario_overview}**."
    )

    df_seg_real = (
        df_real_granular
        .groupby(['Fecha', 'segmento'])['Unidades']
        .sum()
        .reset_index()
    )
    df_seg_real['Tipo'] = 'Real'
    df_seg_real['Fecha'] = pd.to_datetime(df_seg_real['Fecha'])

    df_seg_all = df_seg_real.copy()

    if 'df_share_proy_seg' not in locals() or df_share_proy_seg.empty:
        st.warning("No hay shares proyectados por segmento; se muestran solo datos reales.")
    else:
        df_total_esc = df_proy_total[df_proy_total['Escenario'] == escenario_overview].copy()
        df_total_esc = df_total_esc.rename(columns={'Unidades': 'Unidades_total'})
        df_total_esc['Fecha'] = pd.to_datetime(df_total_esc['Fecha'])

        df_seg_proj = df_share_proy_seg.merge(
            df_total_esc,
            on='Fecha',
            how='inner'
        )
        df_seg_proj['Unidades'] = df_seg_proj['Share'] * df_seg_proj['Unidades_total']
        df_seg_proj['Tipo'] = f'Proyectado_{escenario_overview}'

        df_seg_all = pd.concat(
            [df_seg_real, df_seg_proj[['Fecha', 'segmento', 'Unidades', 'Tipo']]],
            ignore_index=True
        )

    chart_seg = (
        alt.Chart(df_seg_all)
        .mark_bar()
        .encode(
            x=alt.X('Fecha:T', title='Fecha'),
            y=alt.Y('Unidades:Q', title='Unidades'),
            color=alt.Color('segmento:N', title='Segmento'),
            tooltip=[
                alt.Tooltip('Fecha:T', title='Fecha'),
                alt.Tooltip('segmento:N', title='Segmento'),
                alt.Tooltip('Tipo:N', title='Tipo'),
                alt.Tooltip('Unidades:Q', format=',.0f', title='Unidades')
            ]
        )
    )

    st.altair_chart(chart_seg.interactive(), use_container_width=True)


# -------------------------------------------------
# TAB 2: DETALLE POR MODELO / VERSIÓN
# -------------------------------------------------
with tab_modelos:
    st.header("Detalle por modelo / versión")

    st.markdown(
        "En esta pestaña se calcula la **desagregación a nivel modelo/versión**.\n"
        "Las ventas por modelo se estiman a partir del forecast total de la industria, "
        "usando un share de modelo dentro de cada segmento calculado con los últimos "
        "**N meses** de ventas reales."
    )

    # =============================
    # CONFIGURACIÓN + FILTROS (EXPANDER)
    # =============================
    with st.expander("Configuración de Desagregación y Filtros de Detalle", expanded=True):

        # --- Configuración de desagregación ---
        st.subheader("Configuración de Desagregación")
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            meses_share_modelo = st.slider(
                "Meses recientes para calcular el share por modelo:",
                min_value=1,
                max_value=12,
                value=st.session_state['meses_share_modelo'],
                help="Número de meses recientes para distribuir el forecast a modelos."
            )
        with col_d2:
            st.write("")
            st.write("")
            calc_button = st.button("Calcular Mercado Detallado", type="primary")

        st.markdown("---")

        # --- Filtros de detalle por modelo / versión ---
        st.subheader("Filtros de Detalle por modelo / versión")

        segmentos_disp = sorted(df_real_granular['segmento'].unique())
        marcas_disp = sorted(df_real_granular['marca'].unique())
        modelos_disp = sorted(df_real_granular['modelo'].unique())
        provincias_disp = sorted(df_real_granular['provincia'].unique())

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filt_segmento = st.multiselect("Segmento", ["Todas"] + segmentos_disp, default=["Todas"])
            filt_marca = st.multiselect("Marca", ["Todas"] + marcas_disp, default=["Todas"])
        with col_f2:
            filt_modelo = st.multiselect("Modelo", ["Todas"] + modelos_disp, default=["Todas"])
            filt_provincia = st.multiselect("Provincia", ["Todas"] + provincias_disp, default=["Todas"])

        # Rango de fechas para detalle
        min_date_hist = df_real_granular['Fecha'].min().date()
        max_date_hist = df_real_granular['Fecha'].max().date()
        date_range_det = st.date_input(
            "Rango de fechas (detalle):",
            value=(min_date_hist, max_date_hist),
            min_value=min_date_hist,
            max_value=max_date_hist
        )

        try:
            det_start_date, det_end_date = date_range_det
        except ValueError:
            det_start_date, det_end_date = (min_date_hist, max_date_hist)

        escenarios_detalle = ['Normal', 'Optimista', 'Pesimista']
        escenario_detalle = st.radio(
            "Escenario (detalle):",
            escenarios_detalle,
            index=0,
            horizontal=True
        )

    if calc_button:
        with st.spinner("Calculando desagregación top-down por modelo..."):
            df_desagregado = calculate_desagregado(
                st.session_state['df_total_forecast'],
                df_real_granular,
                meses_share=meses_share_modelo
            )
            if df_desagregado.empty:
                st.error("No se pudo generar la desagregación.")
            else:
                st.session_state['df_desagregado'] = df_desagregado
                st.session_state['meses_share_modelo'] = meses_share_modelo
                st.success("Desagregación generada correctamente.")

    df_desag = st.session_state['df_desagregado']

    if df_desag is None or df_desag.empty:
        st.info("Aún no se ha calculado la desagregación. Configura y pulsa **Calcular Mercado Detallado**.")
    else:
        st.markdown(
            f"Desagregación calculada usando los últimos **{st.session_state['meses_share_modelo']}** meses "
            "para el share por modelo."
        )

        df_real_hist = df_real_granular.copy()
        df_real_hist['Tipo'] = 'Real'
        df_real_hist['Escenario'] = 'Real'

        df_union = pd.concat([df_real_hist, df_desag], ignore_index=True)

        seg_filtrar = segmentos_disp if "Todas" in filt_segmento else filt_segmento
        marca_filtrar = marcas_disp if "Todas" in filt_marca else filt_marca
        modelo_filtrar = modelos_disp if "Todas" in filt_modelo else filt_modelo
        prov_filtrar = provincias_disp if "Todas" in filt_provincia else filt_provincia

        df_union_f = df_union[
            (df_union['segmento'].isin(seg_filtrar)) &
            (df_union['marca'].isin(marca_filtrar)) &
            (df_union['modelo'].isin(modelo_filtrar)) &
            (df_union['provincia'].isin(prov_filtrar))
        ].copy()

        df_union_f['Fecha'] = pd.to_datetime(df_union_f['Fecha'])

        df_union_f = df_union_f[
            (df_union_f['Fecha'] >= pd.to_datetime(det_start_date)) &
            (df_union_f['Fecha'] <= pd.to_datetime(det_end_date))
        ]

        df_det_real = df_union_f[df_union_f['Tipo'] == 'Real'].copy()
        df_det_proy = df_union_f[
            (df_union_f['Tipo'] == 'Proyectado') &
            (df_union_f['Escenario'] == escenario_detalle)
        ].copy()

        st.subheader(f"KPIs del Forecast desagregado (Escenario: {escenario_detalle})")
        k1, k2 = st.columns(2)

        if not df_det_proy.empty:
            df_det_proy['Year'] = df_det_proy['Fecha'].dt.year
            proy_por_ano = df_det_proy.groupby('Year')['Unidades'].sum()
            with k1:
                st.markdown("**Proyección por Año (detalle filtrado)**")
                cols_year = st.columns(len(proy_por_ano))
                for i, (yy, total) in enumerate(proy_por_ano.items()):
                    cols_year[i].metric(label=f"{yy}", value=f"{total:,.0f} un.")
        else:
            with k1:
                st.info("Sin proyección para los filtros seleccionados.")

        with k2:
            if not df_det_proy.empty:
                total_proy = df_det_proy.groupby('Fecha')['Unidades'].sum().sum()
                n_meses_proy = df_det_proy['Fecha'].dt.to_period('M').nunique()
                prom_proy = total_proy / n_meses_proy if n_meses_proy > 0 else 0

                if not df_det_real.empty:
                    last_real = df_det_real['Fecha'].max()
                    l12m_ini = last_real - pd.DateOffset(months=11)
                    df_l12m = df_det_real[df_det_real['Fecha'] >= l12m_ini]
                    total_real_l12m = df_l12m.groupby('Fecha')['Unidades'].sum().sum()
                    n_meses_real = df_l12m['Fecha'].dt.to_period('M').nunique()
                    prom_real = total_real_l12m / n_meses_real if n_meses_real > 0 else 0
                    delta = (prom_proy / prom_real - 1) if prom_real > 0 else None
                    st.metric(
                        "Promedio mensual proyectado",
                        f"{prom_proy:,.0f} un.",
                        f"{delta:.1%}" if delta is not None else None,
                        help=f"Promedio real L12M: {prom_real:,.0f} un."
                    )
                else:
                    st.metric("Promedio mensual proyectado", f"{prom_proy:,.0f} un.")
            else:
                st.info("Sin proyección para calcular promedios.")

        st.markdown("---")

        st.subheader("Serie de tiempo (Real vs Proyectado – filtros aplicados)")

        df_ts_real = df_det_real.groupby('Fecha')['Unidades'].sum().reset_index()
        df_ts_real['Tipo'] = 'Real'

        df_ts_proy = df_det_proy.groupby('Fecha')['Unidades'].sum().reset_index()
        df_ts_proy['Tipo'] = f'Proyectado_{escenario_detalle}'

        df_ts = pd.concat([df_ts_real, df_ts_proy], ignore_index=True)
        df_ts = df_ts.rename(columns={'Unidades': 'Ventas'})

        chart_ts = (
            alt.Chart(df_ts)
            .mark_line(point=True)
            .encode(
                x=alt.X('Fecha:T', title='Fecha'),
                y=alt.Y('Ventas:Q', title='Unidades'),
                color=alt.Color('Tipo:N', title='Tipo'),
                tooltip=['Fecha:T', 'Tipo:N', alt.Tooltip('Ventas:Q', format=',.0f')]
            )
            .interactive()
        )
        st.altair_chart(chart_ts, use_container_width=True)

        st.subheader("Tabla detallada por modelo / versión")
        st.dataframe(
            df_det_proy.sort_values(['Fecha', 'marca', 'modelo']),
            use_container_width=True,
            hide_index=True
        )

        if not df_det_proy.empty:
            st.subheader("Top 15 modelos por año (Escenario proyectado)")

            years_disp = sorted(df_det_proy['Fecha'].dt.year.unique())
            anio_sel = st.selectbox("Año a analizar:", options=years_disp, index=0)

            df_anio = df_det_proy[df_det_proy['Fecha'].dt.year == anio_sel].copy()
            if df_anio.empty:
                st.info("No hay datos para ese año con los filtros actuales.")
            else:
                df_top_modelos = (
                    df_anio
                    .groupby(['marca', 'modelo'])['Unidades']
                    .sum()
                    .reset_index()
                    .sort_values('Unidades', ascending=False)
                    .head(15)
                )

                chart_top = (
                    alt.Chart(df_top_modelos)
                    .mark_bar()
                    .encode(
                        x=alt.X('Unidades:Q', title='Unidades proyectadas'),
                        y=alt.Y('modelo:N', sort='-x', title='Modelo'),
                        color='marca:N',
                        tooltip=['marca:N', 'modelo:N', alt.Tooltip('Unidades:Q', format=',.0f')]
                    )
                )
                st.altair_chart(chart_top, use_container_width=True)