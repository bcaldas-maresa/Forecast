import pandas as pd
from sqlalchemy import Date
import os
from sqlalchemy import text
from sqlalchemy.types import Date
from db import get_engine


# --- RUTAS DE ARCHIVOS ---
FILE_INDUSTRIA = os.path.join('data', 'industria.xlsx')
FILE_VARIABLES = os.path.join('data', 'variables.xlsx')

def ensure_dim_vehicle_versions_table(engine):
    """Crea (si no existe) la tabla catálogo de modelos/versiones para selección en la UI."""
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_vehicle_versions (
        id SERIAL PRIMARY KEY,
        marca VARCHAR(100) NOT NULL,
        familia VARCHAR(150) NOT NULL,
        modelo VARCHAR(150) NOT NULL,
        segmento VARCHAR(100),
        tipo_combustible VARCHAR(100),
        origen VARCHAR(100),
        tipo_hibridacion VARCHAR(100),
        last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_dim_vehicle_versions UNIQUE (marca, familia, modelo)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def upsert_dim_vehicle_versions(engine, df_sales_granular):
    """Inserta/actualiza el catálogo (marca, familia, modelo) a partir de sales_granular."""
    if df_sales_granular is None or df_sales_granular.empty:
        print("ADVERTENCIA: df_sales_granular vacío; no se actualiza dim_vehicle_versions.")
        return

    ensure_dim_vehicle_versions_table(engine)

    cols = ['marca', 'familia', 'modelo', 'segmento', 'tipo_combustible', 'origen', 'tipo_hibridacion']
    cols_present = [c for c in cols if c in df_sales_granular.columns]
    df_dim = df_sales_granular[cols_present].drop_duplicates().copy()

    for c in ['marca', 'familia', 'modelo']:
        if c in df_dim.columns:
            df_dim[c] = df_dim[c].fillna('No Especificado').astype(str).str.strip()

    for c in ['segmento', 'tipo_combustible', 'origen', 'tipo_hibridacion']:
        if c not in df_dim.columns:
            df_dim[c] = None

    upsert_sql = text("""
    INSERT INTO dim_vehicle_versions
        (marca, familia, modelo, segmento, tipo_combustible, origen, tipo_hibridacion, last_seen_at)
    VALUES
        (:marca, :familia, :modelo, :segmento, :tipo_combustible, :origen, :tipo_hibridacion, NOW())
    ON CONFLICT (marca, familia, modelo)
    DO UPDATE SET
        segmento = EXCLUDED.segmento,
        tipo_combustible = EXCLUDED.tipo_combustible,
        origen = EXCLUDED.origen,
        tipo_hibridacion = EXCLUDED.tipo_hibridacion,
        last_seen_at = NOW();
    """)

    rows = df_dim.to_dict(orient="records")  # o el df que uses para dim_vehicle_versions

    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    print(f"Catálogo dim_vehicle_versions actualizado: {len(df_dim)} combinaciones únicas.")


def clean_col_names(df):
    """Limpia nombres de columnas para ser compatibles con la BD."""
    df.columns = [
        col.replace(' ', '_').replace('/', '_').replace('.', '_')
        for col in df.columns
    ]
    df = df.rename(columns={
        'Riesgo_Pais': 'Riesgo_Pais',
        'Tasa_de_Interes_Activa': 'Tasa_de_Interes_Activa'
    })
    return df

def load_drivers(engine):
    """Carga los drivers (variables.xlsx) - Sin cambios"""
    try:
        print(f"Cargando {FILE_VARIABLES}...")
        df_vars = pd.read_excel(FILE_VARIABLES)
        df_vars = clean_col_names(df_vars)
        df_vars['Fecha'] = pd.to_datetime(df_vars['Fecha'])
        
        column_order = [
            'Fecha', 'IVA', 'WTI', 'euro_usd', 'PVP_ORO_1_OZ', 'PIB_DEM_CAD_BRU',
            'DEMINT_CORR_AJUS', 'PIB_CAD_AJUS', 'ING_PETROLEO', 'PETRO_PROD',
            'IPC', 'InflacionMensual', 'CreditoSectorPrivado', 'DepositosAlaVista',
            'ICC', 'EmpleoAdecuadoPleno', 'IMP_CBU', 'Riesgo_Pais',
            'Tasa_de_Interes_Activa', 'Cartera_Creditos_de_Consumo',
            'Utilidades', 'Paro', 'CUP_IMPORT', 'Elec_Presidenciales', 'Cambio_IVA'
        ]
        cols_to_load = [col for col in column_order if col in df_vars.columns]
        df_vars_final = df_vars[cols_to_load]

        df_vars_final.to_sql(
            'historical_drivers', 
            engine, 
            if_exists='replace', 
            index=False,
            dtype={'Fecha': Date}
        )
        print("Datos de 'historical_drivers' actualizados.")
    except Exception as e:
        print(f"--- ERROR Cargando Drivers ---")
        print(f"Error: {e}")

def load_granular_sales(engine):
    """
    v4.9: Carga granular leyendo el archivo COMPLETO primero,
    luego seleccionando y renombrando. Esto es más robusto.
    """
    try:
        print(f"Cargando archivo granular {FILE_INDUSTRIA}...")
        print("Esto puede tardar varios minutos...")
        
        # --- INICIO DE LA CORRECCIÓN ---
        
        # 1. Definir el mapeo de las columnas que queremos.
        # (Nombre en Excel - 100% igual a tu cabecera) : (Nombre en Base de Datos)
        column_mapping = {
            'Fecha proceso': 'fecha_proceso', 
            'Segmento': 'segmento', 
            'Marca': 'marca', 
            'Modelo': 'modelo',
            'Familia': 'familia',
            'Provincia': 'provincia',
            'Tipo de combustible': 'tipo_combustible', 
            'País': 'origen', # Renombramos 'País' a 'origen'
            'Tipo de hibridación': 'tipo_hibridacion', 
            'Unidades': 'unidades'
        }
        
        # 2. Leer el archivo Excel COMPLETO
        df_full = pd.read_excel(FILE_INDUSTRIA)
        print(f"Lectura de Excel completada. {len(df_full)} filas leídas.")
        print(f"Columnas encontradas en el Excel: {df_full.columns.tolist()}")

        # 3. Validar que todas las columnas que queremos existan
        required_excel_cols = list(column_mapping.keys())
        missing_cols = [col for col in required_excel_cols if col not in df_full.columns]
        
        if missing_cols:
            raise ValueError(f"¡Columnas Faltantes en 'industria.xlsx'! No se puede continuar. Columnas no encontradas: {missing_cols}")

        # 4. Seleccionar solo las columnas que nos importan
        df_sales_granular = df_full[required_excel_cols].copy()
        
        # 5. Renombrar las columnas a los nombres de la BD
        df_sales_granular = df_sales_granular.rename(columns=column_mapping)
        
        print(f"Columnas seleccionadas y renombradas a: {df_sales_granular.columns.tolist()}")
        # --- FIN DE LA CORRECCIÓN ---
        
        # 6. Limpieza de datos (igual que antes)
        print("Iniciando limpieza de datos...")
        # Convertir a fecha, manejando errores (aunque no deberían ocurrir ahora)
        df_sales_granular['fecha_proceso'] = pd.to_datetime(df_sales_granular['fecha_proceso'], errors='coerce')
        df_sales_granular = df_sales_granular.dropna(subset=['fecha_proceso', 'unidades'])
        
        # Rellenar NaNs en texto
        text_cols = ['segmento', 'marca', 'familia', 'modelo', 'provincia', 'tipo_combustible', 'origen', 'tipo_hibridacion']
        for col in text_cols:
            if col in df_sales_granular.columns:
                df_sales_granular[col] = df_sales_granular[col].fillna('No Especificado')

        print("Limpieza de datos completada.")
        print(f"Cargando {len(df_sales_granular)} filas válidas en la tabla 'sales_granular'...")
        
        if df_sales_granular.empty:
            print("ADVERTENCIA: El DataFrame está vacío. ¿El archivo 'industria.xlsx' tiene datos válidos?")
            return

        # 7. Cargar a la BD (igual que antes)
        df_sales_granular.to_sql(
            'sales_granular',
            engine,
            if_exists='replace',
            index=False,
            dtype={'fecha_proceso': Date},
            chunksize=10000
        )

        # Actualizar catálogo de modelos/versiones para UI (no afecta sales_granular)
        try:
            upsert_dim_vehicle_versions(engine, df_sales_granular)
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo actualizar dim_vehicle_versions: {e}")

        print("¡Datos de 'sales_granular' cargados exitosamente!")
        
    except ValueError as e:
        # Este 'except' ahora capturará nuestro error de 'Columnas Faltantes'
        print(f"--- ERROR Cargando Ventas Granulares (ValueError) ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"--- ERROR Cargando Ventas Granulares (Otro Error) ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    engine = get_engine()
    load_drivers(engine)
    load_granular_sales(engine)
    print("\n--- PROCESO DE CARGA DE DATOS COMPLETADO ---")