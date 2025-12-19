from sqlalchemy import (
    Column, Integer, String, Float, Date, MetaData, Table, Text, Boolean, 
    Index, TIMESTAMP, func, UniqueConstraint
)
from db import get_engine

# --- TUS DATOS DE CONEXIÓN ---

def setup_database():
    """
    v6.1: 
    - Añade la tabla 'forecast_final_projections'.
    - Mantiene las tablas existentes.
    """
    try:
        engine = get_engine()
        meta = MetaData()

        # --- Cargar tablas existentes ---
        # (Esto permite que meta.create_all() no falle en tablas ya creadas)
        try:
            users = Table('users', meta, extend_existing=True, autoload_with=engine)
            historical_drivers = Table('historical_drivers', meta, extend_existing=True, autoload_with=engine)
            forecast_driver_scenarios = Table('forecast_driver_scenarios', meta, extend_existing=True, autoload_with=engine)
            sales_granular = Table('sales_granular', meta, extend_existing=True, autoload_with=engine)
        except Exception as e:
            print(f"Nota: No se pudieron cargar todas las tablas existentes (normal en la primera ejecución): {e}")

        # --- ***NUEVA TABLA: forecast_final_projections*** ---
        # Esta tabla guarda el pronóstico de VENTAS final (ya ajustado)
        forecast_final_projections = Table(
            'forecast_final_projections', meta,
            Column('id', Integer, primary_key=True, autoincrement=True),
            
            Column('projection_name', String(255), nullable=False, 
                   comment="Ej: Forecast Q4 2025 - XGBoost"),
            Column('segmento_base', String(100), nullable=False, 
                   comment="El segmento pronosticado (ej. Total_Filtrado)"),
            Column('modelo_usado', String(100), nullable=False,
                   comment="El modelo ganador (ej. XGBoost)"),
            
            Column('fecha', Date, nullable=False),
            Column('tipo_escenario', String(50), nullable=False, 
                   comment="Normal, Optimista, Pesimista"),
            Column('valor_proyectado', Float),
            
            Column('created_at', TIMESTAMP, server_default=func.now()),
            
            # --- Índices y Restricciones ---
            UniqueConstraint('projection_name', 'fecha', 'tipo_escenario', 
                             name='uq_projection_data'),
            Index('idx_projection_name', 'projection_name')
        )
        # --- NUEVA TABLA: dim_vehicle_versions (catálogo) ---
        dim_vehicle_versions = Table(
            'dim_vehicle_versions', meta,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('marca', String(100), nullable=False),
            Column('familia', String(150), nullable=False),
            Column('modelo', String(150), nullable=False),
            Column('segmento', String(100)),
            Column('tipo_combustible', String(100)),
            Column('origen', String(100)),
            Column('tipo_hibridacion', String(100)),
            Column('last_seen_at', TIMESTAMP, server_default=func.now(), onupdate=func.now()),
            UniqueConstraint('marca', 'familia', 'modelo', name='uq_dim_vehicle_versions'),
            Index('idx_dim_vehicle_versions_marca', 'marca'),
            Index('idx_dim_vehicle_versions_familia', 'familia'),
        )

        # --- NUEVA TABLA: vehicle_main_competitor (estado actual) ---
        vehicle_main_competitor = Table(
            'vehicle_main_competitor', meta,
            Column('id', Integer, primary_key=True, autoincrement=True),

            Column('own_brand', String(100), nullable=False),
            Column('own_familia', String(150), nullable=False),
            Column('own_modelo', String(150), nullable=False),

            Column('competitor_brand', String(100), nullable=False),
            Column('competitor_familia', String(150)),
            Column('competitor_modelo', String(150), nullable=False),

            Column('competition_type', String(20), nullable=False),  # Directa / Indirecta
            Column('notes', Text),

            Column('created_at', TIMESTAMP, server_default=func.now()),
            Column('created_by', String(255)),
            Column('updated_at', TIMESTAMP, server_default=func.now(), onupdate=func.now()),
            Column('updated_by', String(255)),

            UniqueConstraint('own_brand', 'own_familia', 'own_modelo', name='uq_main_competitor_own'),
            Index('idx_main_competitor_own', 'own_brand', 'own_familia', 'own_modelo'),
        )

        # --- NUEVA TABLA: vehicle_main_competitor_history (auditoría completa) ---
        vehicle_main_competitor_history = Table(
            'vehicle_main_competitor_history', meta,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('main_id', Integer),
            Column('action', String(10), nullable=False),  # INSERT / UPDATE / DELETE
            Column('changed_at', TIMESTAMP, server_default=func.now()),
            Column('changed_by', String(255)),
            Column('old_row_json', Text),
            Column('new_row_json', Text),
            Index('idx_main_comp_hist_main_id', 'main_id'),
        )


        meta.create_all(engine)
        print("¡Base de datos y tablas (v6.1) configuradas exitosamente!")
        print("Se ha CREADO/VERIFICADO la tabla 'forecast_final_projections'.")

    except Exception as e:
        print(f"--- ERROR Conectando o Creando la Base de Datos ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    setup_database()