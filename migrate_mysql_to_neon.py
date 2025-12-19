import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import text, inspect

# ====== CONFIG ======
# 1) MySQL (origen)
MYSQL_URL = "mysql+pymysql://root:Bcaldas123@localhost:3306/forecast_db"

# 2) Postgres Neon (destino) - usa tu mismo secrets o env var
NEON_URL = "postgresql://neondb_owner:npg_dKy2J5HTqSfb@ep-wandering-salad-a4ha6mmh-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
  # Ej: postgresql+psycopg2://...

TABLES = [
    # Tablas de forecasting (las que te faltan en Neon)
    "forecast_runs",
    "forecast_model_metrics",
    "forecast_driver_scenarios",
    "forecast_final_projections",
    # Opcional pero recomendado para tu ficha técnica
    "historical_drivers",
]

CHUNKSIZE = 5000  # puedes subirlo a 20000 si va rápido

def table_exists(pg_engine, table_name: str) -> bool:
    q = text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema='public' AND table_name=:t
        )
    """)
    with pg_engine.connect() as c:
        return bool(c.execute(q, {"t": table_name}).scalar())

def truncate_table(pg_engine, table_name: str):
    with pg_engine.begin() as c:
        c.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;'))

def reset_sequence(engine, table_name, pk_col="id", schema="public"):
    insp = inspect(engine)
    cols = [c["name"] for c in insp.get_columns(table_name, schema=schema)]

    # Si la tabla no tiene la columna, no hay nada que resetear
    if pk_col not in cols:
        print(f"  -> skip reset_sequence({table_name}): no column '{pk_col}'")
        return

    sql = text(f"""
    DO $$
    DECLARE
        seq_name text;
    BEGIN
        -- Obtiene la secuencia asociada a una columna serial/identity
        SELECT pg_get_serial_sequence('{schema}.' || quote_ident(:tbl), :col)
          INTO seq_name;

        IF seq_name IS NOT NULL THEN
            EXECUTE format(
              'SELECT setval(%L, COALESCE((SELECT MAX(%I) FROM {schema}.%I), 1), true)',
              seq_name, :col, :tbl
            );
        END IF;
    END $$;
    """)

    with engine.begin() as c:
        c.execute(sql, {"tbl": table_name, "col": pk_col})

def main():
    if not MYSQL_URL:
        raise RuntimeError("Falta MYSQL_URL en variables de entorno.")
    if not NEON_URL:
        raise RuntimeError("Falta DATABASE_URL en variables de entorno (la de Neon).")

    mysql_engine = create_engine(MYSQL_URL)
    neon_engine = create_engine(NEON_URL, pool_pre_ping=True)

    print("Conectando a MySQL y Neon... OK")

    for t in TABLES:
        print(f"\n=== Migrando tabla: {t} ===")

        if not table_exists(neon_engine, t):
            print(f"ERROR: En Neon no existe la tabla '{t}'. Primero créala con database_setup.py.")
            continue

        # 1) Limpiar destino (para evitar duplicados)
        print("Truncando en Neon...")
        truncate_table(neon_engine, t)

        # 2) Copiar data en chunks
        total = 0
        for chunk in pd.read_sql(f"SELECT * FROM {t}", mysql_engine, chunksize=CHUNKSIZE):
            # Ajustes típicos si alguna fecha viene como string
            # (si no aplica, no pasa nada)
            for col in chunk.columns:
                if "fecha" in col.lower() or "created" in col.lower() or "updated" in col.lower():
                    try:
                        chunk[col] = pd.to_datetime(chunk[col], errors="ignore")
                    except Exception:
                        pass

            chunk.to_sql(t, neon_engine, if_exists="append", index=False, method="multi")
            total += len(chunk)
            print(f"  -> {total} filas...")

        # 3) Reset seq si aplica
        reset_sequence(neon_engine, t)
        print(f"OK: {t} migrada. Total filas: {total}")

    # Validación rápida
    print("\n=== Validación rápida ===")
    with neon_engine.connect() as c:
        for t in TABLES:
            if table_exists(neon_engine, t):
                n = c.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
                print(f"{t}: {n}")

    print("\nLISTO. Ya puedes abrir la app y ver proyecciones en Neon.")

if __name__ == "__main__":
    main()