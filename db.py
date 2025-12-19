# db.py
import os
import streamlit as st
from sqlalchemy import create_engine

def get_engine():
    db_url = None

    # 1) Streamlit Cloud / secrets.toml
    if "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]

    # 2) Local env var
    if not db_url:
        db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

    if not db_url:
        raise RuntimeError("No se encontró DATABASE_URL en st.secrets ni en variables de entorno.")

    # SQLAlchemy necesita el driver
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    # Recomendación: evitar channel_binding si existiera
    db_url = db_url.replace("&channel_binding=require", "")

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    return engine
