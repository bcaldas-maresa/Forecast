from sqlalchemy import text
from db import get_engine

engine = get_engine()

with engine.connect() as conn:
    val = conn.execute(text("SELECT 1")).scalar()
    print("OK DB:", val)
