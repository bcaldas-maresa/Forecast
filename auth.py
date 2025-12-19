import streamlit as st
from sqlalchemy import create_engine, text
import bcrypt
from db import get_engine

engine = get_engine()

@st.cache_resource
def get_db_engine():
    """Crea y cachea la conexión a la BD."""
    try:
        engine = get_engine()
        return engine
    except Exception as e:
        st.error(f"Error al crear el 'engine' de SQLAlchemy: {e}")
        st.stop()

def hash_password(password):
    """Hashea una contraseña."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    """Verifica una contraseña contra su hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(email, password):
    """
    Registra un nuevo usuario.
    Verifica el dominio y lo guarda como NO APROBADO.
    """
    # 1. Validar dominio
    if not email.endswith('@corpmaresa.com.ec'):
        return False, "Error: Solo se permiten correos con dominio @corpmaresa.com.ec"
    
    # 2. Hashear contraseña
    password_hash = hash_password(password).decode('utf-8')
    
    # 3. Insertar en la BD
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            sql = text("""
                INSERT INTO users (email, password_hash, is_approved, is_admin)
                VALUES (:email, :pass_hash, :approved, :admin)
            """)
            conn.execute(sql, {
                "email": email,
                "pass_hash": password_hash,
                "approved": False,
                "admin": False
            })
            conn.commit()
        return True, "¡Registro exitoso! Su cuenta está pendiente de aprobación por un administrador."
    except Exception as e:
        if 'Duplicate entry' in str(e):
            return False, "Error: El correo electrónico ya está registrado."
        else:
            return False, f"Error de base de datos: {e}"

def login_user(email, password):
    """
    Verifica las credenciales del usuario Y si está aprobado.
    Devuelve (True/False, Mensaje, is_admin)
    """
    engine = get_db_engine()
    try:
        with engine.connect() as conn:
            sql = text("SELECT password_hash, is_approved, is_admin FROM users WHERE email = :email")
            result = conn.execute(sql, {"email": email}).fetchone()
            
            if not result:
                return False, "Error: Correo o contraseña incorrectos.", False
            
            stored_hash, is_approved, is_admin = result
            
            # 1. Verificar contraseña
            if not check_password(password, stored_hash):
                return False, "Error: Correo o contraseña incorrectos.", False
            
            # 2. Verificar aprobación
            if not is_approved:
                return False, "Error: Su cuenta aún no ha sido aprobada por un administrador.", False
            
            # ¡Éxito!
            st.session_state['logged_in'] = True
            st.session_state['email'] = email
            st.session_state['is_admin'] = is_admin
            return True, "Login exitoso", is_admin
            
    except Exception as e:
        return False, f"Error de base de datos: {e}", False