import streamlit as st
from auth import register_user


# --- CSS para Ocultar el Menú ---
# Un usuario en la página de registro NUNCA está logueado,
# así que podemos ocultar el menú de forma segura.
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
# --- Fin del CSS ---

st.title("Registro de Nuevo Usuario")
st.warning("Solo se permiten correos con dominio **@corpmaresa.com.ec**")

with st.form("register_form"):
    email = st.text_input("Correo Electrónico Corporativo")
    password = st.text_input("Contraseña", type="password")
    confirm_password = st.text_input("Confirmar Contraseña", type="password")
    
    submitted = st.form_submit_button("Registrarse")
    
    if submitted:
        if not email or not password or not confirm_password:
            st.error("Por favor, llene todos los campos.")
        elif password != confirm_password:
            st.error("Las contraseñas no coinciden.")
        else:
            success, message = register_user(email, password)
            if success:
                st.success(message)
            else:
                st.error(message)

st.page_link("app.py", label="Volver a Login", icon="🏠")

# --- Logo inferior y copyright (IAN) ---
st.sidebar.markdown("---")
st.sidebar.image("img/Logo_IAN.png", use_container_width=True)
st.sidebar.markdown(
    "<small>© Maresa. Todos los derechos reservados.</small>",
    unsafe_allow_html=True,
)