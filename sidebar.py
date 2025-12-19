# sidebar.py
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "img"

MARESA_LOGO = IMG_DIR / "Maresa_Logo.png"
IAN_LOGO = IMG_DIR / "Logo_IAN.png"

def render_sidebar():
    with st.sidebar:
        # --- Logo superior (Maresa) ---
        if MARESA_LOGO.exists():
            st.image(str(MARESA_LOGO), use_container_width=True)
        else:
            st.warning("No se encontró img/Maresa_Logo.png")

        # --- Menú de navegación (igual en todas las páginas) ---
        st.page_link("app.py", label="app")
        st.page_link("pages/1_forecast_App.py", label="Forecast App")
        st.page_link("pages/2_Main_Competitors.py", label="Main Competitors")
        st.page_link("pages/3_Register.py", label="Register")

        # Admin solo si es admin
        if st.session_state.get("is_admin", False):
            st.page_link("pages/4_Admin.py", label="Admin")

        st.markdown("---")

        # --- Botón cerrar sesión (si lo tienes en app, lo dejamos igual) ---
        if st.button("Cerrar Sesión"):
            # Ajusta las keys según tu login
            for k in ["logged_in", "email", "is_admin", "is_approved"]:
                st.session_state.pop(k, None)
            st.rerun()

        # --- Logo inferior (IAN) + copyright ---
        st.markdown("---")
        if IAN_LOGO.exists():
            st.image(str(IAN_LOGO), use_container_width=True)

        st.sidebar.markdown(
            "<small>© Maresa. Todos los derechos reservados.</small>",
            unsafe_allow_html=True,
        )