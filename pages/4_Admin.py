import streamlit as st
import pandas as pd
from sqlalchemy import text, bindparam
from sidebar import render_sidebar
from db import get_engine


# --- BLOQUEO DE SEGURIDAD (Solo Admin) ---
if st.session_state.get('is_admin', False) is False:
    st.error("Acceso denegado. Esta página es solo para administradores.")
    st.page_link("app.py", label="Ir a Inicio", icon="🏠")
    st.stop()
# --- FIN DE BLOQUEO ---

render_sidebar()

@st.cache_resource
def get_db_engine():
    try:
        return get_engine()
    except Exception as e:
        st.error(f"Error al crear el 'engine' de SQLAlchemy: {e}")
        st.stop()

engine = get_db_engine()

st.title("Panel de Administrador 🛡️")
st.subheader("Aprobación de Nuevos Usuarios")

# Cargar usuarios pendientes
try:
    pending_users = pd.read_sql(
        text("SELECT email, created_at FROM users WHERE is_approved = false ORDER BY created_at DESC"),
        engine
    )

    if pending_users.empty:
        st.success("No hay usuarios pendientes de aprobación.")
    else:
        st.warning(f"Hay {len(pending_users)} usuarios esperando aprobación:")

        pending_users['Aprobar'] = False
        edited_df = st.data_editor(
            pending_users,
            column_config={
                "email": "Correo",
                "created_at": "Fecha de Registro",
                "Aprobar": st.column_config.CheckboxColumn("¿Aprobar?", default=False)
            },
            hide_index=True,
            use_container_width=True
        )

        if st.button("Guardar Aprobaciones", type="primary"):
            users_to_approve = edited_df.loc[edited_df['Aprobar'] == True, 'email'].tolist()

            if not users_to_approve:
                st.info("No se seleccionó ningún usuario para aprobar.")
            else:
                with st.spinner("Procesando..."):
                    # IN con expanding (SQLAlchemy)
                    sql = (
                        text("UPDATE users SET is_approved = true WHERE email IN :emails")
                        .bindparams(bindparam("emails", expanding=True))
                    )

                    with engine.begin() as conn:  # abre transacción y hace commit automático
                        conn.execute(sql, {"emails": users_to_approve})

                    st.success(f"¡Se aprobaron {len(users_to_approve)} usuarios!")
                    st.rerun()

except Exception as e:
    st.error(f"Error al cargar usuarios: {e}")
    st.exception(e)