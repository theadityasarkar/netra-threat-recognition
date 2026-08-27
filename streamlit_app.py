"""
Streamlit Community Cloud entry point.
"""

from __future__ import annotations

import streamlit as st

from src.config import APP_ICON, APP_NAME
from src.ui.theme import inject_theme

try:
    st.set_page_config(
        page_title=f"{APP_NAME} · Criminal Network Analysis",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

inject_theme()

from app import main

if __name__ == "__main__":
    main()
