import os
import streamlit as st

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data File Paths
DATA_PATHS = {
    "batsman": os.path.join(BASE_DIR, "odi_batsman.csv"),
    "all_rounder": os.path.join(BASE_DIR, "odi_all_rounders.csv"),
    "bowler": os.path.join(BASE_DIR, "odi_bowler.csv"),
    "yearwise": os.path.join(BASE_DIR, "yearwise_data.csv"),
}

# UI Styles
def apply_custom_styles():
    st.set_page_config(
        page_title="Cricket Analysis Dashboard",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob, .stDeployButton, .st-emotion-cache-1avcm0n {display: none !important;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Application Constants
MENU_OPTIONS = [
    "Format Wise Analysis",
    "Select Playing 11",
    "Player Comparison",
    "Player Analysis",
    "Smart Scout (AI)",
    "Ask Expert (AI)",
]

FORMATS = ['Odi', 'T20', 'Test']
