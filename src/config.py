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

def apply_custom_styles():
    """Apply premium 'Cricket Pro' CSS theme including Glassmorphism and modern typography."""
    premium_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Global Styling */
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
        }

        /* Hide Streamlit default UI elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none !important;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}

        /* Glassmorphism Containers */
        [data-testid="stExpander"], 
        div.stMetric, 
        form, 
        [data-testid="stForm"],
        div.stTabs [data-baseweb="tab-panel"] {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }

        /* Hover effects for cards */
        [data-testid="column"]:hover div.stMetric {
            transform: translateY(-5px);
            border: 1px solid rgba(16, 185, 129, 0.3) !important;
            box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.1) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f172a !important;
            background-image: linear-gradient(180deg, #1e3a8a 0%, #0f172a 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Nav Radio Styling */
        [data-testid="stSidebarNav"] {padding-top: 2rem;}
        div[data-testid="stSidebarUserContent"] .stRadio > label {
            color: #94a3b8 !important;
            font-weight: 600 !important;
        }

        /* Button Styling */
        .stButton > button {
            width: 100%;
            border-radius: 12px !important;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        }

        /* Metric Styling */
        [data-testid="stMetricValue"] {
            color: #10b981 !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px 12px 0 0 !important;
            color: #94a3b8 !important;
            padding: 10px 20px !important;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(16, 185, 129, 0.1) !important;
            color: #10b981 !important;
            border-bottom: 3px solid #10b981 !important;
        }

        /* Plots Styling */
        .js-plotly-plot .plotly .modebar {
            background-color: transparent !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {width: 8px;}
        ::-webkit-scrollbar-track {background: #0f172a;}
        ::-webkit-scrollbar-thumb {
            background: #1e3a8a;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {background: #10b981;}

        /* Mobile Fixes */
        @media (max-width: 768px) {
            [data-testid="stSidebar"] {width: 100% !important;}
            .stApp {padding: 0.5rem !important;}
        }
        </style>
    """
    st.markdown(premium_css, unsafe_allow_html=True)

# Application Constants
MENU_OPTIONS = [
    "Format Wise Analysis",
    "Select Playing 11",
    "Player Comparison",
    "Player Analysis",
    "🎯 Next Match Prediction",
    "📈 Yearly Performance Prediction",
    "Smart Scout (AI)",
    "Ask Expert (AI)",
]

FORMATS = ['Odi', 'T20', 'Test']
