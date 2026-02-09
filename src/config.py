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
    "wc_players": os.path.join(BASE_DIR, "wc_players.csv"),
}

def apply_custom_styles():
    """Apply comprehensive RESPONSIVE 'Cricket Pro' CSS theme - Desktop & Mobile Optimized."""
    premium_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* ============= GLOBAL STYLING ============= */
        :root {
            --primary: #10b981;
            --primary-dark: #059669;
            --secondary: #3b82f6;
            --glass-bg: rgba(255, 255, 255, 0.85);
            --glass-border: rgba(16, 185, 129, 0.2);
            --shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1);
        }

        .stApp {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 50% 50%, #f0fdf4 0%, #dcfce7 100%);
            color: #1f2937 !important;
        }

        /* Smooth page transitions */
        .stApp > div {
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ============= HIDE DEFAULT ELEMENTS ============= */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none !important;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}

        /* ============= SIDEBAR STYLING ============= */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #064e3b 0%, #065f46 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.1) !important;
        }

        [data-testid="stSidebarUserContent"] * {
            color: #ecfdf5 !important;
        }

        /* ============= ELITE CARDS (GLASSMORPISM) ============= */
        .elite-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 24px;
            box-shadow: var(--shadow);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 20px;
        }

        .elite-card:hover {
            transform: translateY(-5px) scale(1.01);
            border-color: var(--primary);
            box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15);
        }

        /* Applied to standard Streamlit containers */
        [data-testid="stExpander"],
        div.stMetric,
        [data-testid="stForm"],
        div.stTabs [data-baseweb="tab-panel"],
        [data-testid="column"] > div {
            background: var(--glass-bg) !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 20px !important;
            box-shadow: var(--shadow) !important;
            transition: all 0.3s ease !important;
        }

        /* ============= PREMIUM BUTTONS ============= */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
            color: #e2e8f0 !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }

        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4) !important;
        }

        /* ============= SEARCH BAR (ELITE) ============= */
        .search-container {
            position: relative;
            margin-bottom: 2rem;
        }

        .search-input {
            width: 100%;
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 30px !important;
            padding: 12px 20px !important;
            color: #e2e8f0 !important;
            font-size: 0.9rem !important;
        }

        /* ============= PLAYER BATTLE TILES ============= */
        .battle-tile {
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            background: linear-gradient(145deg, #ffffff, #f0fdf4);
            border: 1px solid rgba(16, 185, 129, 0.1);
            text-align: center;
            padding: 20px;
        }

        .battle-avatar {
            width: 120px;
            height: 120px;
            border-radius: 60px;
            object-fit: cover;
            border: 4px solid var(--primary);
            margin: 0 auto 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }

        /* ============= METRICS & STATS ============= */
        [data-testid="stMetricValue"] {
            font-family: 'Poppins', sans-serif !important;
            background: linear-gradient(90deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
        }

        /* ============= TAB STYLING ============= */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            gap: 15px !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px !important;
            border: 1px solid transparent !important;
            padding: 8px 16px !important;
            background: rgba(255,255,255,0.5) !important;
        }

        .stTabs [aria-selected="true"] {
            background: white !important;
            border-color: var(--primary) !important;
            color: var(--primary-dark) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        }

        /* ============= RESPONSIVE FIXES ============= */
        @media (max-width: 768px) {
            .elite-card { padding: 16px; }
            h1 { font-size: 1.8rem !important; }
            .battle-avatar { width: 80px; height: 80px; }
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
