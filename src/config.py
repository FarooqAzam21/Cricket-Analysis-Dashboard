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
    """Apply comprehensive RESPONSIVE 'Cricket Pro' CSS theme - Desktop & Mobile Optimized."""
    premium_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

        /* ============= GLOBAL STYLING ============= */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            scroll-behavior: smooth;
        }

        .stApp {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            color: #1f2937;
            line-height: 1.6;
        }

        /* ============= HIDE DEFAULT ELEMENTS ============= */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none !important;}
        [data-testid="stHeader"] {background: rgba(0,0,0,0) !important;}

        /* ============= SIDEBAR STYLING ============= */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10b981 0%, #059669 100%) !important;
            border-right: 3px solid #059669 !important;
            min-height: 100vh;
        }

        [data-testid="stSidebarUserContent"] {
            color: white !important;
            padding: 1rem !important;
        }

        [data-testid="stSidebarUserContent"] * {
            color: white !important;
        }

        [data-testid="stSidebarUserContent"] h1,
        [data-testid="stSidebarUserContent"] h2,
        [data-testid="stSidebarUserContent"] h3,
        [data-testid="stSidebarUserContent"] p {
            color: white !important;
            margin: 0.5rem 0;
        }

        /* ============= BUTTON STYLING ============= */
        .stButton > button {
            width: 100%;
            border-radius: 12px !important;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 12px 20px !important;
            font-size: 0.95rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
            cursor: pointer !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* ============= FORM ELEMENTS ============= */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {
            border: 2px solid rgba(16, 185, 129, 0.3) !important;
            border-radius: 10px !important;
            padding: 12px !important;
            font-size: 0.95rem !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {
            border: 2px solid #10b981 !important;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
            outline: none !important;
        }

        /* ============= CONTAINERS & CARDS ============= */
        [data-testid="stExpander"],
        div.stMetric,
        form,
        [data-testid="stForm"],
        div.stTabs [data-baseweb="tab-panel"],
        [data-testid="column"] {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(12px) !important;
            border: 2px solid rgba(16, 185, 129, 0.2) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1) !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="column"]:hover {
            transform: translateY(-4px);
            border: 2px solid rgba(16, 185, 129, 0.5) !important;
            box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.2) !important;
        }

        /* ============= METRICS & VALUES ============= */
        [data-testid="stMetricValue"] {
            color: #10b981 !important;
            font-size: 2rem !important;
            font-weight: 800 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #6b7280 !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem !important;
        }

        /* ============= TABS STYLING ============= */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent !important;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: rgba(16, 185, 129, 0.05) !important;
            border-radius: 10px !important;
            color: #059669 !important;
            padding: 10px 20px !important;
            border: 2px solid transparent !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(16, 185, 129, 0.15) !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(16, 185, 129, 0.2) !important;
            color: #059669 !important;
            border-bottom: 3px solid #10b981 !important;
        }

        /* ============= MESSAGES ============= */
        .stSuccess {
            background-color: rgba(16, 185, 129, 0.1) !important;
            border-left: 5px solid #10b981 !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }

        .stError {
            background-color: rgba(239, 68, 68, 0.1) !important;
            border-left: 5px solid #ef4444 !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }

        .stWarning {
            background-color: rgba(251, 191, 36, 0.1) !important;
            border-left: 5px solid #fbbf24 !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }

        .stInfo {
            background-color: rgba(59, 130, 246, 0.1) !important;
            border-left: 5px solid #3b82f6 !important;
            border-radius: 8px !important;
            padding: 15px !important;
        }

        /* ============= HEADERS ============= */
        h1, h2, h3, h4, h5, h6 {
            color: #059669 !important;
            font-weight: 700 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }

        h1 {
            font-size: 2.5rem !important;
            border-bottom: 3px solid #10b981;
            padding-bottom: 0.5rem;
        }

        h2 {
            font-size: 2rem !important;
        }

        h3 {
            font-size: 1.5rem !important;
        }

        /* ============= SCROLLBAR ============= */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #f0fdf4;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #10b981, #059669);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #059669, #047857);
        }

        /* ============= DIVIDERS ============= */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #10b981, transparent);
            margin: 2rem 0;
        }

        /* ============= PLOTS & CHARTS ============= */
        .js-plotly-plot .plotly .modebar {
            background-color: transparent !important;
        }

        .stPlotlyChart {
            background: rgba(255, 255, 255, 0.6) !important;
            border-radius: 12px !important;
            padding: 15px !important;
        }

        /* ============= RESPONSIVE DESIGN ============= */
        
        /* Tablet (768px) */
        @media (max-width: 1024px) {
            .stApp {
                padding: 1rem !important;
            }

            h1 {
                font-size: 2rem !important;
            }

            h2 {
                font-size: 1.5rem !important;
            }

            [data-testid="stExpander"],
            div.stMetric,
            form,
            [data-testid="stForm"],
            div.stTabs [data-baseweb="tab-panel"] {
                padding: 15px !important;
            }

            .stButton > button {
                padding: 10px 16px !important;
                font-size: 0.9rem !important;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.5rem !important;
            }
        }

        /* Mobile (< 768px) */
        @media (max-width: 768px) {
            /* Container adjustments */
            .stApp {
                padding: 0.5rem !important;
                background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            }

            [data-testid="stAppViewContainer"] {
                padding-left: 0 !important;
                padding-right: 0 !important;
            }

            /* Sidebar on mobile */
            [data-testid="stSidebar"] {
                width: 100% !important;
                position: relative !important;
            }

            [data-testid="stSidebarUserContent"] {
                padding: 0.8rem !important;
            }

            /* Headers */
            h1 {
                font-size: 1.5rem !important;
                padding-bottom: 0.3rem;
                margin-bottom: 0.8rem !important;
            }

            h2 {
                font-size: 1.2rem !important;
                margin-bottom: 0.6rem !important;
            }

            h3 {
                font-size: 1.1rem !important;
                margin-bottom: 0.5rem !important;
            }

            /* Containers */
            [data-testid="stExpander"],
            div.stMetric,
            form,
            [data-testid="stForm"],
            div.stTabs [data-baseweb="tab-panel"],
            [data-testid="column"] {
                padding: 12px !important;
                border-radius: 12px !important;
                margin-bottom: 0.8rem !important;
            }

            /* Buttons */
            .stButton > button {
                padding: 12px 16px !important;
                font-size: 0.9rem !important;
                border-radius: 10px !important;
                min-height: 44px !important;
            }

            /* Inputs */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select,
            .stNumberInput > div > div > input {
                padding: 12px !important;
                font-size: 1rem !important;
                min-height: 44px !important;
            }

            /* Metrics */
            [data-testid="stMetricValue"] {
                font-size: 1.8rem !important;
            }

            [data-testid="stMetricLabel"] {
                font-size: 0.75rem !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab"] {
                padding: 8px 12px !important;
                font-size: 0.85rem !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            /* Messages */
            .stSuccess,
            .stError,
            .stWarning,
            .stInfo {
                padding: 12px !important;
                margin: 0.8rem 0 !important;
                border-radius: 8px !important;
            }

            /* Columns stacking */
            [data-testid="column"] {
                width: 100% !important;
            }

            /* Reduce animation on mobile */
            * {
                transition: all 0.2s ease !important;
            }

            /* Better spacing */
            .stMarkdown {
                margin: 0.5rem 0 !important;
            }

            /* Reduce form padding */
            form {
                gap: 0.8rem !important;
            }

            /* Improve table visibility */
            .streamlit-table {
                font-size: 0.85rem !important;
            }
        }

        /* Small phones (< 480px) */
        @media (max-width: 480px) {
            .stApp {
                padding: 0.25rem !important;
            }

            h1 {
                font-size: 1.3rem !important;
            }

            h2 {
                font-size: 1.1rem !important;
            }

            [data-testid="stExpander"],
            div.stMetric,
            form {
                padding: 10px !important;
            }

            .stButton > button {
                padding: 10px 12px !important;
                font-size: 0.85rem !important;
            }

            [data-testid="stMetricValue"] {
                font-size: 1.5rem !important;
            }
        }

        /* ============= ACCESSIBILITY ============= */
        a {
            color: #10b981;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        a:hover {
            color: #059669;
            text-decoration: underline;
        }

        /* Focus states for keyboard navigation */
        button:focus,
        input:focus,
        select:focus {
            outline: 2px solid #10b981;
            outline-offset: 2px;
        }

        /* ============= PRINT STYLES ============= */
        @media print {
            [data-testid="stSidebar"],
            .stButton > button {
                display: none !important;
            }
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
