import streamlit as st
from src.config import apply_custom_styles, MENU_OPTIONS
from src.data_loader import load_all_data, _get_csv_mtime_key
from src.ui.format_wise import render_format_analysis
from src.ui.team_builder import render_team_builder
from src.ui.comparison import render_comparison
from src.ui.analysis import render_player_analysis
from src.ui.predictions import render_predictions
from src.ui.smart_scout import render_smart_scout
from src.ui.ai_chat import render_ai_chat
from src.auth import login, signup
from src.database import init_db
import os

def render_auth():
    st.title("🔐 Dashboard Login")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                success, res = login(u, p)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error(res)
                    
    with tab2:
        with st.form("signup_form"):
            u = st.text_input("New Username")
            p = st.text_input("New Password", type="password")
            if st.form_submit_button("Create Account"):
                success, msg = signup(u, p)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

def main():
    # 0. Initialize DB and Auth state
    init_db()
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # 1. Set page config at the very beginning
    logo_path = os.path.join("assets", "logo.png")
    page_icon = logo_path if os.path.exists(logo_path) else "🏏"

    st.set_page_config(
        page_title="Cricket Pro Analytics",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    if not st.session_state.authenticated:
        render_auth()
        st.stop()

    # Apply Premium Theme
    apply_custom_styles()
    
    # 2. Sidebar elements
    with st.sidebar:
        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        
        st.markdown(f"### Welcome, **{st.session_state.username}**")
        st.markdown("---")
        
        menu = st.radio("Navigate to", MENU_OPTIONS)
        
        st.markdown("---")
        # Global Filters in Sidebar
        st.subheader("Team Preference")
        all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data(_csv_mtime_key=_get_csv_mtime_key())
        
        teams = ['All']
        if all_players is not None and 'Team' in all_players.columns:
            teams += sorted(all_players['Team'].dropna().unique().tolist())
        selected_team = st.selectbox("Select Display Team", teams)
        
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.clear()
            st.rerun()
            
        st.info("Developed by **Farooq Azam**")

    # 3. Main Dashboard Layout
    st.title("🏏 Cricket Pro Analytics")
    st.markdown("*Advanced Player Performance Insights & AI Scouts*")
    st.markdown("---")

    if all_players is None:
        st.stop()

    # 4. Routing
    if menu == "Format Wise Analysis":
        render_format_analysis(batsmen, all_rounders, df_bowler, wicket_keepers)
    elif menu == "Select Playing 11":
        render_team_builder(all_players)
    elif menu == "Player Comparison":
        render_comparison(all_players)
    elif menu == "Player Analysis":
        render_player_analysis(all_players)
    elif menu == "🎯 Next Match Prediction":
        from src.ui.predictions import render_next_match_prediction
        render_next_match_prediction(df_batsman, df_allrounder, df_bowler, wicket_keepers)
    elif menu == "📈 Yearly Performance Prediction":
        from src.ui.predictions import render_yearly_prediction
        render_yearly_prediction(year_wise)
    elif menu == "Smart Scout (AI)":
        render_smart_scout(all_players)
    elif menu == "Ask Expert (AI)":
        render_ai_chat(all_players)

if __name__ == "__main__":
    main()
