import streamlit as st
from src.config import apply_custom_styles, MENU_OPTIONS
from src.data_loader import load_all_data, _get_csv_cache_key
from src.ui.format_wise import render_format_analysis
from src.ui.team_builder import render_team_builder
from src.ui.comparison import render_comparison
from src.ui.analysis import render_player_analysis
from src.ui.predictions import render_predictions
from src.ui.smart_scout import render_smart_scout
from src.ui.ai_chat import render_ai_chat
from src.ui.tournament_home import show_tournament_home
from src.ui.fantasy_cricket import show_fantasy_cricket
from src.ui.leaderboard import show_leaderboard
from src.ui.admin_tournament import show_admin_panel
from src.ui.home_page import show_home_page
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
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
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
        
        # Show admin option only for admin user
        menu_options = ["🏠 Home", "🏏 Cricket Analysis", "🏆 Tournament"]
        if st.session_state.username == 'admin':
            menu_options = ["🏠 Home", "🏏 Cricket Analysis", "🏆 Tournament", "⚙️ Admin Panel"]
        
        menu = st.radio("Navigate to", menu_options)
        st.session_state.page = menu
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    # 3. Main routing based on menu selection
    if st.session_state.page == "🏠 Home":
        show_home_page()
    
    elif st.session_state.page == "🏏 Cricket Analysis":
        # Sub-menu for cricket analysis
        analysis_options = [
            "Format Wise Analysis", "Select Playing 11", "Player Comparison",
            "Player Analysis", "🎯 Next Match Prediction",
            "📈 Yearly Performance Prediction", "Smart Scout (AI)", "Ask Expert (AI)"
        ]
        
        analysis_menu = st.sidebar.selectbox("Cricket Analysis", analysis_options)
        
        all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data(_csv_cache_key=_get_csv_cache_key())
        
        if all_players is None:
            st.stop()
        
        # Sidebar filters for analysis
        st.sidebar.markdown("---")
        st.sidebar.subheader("Team Preference")
        teams = ['All']
        if all_players is not None and 'Team' in all_players.columns:
            teams += sorted(all_players['Team'].dropna().unique().tolist())
        selected_team = st.sidebar.selectbox("Select Display Team", teams)
        
        # Routing for cricket analysis features
        if analysis_menu == "Format Wise Analysis":
            render_format_analysis(batsmen, all_rounders, df_bowler, wicket_keepers)
        elif analysis_menu == "Select Playing 11":
            render_team_builder(all_players)
        elif analysis_menu == "Player Comparison":
            render_comparison(all_players)
        elif analysis_menu == "Player Analysis":
            render_player_analysis(all_players)
        elif analysis_menu == "🎯 Next Match Prediction":
            from src.ui.predictions import render_next_match_prediction
            render_next_match_prediction(df_batsman, df_allrounder, df_bowler, wicket_keepers)
        elif analysis_menu == "📈 Yearly Performance Prediction":
            from src.ui.predictions import render_yearly_prediction
            render_yearly_prediction(year_wise)
        elif analysis_menu == "Smart Scout (AI)":
            render_smart_scout(all_players)
        elif analysis_menu == "Ask Expert (AI)":
            render_ai_chat(all_players)
    
    elif st.session_state.page == "🏆 Tournament":
        # Tournament menu for all users
        st.title("🏆 T20 World Cup Fantasy League")
        tournament_options = ["Tournament Home", "Fantasy Cricket", "Leaderboard"]
        tournament_menu = st.sidebar.selectbox("Tournament", tournament_options)
        
        if tournament_menu == "Tournament Home":
            show_tournament_home()
        elif tournament_menu == "Fantasy Cricket":
            show_fantasy_cricket()
        elif tournament_menu == "Leaderboard":
            show_leaderboard()
    
    elif st.session_state.page == "⚙️ Admin Panel":
        # Admin panel - only for admin
        show_admin_panel()
        
    st.sidebar.info("Developed by **Farooq Azam**")

if __name__ == "__main__":
    main()
