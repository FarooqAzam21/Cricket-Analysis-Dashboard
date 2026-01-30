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
    """Render improved mobile-friendly login/signup page with green and white theme."""
    # Apply custom theme for login page
    login_css = """
    <style>
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 20px;
        font-family: 'Inter', sans-serif;
    }
    
    .login-card {
        background: white;
        border-radius: 20px;
        padding: 40px 30px;
        width: 100%;
        max-width: 420px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .login-title {
        color: #059669;
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 10px 0;
    }
    
    .login-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin: 0;
    }
    
    .cricket-emoji {
        font-size: 60px;
        margin-bottom: 20px;
    }
    
    @media (max-width: 600px) {
        .login-card {
            padding: 30px 20px;
        }
        .login-title {
            font-size: 24px;
        }
        .cricket-emoji {
            font-size: 48px;
        }
    }
    </style>
    """
    
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
        <div class="login-card">
        <div class="login-header">
            <div class="cricket-emoji">🏏</div>
            <h1 class="login-title">Cricket Pro</h1>
            <p class="login-subtitle">T20 World Cup Fantasy League</p>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔓 Login", "✍️ Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            with st.form("login_form"):
                u = st.text_input("👤 Username", placeholder="Enter your username")
                p = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.form_submit_button("🔓 Login", use_container_width=True):
                        success, res = login(u, p)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = u
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error(f"❌ {res}")
                            
        with tab2:
            st.markdown("### Create New Account")
            with st.form("signup_form"):
                u = st.text_input("👤 Username", placeholder="Choose a username", key="signup_username")
                p = st.text_input("🔑 Password", type="password", placeholder="Choose a strong password", key="signup_password")
                p_confirm = st.text_input("🔑 Confirm Password", type="password", placeholder="Confirm your password", key="signup_confirm")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.form_submit_button("✅ Create Account", use_container_width=True):
                        if p != p_confirm:
                            st.error("❌ Passwords don't match!")
                        else:
                            success, msg = signup(u, p)
                            if success:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")

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

    # Apply Green & White Premium Theme
    apply_custom_styles()
    
    # 2. Sidebar elements - IMPROVED MOBILE DESIGN
    with st.sidebar:
        # Logo and header
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: #10b981; margin: 0; font-size: 28px;'>🏏 Cricket Pro</h1>
            <p style='color: #6b7280; margin: 5px 0 0 0; font-size: 12px;'>T20 Fantasy League</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown("---")
        
        # Show admin option only for admin user
        menu_options = [
            ("🏠", "Home"),
            ("📊", "Analysis"),
            ("🏆", "Tournament"),
        ]
        
        if st.session_state.username == 'admin':
            menu_options.append(("⚙️", "Admin"))
        
        # Mobile-friendly menu with better icons
        st.markdown("### Navigation")
        cols = st.columns(len(menu_options))
        
        for idx, (icon, label) in enumerate(menu_options):
            with cols[idx]:
                if st.button(f"{icon}\n{label}", use_container_width=True, key=f"nav_{label}"):
                    if label == "Home":
                        st.session_state.page = "🏠 Home"
                    elif label == "Analysis":
                        st.session_state.page = "🏏 Cricket Analysis"
                    elif label == "Tournament":
                        st.session_state.page = "🏆 Tournament"
                    elif label == "Admin":
                        st.session_state.page = "⚙️ Admin Panel"
                    st.rerun()
        
        st.markdown("---")
        
        # Logout button with better styling
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        st.info("🚀 **Cricket Pro** - Fantasy League Manager")
    
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
