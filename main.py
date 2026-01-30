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
    """Render production-grade mobile-friendly login/signup page."""
    
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
        font-family: 'Poppins', sans-serif;
    }
    
    .login-card {
        background: white;
        border-radius: 20px;
        padding: 40px 30px;
        width: 100%;
        max-width: 420px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: slideUp 0.5s ease;
        backdrop-filter: blur(10px);
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
        letter-spacing: -0.5px;
    }
    
    .login-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin: 0;
        font-weight: 500;
    }
    
    .cricket-emoji {
        font-size: 60px;
        margin-bottom: 20px;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Responsive for tablets */
    @media (max-width: 768px) {
        .login-card {
            padding: 30px 20px;
            max-width: 100%;
            margin: 20px;
            border-radius: 16px;
        }
        .login-title {
            font-size: 24px;
        }
        .cricket-emoji {
            font-size: 48px;
        }
    }
    
    /* Responsive for mobile phones */
    @media (max-width: 480px) {
        .login-card {
            padding: 25px 16px;
            margin: 10px;
            border-radius: 14px;
        }
        .login-title {
            font-size: 20px;
        }
        .login-subtitle {
            font-size: 12px;
        }
        .cricket-emoji {
            font-size: 40px;
            margin-bottom: 15px;
        }
        .login-header {
            margin-bottom: 30px;
        }
    }
    </style>
    """
    
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Center the login card
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
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input(
                    "👤 Username",
                    placeholder="Enter your username",
                    key="login_username",
                    max_chars=50
                )
                p = st.text_input(
                    "🔑 Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                    max_chars=50
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.form_submit_button("🔓 Login", use_container_width=True):
                        if not u or not p:
                            st.error("❌ Please enter both username and password")
                        else:
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
            with st.form("signup_form", clear_on_submit=True):
                u = st.text_input(
                    "👤 Username",
                    placeholder="Choose a username",
                    key="signup_username",
                    max_chars=50
                )
                p = st.text_input(
                    "🔑 Password",
                    type="password",
                    placeholder="Choose a strong password (min 6 chars)",
                    key="signup_password",
                    max_chars=50
                )
                p_confirm = st.text_input(
                    "🔑 Confirm Password",
                    type="password",
                    placeholder="Confirm your password",
                    key="signup_confirm",
                    max_chars=50
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.form_submit_button("✅ Create Account", use_container_width=True):
                        if not u or not p:
                            st.error("❌ Username and password are required!")
                        elif len(p) < 6:
                            st.error("❌ Password must be at least 6 characters!")
                        elif p != p_confirm:
                            st.error("❌ Passwords don't match!")
                        else:
                            success, msg = signup(u, p)
                            if success:
                                st.success(f"✅ {msg}")
                                st.info("Now you can login with your new account!")
                            else:
                                st.error(f"❌ {msg}")

def main():
    """Main application entry point with responsive design."""
    
    # Initialize database and auth state
    init_db()
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Set page configuration
    logo_path = os.path.join("assets", "logo.png")
    page_icon = logo_path if os.path.exists(logo_path) else "🏏"

    st.set_page_config(
        page_title="Cricket Pro - Fantasy League",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'About': "🏏 Cricket Pro - T20 World Cup Fantasy League Manager",
            'Get help': None,
            'Report a bug': None
        }
    )
    
    # Show auth if not authenticated
    if not st.session_state.authenticated:
        render_auth()
        st.stop()

    # Apply responsive theme globally
    apply_custom_styles()
    
    # Responsive sidebar navigation
    with st.sidebar:
        # Header with responsive sizing
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px; animation: fadeIn 0.5s ease;'>
            <h1 style='color: white; margin: 0; font-size: clamp(24px, 5vw, 32px);'>🏏 Cricket Pro</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: clamp(11px, 2.5vw, 14px);'>T20 Fantasy League</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown("---")
        
        # Navigation menu options
        menu_options = [
            ("🏠", "Home"),
            ("📊", "Analysis"),
            ("🏆", "Tournament"),
        ]
        
        # Add admin option for admin users
        if st.session_state.username == 'admin':
            menu_options.append(("⚙️", "Admin"))
        
        # Responsive navigation layout
        st.markdown("### Navigation")
        
        # Create responsive columns for buttons
        nav_cols = st.columns(min(len(menu_options), 2))
        
        # Track page changes
        prev_page = st.session_state.get('prev_page', st.session_state.page)
        
        for idx, (icon, label) in enumerate(menu_options):
            col_idx = idx % len(nav_cols)
            with nav_cols[col_idx]:
                if st.button(
                    f"{icon}\n{label}",
                    use_container_width=True,
                    key=f"nav_{label}",
                    help=f"Go to {label}"
                ):
                    if label == "Home":
                        st.session_state.page = "🏠 Home"
                    elif label == "Analysis":
                        st.session_state.page = "🏏 Cricket Analysis"
                    elif label == "Tournament":
                        st.session_state.page = "🏆 Tournament"
                    elif label == "Admin":
                        st.session_state.page = "⚙️ Admin Panel"
                    st.session_state.prev_page = st.session_state.page
                    st.rerun()
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        st.info("🚀 **Cricket Pro** v2.0\n\nFantasy League Manager")
    
    # Main content area - responsive routing
    if st.session_state.page == "🏠 Home":
        # Load data for home page stats
        try:
            home_data, _, _, _, _, _, _, _ = load_all_data(_csv_cache_key=_get_csv_cache_key())
        except:
            home_data = None
        show_home_page(home_data)
    
    elif st.session_state.page == "🏏 Cricket Analysis":
        # Cricket analysis sub-menu
        st.markdown("""
        <style>
        .analysis-header {
            margin-bottom: 20px;
        }
        @media (max-width: 768px) {
            .analysis-header {
                margin-bottom: 15px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        analysis_options = [
            "Format Wise Analysis",
            "Select Playing 11",
            "Player Comparison",
            "Player Analysis",
            "🎯 Next Match Prediction",
            "📈 Yearly Performance Prediction",
            "Smart Scout (AI)",
            "Ask Expert (AI)"
        ]
        
        analysis_menu = st.sidebar.selectbox("Cricket Analysis", analysis_options, key="analysis_menu")
        
        # Load data
        all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data(_csv_cache_key=_get_csv_cache_key())
        
        if all_players is None or all_players.empty:
            st.error("❌ Failed to load player data")
            st.info(f"Debug: all_players is None={all_players is None}, empty={all_players.empty if all_players is not None else 'N/A'}")
            st.stop()
        
        st.success(f"✅ Loaded {len(all_players)} players successfully")
        # Sidebar filters for analysis - responsive
        st.sidebar.markdown("---")
        st.sidebar.subheader("Team Preference")
        teams = ['All']
        if all_players is not None and 'Team' in all_players.columns:
            teams += sorted(all_players['Team'].dropna().unique().tolist())
        selected_team = st.sidebar.selectbox("Select Team", teams, key="team_filter")
        
        # Route to appropriate analysis page
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
        # Tournament section
        st.title("🏆 T20 World Cup Fantasy League")
        
        tournament_options = ["Tournament Home", "Fantasy Cricket", "Leaderboard"]
        tournament_menu = st.sidebar.selectbox("Tournament", tournament_options, key="tournament_menu")
        
        if tournament_menu == "Tournament Home":
            show_tournament_home()
        elif tournament_menu == "Fantasy Cricket":
            show_fantasy_cricket()
        elif tournament_menu == "Leaderboard":
            show_leaderboard()
    
    elif st.session_state.page == "⚙️ Admin Panel":
        # Admin panel for authorized users only
        if st.session_state.username == 'admin':
            show_admin_panel()
        else:
            st.error("❌ Access denied. Admin panel is restricted.")
            st.stop()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by **Farooq Azam** | Cricket Pro v2.0")

if __name__ == "__main__":
    main()
