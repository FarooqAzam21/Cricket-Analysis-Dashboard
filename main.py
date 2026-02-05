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
    """Render modern, production-grade login/signup page with glass-morphism design."""
    
    login_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .auth-wrapper {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        overflow-y: auto;
        font-family: 'Poppins', sans-serif;
    }
    
    .auth-background {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
        animation: pulse 8s ease-in-out infinite;
        pointer-events: none;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 0.8; }
    }
    
    .cricket-pattern {
        position: fixed;
        width: 100%;
        height: 100%;
        opacity: 0.03;
        background-image: repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(255,255,255,.1) 35px, rgba(255,255,255,.1) 70px);
        pointer-events: none;
    }
    
    .auth-container {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 20px;
        position: relative;
        z-index: 1;
    }
    
    .auth-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 48px 40px;
        width: 100%;
        max-width: 480px;
        box-shadow: 
            0 0 60px rgba(16, 185, 129, 0.2),
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 0 0 1px rgba(255, 255, 255, 0.2);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    
    .auth-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #3b82f6, #10b981);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(40px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .cricket-logo {
        font-size: 72px;
        margin-bottom: 16px;
        display: inline-block;
        animation: bounce 2s ease-in-out infinite;
        filter: drop-shadow(0 4px 12px rgba(16, 185, 129, 0.3));
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-12px) rotate(5deg); }
    }
    
    .auth-title {
        color: #0f172a;
        font-size: 36px;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #0f172a 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .auth-subtitle {
        color: #64748b;
        font-size: 15px;
        margin: 0;
        font-weight: 500;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 12px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .auth-card {
            padding: 36px 24px;
            max-width: 100%;
            border-radius: 20px;
        }
        .auth-title {
            font-size: 28px;
        }
        .cricket-logo {
            font-size: 56px;
        }
    }
    
    @media (max-width: 480px) {
        .auth-container {
            padding: 16px;
        }
        .auth-card {
            padding: 28px 20px;
            border-radius: 16px;
        }
        .auth-title {
            font-size: 24px;
        }
        .auth-subtitle {
            font-size: 13px;
        }
        .cricket-logo {
            font-size: 48px;
            margin-bottom: 12px;
        }
        .auth-header {
            margin-bottom: 30px;
        }
    }
    
    /* Form Styling */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }
    
    .stButton > button {
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        border: none !important;
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }
    </style>
    """
    
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Background layers
    st.markdown("""
    <div class="auth-wrapper">
        <div class="auth-background"></div>
        <div class="cricket-pattern"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="auth-container">
        <div class="auth-card">
        <div class="auth-header">
            <div class="cricket-logo">🏏</div>
            <h1 class="auth-title">Cricket Pro</h1>
            <p class="auth-subtitle">T20 World Cup Fantasy League</p>
            <span class="feature-badge">⚡ Elite Analytics Platform</span>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔓 Login", "✨ Sign Up"])
        
        with tab1:
            st.markdown("### 👋 Welcome Back!")
            st.caption("Enter your credentials to access your fantasy league")
            
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    key="login_username",
                    max_chars=50
                )
                p = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                    max_chars=50
                )
                
                col_submit = st.columns([1, 2, 1])
                with col_submit[1]:
                    if st.form_submit_button("🚀 Login", use_container_width=True):
                        if not u or not p:
                            st.error("❌ Please enter both username and password")
                        else:
                            success, res = login(u, p)
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.username = u
                                st.success("✅ Login successful!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ {res}")
                                
        with tab2:
            st.markdown("### ✨ Join Cricket Pro")
            st.caption("Create your account to start building fantasy teams")
            
            with st.form("signup_form", clear_on_submit=True):
                u = st.text_input(
                    "Username",
                    placeholder="Choose a unique username",
                    key="signup_username",
                    max_chars=50
                )
                p = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Minimum 6 characters",
                    key="signup_password",
                    max_chars=50
                )
                p_confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password",
                    key="signup_confirm",
                    max_chars=50
                )
                
                col_submit = st.columns([1, 2, 1])
                with col_submit[1]:
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
                                st.info("💡 Switch to the Login tab to sign in!")
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
    
    # Enhanced sidebar with custom icons and collapsible sections
    sidebar_css = """
    <style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid transparent;
    }
    
    .sidebar-nav-item:hover {
        background: rgba(16, 185, 129, 0.15);
        border-left-color: #10b981;
        transform: translateX(4px);
    }
    
    .sidebar-nav-item.active {
        background: rgba(16, 185, 129, 0.2);
        border-left-color: #10b981;
    }
    
    .nav-icon {
        width: 24px;
        height: 24px;
        margin-right: 12px;
        filter: brightness(0) invert(1);
        opacity: 0.9;
    }
    
    .nav-label {
        color: #e2e8f0;
        font-weight: 500;
        font-size: 14px;
    }
    
    .section-header {
        color: #64748b;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 10px 0;
        padding: 0 16px;
    }
    
    .user-profile {
        background: rgba(16, 185, 129, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .user-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981, #059669);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    .user-name {
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
        margin: 0;
    }
    
    .user-role {
        color: #94a3b8;
        font-size: 12px;
        margin: 0;
    }
    </style>
    """
    
    st.markdown(sidebar_css, unsafe_allow_html=True)
    
    # Responsive sidebar navigation
    with st.sidebar:
        # Modern header
        st.markdown("""
        <div style='text-align: center; margin-bottom: 24px;'>
            <div style='font-size: 48px; margin-bottom: 8px;'>🏏</div>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>Cricket Pro</h2>
            <p style='color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;'>Fantasy League Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User Profile Card
        st.markdown(f"""
        <div class="user-profile">
            <div class="user-avatar">👤</div>
            <p class="user-name">{st.session_state.username}</p>
            <p class="user-role">{'Administrator' if st.session_state.username == 'admin' else 'Fantasy Manager'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Search
        st.markdown('<p class="section-header">🔍 Quick Search</p>', unsafe_allow_html=True)
        try:
            from src.data_loader import load_all_data, _get_csv_cache_key
            all_players_search, _, _, _, _, _, _, _ = load_all_data(_csv_cache_key=_get_csv_cache_key())
            
            if all_players_search is not None and not all_players_search.empty:
                search_list = [""] + sorted(all_players_search['player'].unique().tolist())
                selected_search = st.selectbox(
                    "Find any player...",
                    options=search_list,
                    index=0,
                    key="global_player_search",
                    label_visibility="collapsed"
                )
                
                if selected_search:
                    st.session_state.page = "🏏 Cricket Analysis"
                    st.session_state.analysis_menu = "Player Analysis"
                    st.session_state.preselected_player = selected_search
                    st.toast(f"Navigating to {selected_search}'s profile...", icon="🚀")
                    st.rerun()
        except Exception:
            pass
            
        st.markdown("---")
        
        # Main Navigation
        st.markdown('<p class="section-header">⚡ Main Menu</p>', unsafe_allow_html=True)
        
        # Navigation with custom icons (using Unicode dark icons)
        nav_items = [
            {"icon": "🏠", "label": "Home", "page": "🏠 Home"},
            {"icon": "📊", "label": "Analytics", "page": "🏏 Cricket Analysis"},
            {"icon": "🏆", "label": "Tournament", "page": "🏆 Tournament"},
        ]
        
        # Add admin option
        if st.session_state.username == 'admin':
            nav_items.append({"icon": "⚙️", "label": "Admin Panel", "page": "⚙️ Admin Panel"})
        
        for item in nav_items:
            if st.button(
                f"{item['icon']} {item['label']}",
                use_container_width=True,
                key=f"nav_{item['label']}"
            ):
                st.session_state.page = item['page']
                st.rerun()
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown('<p class="section-header">⚡ Quick Access</p>', unsafe_allow_html=True)
        
        quick_actions = [
            {"icon": "🎯", "label": "Create Fantasy Team", "page": "🏆 Tournament", "sub": "fantasy"},
            {"icon": "📈", "label": "Leaderboard", "page": "🏆 Tournament", "sub": "leaderboard"},
        ]
        
        for action in quick_actions:
            if st.button(
                f"{action['icon']} {action['label']}",
                use_container_width=True,
                key=f"quick_{action['label']}"
            ):
                st.session_state.page = action['page']
                if action['sub'] == 'fantasy':
                    st.session_state.tournament_tab = 'fantasy'
                elif action['sub'] == 'leaderboard':
                    st.session_state.tournament_tab = 'leaderboard'
                st.rerun()
        
        st.markdown("---")
        
        # Logout  
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.success("Logged out successfully!")
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px;'>
            <p style='color: #10b981; font-weight: 600; margin: 0; font-size: 12px;'>🚀 Cricket Pro Elite</p>
            <p style='color: #64748b; margin: 4px 0 0 0; font-size: 10px;'>v2.5 • Premium Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
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
