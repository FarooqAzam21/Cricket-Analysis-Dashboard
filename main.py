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
from src.ui.home_page import show_home_page
from src.ui.warehouse_modeling import render_warehouse_modeling
from src.database import init_db
import os

def main():
    """Main application entry point with responsive design."""
    
    # Initialize database and default open-access state
    init_db()
    if 'username' not in st.session_state:
        st.session_state.username = "Analyst"
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Set page configuration
    logo_path = os.path.join("assets", "logo.png")
    page_icon = logo_path if os.path.exists(logo_path) else "🏏"

    st.set_page_config(
        page_title="Analytics Warehouse - Modeling & BI",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'About': "Data Warehousing, Data Modeling, Visualization and AI Analytics",
            'Get help': None,
            'Report a bug': None
        }
    )
    
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
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>Analytics Warehouse</h2>
            <p style='color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;'>Warehouse - Modeling - BI</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User Profile Card
        st.markdown(f"""
        <div class="user-profile">
            <div class="user-avatar">👤</div>
            <p class="user-name">{st.session_state.username}</p>
            <p class="user-role">{'Administrator' if st.session_state.username == 'admin' else 'Analytics User'}</p>
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
            {"icon": "DW", "label": "Warehouse", "page": "Warehouse Modeling"},
            {"icon": "📊", "label": "Visual Analytics", "page": "🏏 Cricket Analysis"},
        ]
        
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
            {"icon": "DW", "label": "Data Model", "page": "Warehouse Modeling", "sub": None},
            {"icon": "🧠", "label": "Smart Scout AI", "page": "🏏 Cricket Analysis", "sub": "Smart Scout (AI)"},
            {"icon": "💬", "label": "Ask Expert AI", "page": "🏏 Cricket Analysis", "sub": "Ask Expert (AI)"},
        ]
        
        for action in quick_actions:
            if st.button(
                f"{action['icon']} {action['label']}",
                use_container_width=True,
                key=f"quick_{action['label']}"
            ):
                st.session_state.page = action['page']
                if action['sub']:
                    st.session_state.analysis_menu = action['sub']
                st.rerun()
        
        st.markdown("---")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px;'>
            <p style='color: #10b981; font-weight: 600; margin: 0; font-size: 12px;'>Analytics Warehouse</p>
            <p style='color: #64748b; margin: 4px 0 0 0; font-size: 10px;'>v2.5 • BI Modeling</p>
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

    elif st.session_state.page == "Warehouse Modeling":
        all_players, _, _, _, year_wise, _, _, _ = load_all_data(_csv_cache_key=_get_csv_cache_key())

        if all_players is None or all_players.empty:
            st.error("❌ Failed to load warehouse data")
            st.stop()

        render_warehouse_modeling(all_players, year_wise)
    
    elif st.session_state.page == "🏏 Cricket Analysis":
        # Analytics workbench sub-menu
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
            "Warehouse & Data Modeling",
            "Format Wise Analysis",
            "Select Playing 11",
            "Player Comparison",
            "Player Analysis",
            "🎯 Next Match Prediction",
            "📈 Yearly Performance Prediction",
            "Smart Scout (AI)",
            "Ask Expert (AI)"
        ]
        
        analysis_menu = st.sidebar.selectbox("Analytics Workbench", analysis_options, key="analysis_menu")
        
        # Load data
        all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data(_csv_cache_key=_get_csv_cache_key())
        
        if all_players is None or all_players.empty:
            st.error("❌ Failed to load player data")
            st.info(f"Debug: all_players is None={all_players is None}, empty={all_players.empty if all_players is not None else 'N/A'}")
            st.stop()
        
        st.success(f"✅ Loaded {len(all_players)} warehouse records successfully")
        # Sidebar filters for analysis - responsive
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dimension Filter")
        teams = ['All']
        if all_players is not None and 'Team' in all_players.columns:
            teams += sorted(all_players['Team'].dropna().unique().tolist())
        selected_team = st.sidebar.selectbox("Select Team", teams, key="team_filter")
        
        # Route to appropriate analysis page
        if analysis_menu == "Warehouse & Data Modeling":
            render_warehouse_modeling(all_players, year_wise)
        elif analysis_menu == "Format Wise Analysis":
            render_format_analysis(batsmen, all_rounders, df_bowler, wicket_keepers)
        elif analysis_menu == "Select Playing 11":
            render_team_builder(all_players)
        elif analysis_menu == "Player Comparison":
            render_comparison(all_players)
        elif analysis_menu == "Player Analysis":
            render_player_analysis(all_players, year_wise)
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
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by **Farooq Azam** | Analytics Warehouse v2.0")

if __name__ == "__main__":
    main()
