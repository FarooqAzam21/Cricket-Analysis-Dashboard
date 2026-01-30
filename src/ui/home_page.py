import streamlit as st

def show_home_page(all_players=None):
    """Display professional, fully responsive home page after login"""
    
    # Load data if not provided
    if all_players is None:
        from ..data_loader import load_all_data, _get_csv_cache_key
        try:
            all_players, _, _, _, _, _, _, _ = load_all_data(_csv_cache_key=_get_csv_cache_key())
        except:
            all_players = None
    # Comprehensive responsive CSS for professional sports styling
    st.markdown("""
    <style>
        /* Main header with green theme and responsive sizing */
        .home-header {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: clamp(30px, 8vw, 60px) 20px;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: clamp(20px, 5vw, 30px);
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
            animation: slideDown 0.6s ease;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .home-header h1 {
            font-size: clamp(28px, 7vw, 48px);
            margin: 0;
            font-weight: 800;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            letter-spacing: -0.5px;
        }
        
        .home-header p {
            font-size: clamp(14px, 4vw, 20px);
            margin: clamp(5px, 2vw, 15px) 0 0 0;
            opacity: 0.95;
            font-weight: 500;
        }
        
        /* Feature cards with glassmorphism */
        .feature-card {
            background: rgba(255, 255, 255, 0.95);
            border-left: 5px solid #10b981;
            border-radius: 12px;
            padding: clamp(16px, 4vw, 24px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: clamp(15px, 3vw, 20px);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
        }
        
        .feature-card h3 {
            color: #059669;
            margin: 0 0 8px 0;
            font-size: clamp(16px, 3vw, 22px);
            font-weight: 700;
        }
        
        .feature-card p {
            margin: 0;
            color: #4b5563;
            font-size: clamp(13px, 2.5vw, 15px);
            line-height: 1.6;
        }
        
        /* Stats boxes with responsive grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(250px, 100%), 1fr));
            gap: clamp(12px, 3vw, 20px);
            margin: clamp(20px, 5vw, 40px) 0;
        }
        
        .stats-box {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: clamp(20px, 5vw, 30px);
            border-radius: 14px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .stats-box:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 36px rgba(16, 185, 129, 0.35);
        }
        
        .stats-box-value {
            font-size: clamp(28px, 6vw, 42px);
            font-weight: 800;
            margin: clamp(8px, 2vw, 12px) 0;
            letter-spacing: -0.5px;
        }
        
        .stats-box-label {
            font-size: clamp(12px, 2.5vw, 16px);
            opacity: 0.9;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Quick links section */
        .quick-links {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: clamp(12px, 3vw, 16px);
            margin: clamp(20px, 5vw, 30px) 0;
        }
        
        .quick-link-btn {
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border: 2px solid #10b981;
            color: #059669;
            padding: clamp(16px, 4vw, 24px);
            border-radius: 12px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 700;
            font-size: clamp(14px, 2.5vw, 18px);
            text-decoration: none;
        }
        
        .quick-link-btn:hover {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);
        }
        
        /* Responsive layout for tablets and mobile */
        @media (max-width: 1024px) {
            .home-header {
                padding: 40px 16px;
                margin-bottom: 24px;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .home-header h1 {
                font-size: 28px;
            }
            
            .home-header p {
                font-size: 16px;
            }
            
            .feature-card {
                padding: 16px;
                margin-bottom: 12px;
                border-left: 4px solid #10b981;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
                gap: 12px;
            }
            
            .stats-box {
                padding: 20px;
            }
            
            .quick-links {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            .quick-link-btn {
                padding: 16px;
                font-size: 16px;
            }
        }
        
        @media (max-width: 480px) {
            .home-header {
                padding: 24px 12px;
                margin-bottom: 16px;
                border-radius: 12px;
            }
            
            .home-header h1 {
                font-size: 24px;
            }
            
            .home-header p {
                font-size: 13px;
                margin-top: 8px;
            }
            
            .feature-card {
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            
            .feature-card h3 {
                font-size: 16px;
                margin-bottom: 6px;
            }
            
            .feature-card p {
                font-size: 13px;
            }
            
            .stats-box {
                padding: 16px;
                border-radius: 12px;
            }
            
            .stats-box-value {
                font-size: 28px;
            }
            
            .stats-box-label {
                font-size: 12px;
            }
        }
        
        /* Print styles */
        @media print {
            .home-header {
                background: none;
                color: #333;
                border: 1px solid #ddd;
            }
            
            .stats-box {
                background: none;
                color: #333;
                border: 1px solid #ddd;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="home-header">
        <h1>🏏 Cricket Pro Analytics</h1>
        <p>Professional T20 Cricket Management & Fantasy League Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown(f"""
    ## Welcome, {st.session_state.username}! 👋
    
    Your complete cricket analysis and management platform is ready.
    """)
    
    # Quick stats with responsive grid - use real data if available
    if all_players is not None and len(all_players) > 0:
        num_players = len(all_players)
        num_teams = all_players['Team'].nunique() if 'Team' in all_players.columns else 0
        num_formats = all_players['Format'].nunique() if 'Format' in all_players.columns else 0
        num_tournaments = max(50, num_formats * 5)  # Estimate
    else:
        num_players = 1000
        num_teams = 20
        num_formats = 3
        num_tournaments = 50
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stats-box">
            <div class="stats-box-value">{num_players}</div>
            <div class="stats-box-label">Players Database</div>
        </div>
        <div class="stats-box">
            <div class="stats-box-value">{num_tournaments}+</div>
            <div class="stats-box-label">Active Tournaments</div>
        </div>
        <div class="stats-box">
            <div class="stats-box-value">{num_teams}</div>
            <div class="stats-box-label">International Teams</div>
        </div>
        <div class="stats-box">
            <div class="stats-box-value">{num_formats}</div>
            <div class="stats-box-label">Cricket Formats</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section - fully responsive
    st.markdown("---")
    st.markdown("## 🚀 Platform Features")
    
    # Responsive feature columns
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Cricket Analysis</h3>
            <p>In-depth player statistics, performance metrics, and format-wise analysis to make informed decisions.</p>
        </div>
        
        <div class="feature-card">
            <h3>🏆 Tournament Management</h3>
            <p>Create and manage T20 tournaments with flexible stages: Group Stage, Super 8, Knockouts, and Finals.</p>
        </div>
        
        <div class="feature-card">
            <h3>⚡ Fantasy Cricket</h3>
            <p>Build your dream teams before matches, earn points based on performance, and compete on leaderboards.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Smart Scout (AI)</h3>
            <p>AI-powered player recommendations and insights for team building and strategy optimization.</p>
        </div>
        
        <div class="feature-card">
            <h3>📈 Predictions & Comparisons</h3>
            <p>Advanced match predictions and player comparisons using machine learning and statistical analysis.</p>
        </div>
        
        <div class="feature-card">
            <h3>💬 Expert Chat (AI)</h3>
            <p>Ask cricket questions to our AI expert and get intelligent, context-aware recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About section
    st.markdown("---")
    st.markdown("## ℹ️ About Cricket Pro")
    
    about_text = """
    **Cricket Pro Analytics** is a comprehensive platform designed for cricket enthusiasts, team managers, 
    and fantasy league participants. Our platform combines advanced statistics, AI-powered insights, and 
    flexible tournament management to deliver a professional-grade cricket analysis and management experience.
    
    ### Our Mission
    To empower cricket professionals and enthusiasts with data-driven insights and tools for better decision-making 
    in player selection, team building, and match analysis.
    
    ### What We Offer
    - **Real-time Statistics**: Access to comprehensive player and team data
    - **Advanced Analytics**: Performance metrics, trends, and insights
    - **Tournament Platform**: Flexible tournament creation with multiple stage formats
    - **Fantasy League**: Compete with friends in fantasy cricket leagues
    - **AI Assistant**: Smart recommendations powered by machine learning
    
    ### Tournament Formats Supported
    - ✅ Group Stage (Round Robin)
    - ✅ Super 8 (Top 2 from each group)
    - ✅ Semi-Finals
    - ✅ Finals
    - ✅ Custom Knockouts
    
    ### Scoring System
    - **Batting**: Runs + 4s/6s bonuses + milestone bonuses (50, 100)
    - **Bowling**: Wickets + economy rate bonuses
    - **Fielding**: Catches, Run-outs, Stumpings
    - **NRR**: Net Run Rate calculation for group stage qualification
    """
    
    st.info(about_text)
    
    # Quick navigation section
    st.markdown("---")
    st.markdown("## 🔗 Quick Navigation")
    
    # Responsive navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns(3, gap="small")
    
    with nav_col1:
        if st.button("📊 Analysis", use_container_width=True, help="Go to cricket analysis"):
            st.session_state.page = "🏏 Cricket Analysis"
            st.rerun()
    
    with nav_col2:
        if st.button("🏆 Tournament", use_container_width=True, help="Manage tournaments"):
            st.session_state.page = "🏆 Tournament"
            st.rerun()
    
    with nav_col3:
        if st.button("🏏 Fantasy", use_container_width=True, help="Fantasy cricket"):
            st.session_state.page = "🏆 Tournament"
            st.session_state.fantasy_page = "Fantasy Cricket"
            st.rerun()
    
    # Footer with responsive styling
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: clamp(15px, 3vw, 20px); margin-top: 30px;">
        <p style="margin: 5px 0; font-size: clamp(12px, 2.5vw, 14px);">
            <strong>Cricket Pro Analytics v2.0</strong>
        </p>
        <p style="margin: 5px 0; font-size: clamp(11px, 2vw, 13px);">
            Professional Cricket Management Platform
        </p>
        <p style="margin: 5px 0; font-size: clamp(10px, 1.8vw, 12px); opacity: 0.8;">
            Designed for serious cricket professionals | Powered by Advanced Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_home_page()
