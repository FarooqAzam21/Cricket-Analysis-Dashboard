import streamlit as st

def show_home_page():
    """Display professional home page after login"""
    
    # Custom CSS for professional sports styling
    st.markdown("""
    <style>
        .home-header {
            background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
            padding: 60px 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .home-header h1 {
            font-size: 3em;
            margin: 0;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .home-header p {
            font-size: 1.3em;
            margin: 10px 0 0 0;
            opacity: 0.95;
        }
        .feature-card {
            background: white;
            border-left: 5px solid #1abc9c;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .feature-card h3 {
            color: #1abc9c;
            margin-top: 0;
        }
        .stats-box {
            background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 10px;
        }
        .stats-box-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stats-box-label {
            font-size: 0.9em;
            opacity: 0.9;
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
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-box-value">1000+</div>
            <div class="stats-box-label">Players</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-box-value">50+</div>
            <div class="stats-box-label">Tournaments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-box-value">100+</div>
            <div class="stats-box-label">Teams</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-box">
            <div class="stats-box-value">5000+</div>
            <div class="stats-box-label">Fantasy Teams</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("---")
    st.markdown("## 🚀 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Cricket Analysis</h3>
            <p>In-depth player statistics, performance metrics, and format-wise analysis to make informed decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🏆 Tournament Management</h3>
            <p>Create and manage T20 tournaments with flexible stages: Group Stage, Super 8, Knockouts, and Finals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Fantasy Cricket</h3>
            <p>Build your dream teams before matches, earn points based on performance, and compete on leaderboards.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Smart Scout</h3>
            <p>AI-powered player recommendations and insights for team building and strategy optimization.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Predictions & Comparisons</h3>
            <p>Advanced match predictions and player comparisons using machine learning and statistical analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Admin Panel</h3>
            <p>Full control over tournaments, teams, players, and match results with flexible scheduling.</p>
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
    
    # Quick links
    st.markdown("---")
    st.markdown("## 🔗 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Go to Analysis", use_container_width=True):
            st.session_state.page = "Cricket Analysis"
            st.rerun()
    
    with col2:
        if st.button("🏆 Manage Tournaments", use_container_width=True):
            st.session_state.page = "Admin Panel"
            st.rerun()
    
    with col3:
        if st.button("🏏 Fantasy Cricket", use_container_width=True):
            st.session_state.page = "Fantasy Cricket"
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p>Cricket Pro Analytics v2.0 | Professional Cricket Management Platform</p>
        <p>Designed for serious cricket professionals | Powered by Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_home_page()
