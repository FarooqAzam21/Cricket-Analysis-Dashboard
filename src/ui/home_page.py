import streamlit as st


def show_home_page(all_players=None):
    """Display the data warehousing and analytics landing page after login."""

    if all_players is None:
        from ..data_loader import load_all_data, _get_csv_cache_key

        try:
            all_players, _, _, _, _, _, _, _ = load_all_data(_csv_cache_key=_get_csv_cache_key())
        except Exception:
            all_players = None

    st.markdown(
        """
        <style>
            .home-header {
                background: linear-gradient(135deg, #10b981 0%, #2563eb 100%);
                padding: clamp(28px, 7vw, 56px) 20px;
                border-radius: 14px;
                color: #f8fafc;
                text-align: center;
                margin-bottom: clamp(18px, 4vw, 28px);
                box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
            }
            .home-header h1 {
                font-size: clamp(28px, 7vw, 48px);
                margin: 0;
                font-weight: 800;
            }
            .home-header p {
                font-size: clamp(14px, 4vw, 20px);
                margin: 12px 0 0 0;
                opacity: 0.95;
                font-weight: 500;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(min(230px, 100%), 1fr));
                gap: clamp(12px, 3vw, 20px);
                margin: clamp(20px, 5vw, 36px) 0;
            }
            .stats-box {
                background: #ffffff;
                color: #0f172a;
                padding: clamp(18px, 5vw, 28px);
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(16, 185, 129, 0.22);
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }
            .stats-box-value {
                font-size: clamp(28px, 6vw, 42px);
                font-weight: 800;
                margin: 6px 0;
                color: #059669;
            }
            .stats-box-label {
                font-size: clamp(12px, 2.5vw, 15px);
                color: #475569;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .feature-card {
                background: rgba(255, 255, 255, 0.95);
                border-left: 5px solid #10b981;
                border-radius: 10px;
                padding: clamp(16px, 4vw, 22px);
                box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
                margin-bottom: clamp(14px, 3vw, 18px);
            }
            .feature-card h3 {
                color: #047857;
                margin: 0 0 8px 0;
                font-size: clamp(16px, 3vw, 21px);
                font-weight: 750;
            }
            .feature-card p {
                margin: 0;
                color: #475569;
                font-size: clamp(13px, 2.5vw, 15px);
                line-height: 1.55;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-header">
            <h1>Analytics Warehouse</h1>
            <p>Data Warehousing, Data Modeling, Visualization & AI Analytics Project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        ## Welcome, {st.session_state.username}!

        Your data warehousing, modeling, visualization, and analytics workspace is ready.
        """
    )

    if all_players is not None and len(all_players) > 0:
        num_players = len(all_players)
        num_teams = all_players["Team"].nunique() if "Team" in all_players.columns else 0
        num_formats = all_players["Format"].nunique() if "Format" in all_players.columns else 0
    else:
        num_players = 1000
        num_teams = 20
        num_formats = 3

    st.markdown(
        f"""
        <div class="stats-grid">
            <div class="stats-box">
                <div class="stats-box-value">{num_players}</div>
                <div class="stats-box-label">Warehouse Records</div>
            </div>
            <div class="stats-box">
                <div class="stats-box-value">{num_teams}</div>
                <div class="stats-box-label">Dimension Members</div>
            </div>
            <div class="stats-box">
                <div class="stats-box-value">{num_formats}</div>
                <div class="stats-box-label">Business Domains</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## Platform Features")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Data Warehouse</h3>
                <p>Profile raw CSV sources, validate staging data, and organize clean records for analytics marts.</p>
            </div>
            <div class="feature-card">
                <h3>Data Modeling</h3>
                <p>Explore a star schema with fact performance records and player, team, format, time, and role dimensions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>Visualization Analytics</h3>
                <p>Use interactive BI charts for trend analysis, dimensional comparison, and data quality monitoring.</p>
            </div>
            <div class="feature-card">
                <h3>AI Modeling</h3>
                <p>Prepare model-ready features for scouting, predictions, similarity analysis, and expert chat workflows.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## About This Project")
    st.info(
        """
        This project demonstrates an end-to-end analytics workflow: source ingestion, data profiling,
        data quality checks, dimensional warehouse modeling, analytics marts, visualization dashboards,
        and AI-ready feature preparation using cricket performance data as the business domain.
        """
    )

    st.markdown("---")
    st.markdown("## Quick Navigation")

    nav_col1, nav_col2, nav_col3 = st.columns(3, gap="small")

    with nav_col1:
        if st.button("Warehouse Model", width="stretch", help="Open data warehouse and modeling"):
            st.session_state.page = "Warehouse Modeling"
            st.rerun()

    with nav_col2:
        if st.button("Visual Analytics", width="stretch", help="Go to visual analytics"):
            st.session_state.page = "🏏 Cricket Analysis"
            st.session_state.analysis_menu = "Format Wise Analysis"
            st.rerun()

    with nav_col3:
        if st.button("AI Modeling", width="stretch", help="Open AI scouting"):
            st.session_state.page = "🏏 Cricket Analysis"
            st.session_state.analysis_menu = "Smart Scout (AI)"
            st.rerun()

    st.markdown(
        """
        <div style="text-align: center; color: #64748b; padding: clamp(15px, 3vw, 20px); margin-top: 30px;">
            <p style="margin: 5px 0; font-size: clamp(12px, 2.5vw, 14px);">
                <strong>Analytics Warehouse v2.0</strong>
            </p>
            <p style="margin: 5px 0; font-size: clamp(11px, 2vw, 13px);">
                Data Warehousing, Modeling and BI Visualization Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    show_home_page()
