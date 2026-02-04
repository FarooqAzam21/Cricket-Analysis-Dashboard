import streamlit as st
import plotly.express as px
import pandas as pd

def render_player_analysis(all_players):
    st.markdown("---")
    st.header("🔍 Player Search & Analysis")
    
    player_list = sorted(all_players['player'].dropna().unique().tolist())
    
    # Check for preselected player from Global Search
    default_idx = 0
    if 'preselected_player' in st.session_state and st.session_state.preselected_player in player_list:
        default_idx = player_list.index(st.session_state.preselected_player)
        # Clear it immediately to allow manual selection later
        del st.session_state.preselected_player
        
    selected_player = st.selectbox("Search Player", player_list, index=default_idx, key='player_search_box')

    if selected_player:
        player_data = all_players[all_players['player'] == selected_player]
        if not player_data.empty:
            player_row = player_data.iloc[0]
            
            # ELITE PLAYER HEADER
            st.markdown(f"""
            <div class="elite-card">
                <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
                    <img src="{player_row.get('image_url', 'https://via.placeholder.com/150?text=No+Img')}" 
                         style="width: 180px; height: 180px; border-radius: 90px; object-fit: cover; border: 5px solid var(--primary); box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
                    <div style="flex-grow: 1;">
                        <h1 style="margin: 0; border: none; padding: 0;">{player_row.get('player','Unknown')}</h1>
                        <p style="font-size: 1.2rem; margin: 10px 0; color: var(--primary-dark) !important; font-weight: 600;">
                            {player_row.get('Team','-')} | {player_row.get('role','-')}
                        </p>
                        <div style="display: flex; gap: 15px; margin-top: 15px;">
                            <span style="background: rgba(16, 185, 129, 0.1); padding: 5px 15px; border-radius: 20px; font-weight: 600;">🏏 {player_row.get('batting_position','-')} Pos</span>
                            <span style="background: rgba(59, 130, 246, 0.1); padding: 5px 15px; border-radius: 20px; font-weight: 600;">📊 Stats Ready</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")

            if 'Format' in player_data.columns:
                formats = player_data['Format'].unique()
                
                # Tabs for different formats
                tabs = st.tabs([f"🏏 {fmt}" for fmt in formats])
                
                for idx, fmt in enumerate(formats):
                    with tabs[idx]:
                        fmt_data = player_data[player_data['Format'] == fmt]
                        
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Matches", int(fmt_data['matches'].sum()))
                            st.metric("Runs", int(fmt_data['runs'].sum()))
                        with cols[1]:
                            st.metric("Innings", int(fmt_data['Innings'].sum() if 'Innings' in fmt_data.columns else 0))
                            st.metric("Average", round(fmt_data['average'].mean(), 2))
                        with cols[2]:
                            st.metric("Strike Rate", round(fmt_data['strike_rate'].mean(), 2))
                            st.metric("Wickets", int(fmt_data['wickets'].sum()))
                            
                        if fmt_data['wickets'].sum() > 0:
                            st.markdown("---")
                            bowl_cols = st.columns(2)
                            bowl_cols[0].metric("Bowling Avg", round(fmt_data['bowling_average'].mean(), 2))
                            bowl_cols[1].metric("Economy", round(fmt_data['economy'].mean(), 2))

                st.markdown("---")
                # Format Comparison Chart
                st.subheader("📈 Performance Analysis")
                chart_tabs = st.tabs(["💰 Runs by Format", "🎯 Avg vs SR"])
                
                with chart_tabs[0]:
                    format_stats = player_data.groupby('Format')[['runs', 'matches', 'average', 'strike_rate']].mean().reset_index()
                    fig = px.bar(format_stats, x='Format', y='runs', color='Format', 
                                title=f"Runs by Format for {selected_player}",
                                template="plotly_white",
                                color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_tabs[1]:
                    fig2 = px.scatter(format_stats, x='average', y='strike_rate', size='runs', color='Format',
                                     title="Consistency vs Impact (Avg vs SR)",
                                     template="plotly_white",
                                     text='Format')
                    fig2.update_traces(textposition='top center')
                    st.plotly_chart(fig2, use_container_width=True)
