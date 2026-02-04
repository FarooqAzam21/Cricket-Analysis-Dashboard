import streamlit as st
import plotly.graph_objects as go

def render_comparison(all_players):
    st.markdown("---")
    st.header("⚔️ Player Battle Arena")
    st.info("Compare up to 4 players simultaneously to find the ultimate statistical match-winner.")

    try:
        # Multi-player selection with multiselect or separate selects
        player_list = sorted(all_players['player'].unique().tolist())
        
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            selected_players = st.multiselect(
                "Select Players for Battle (Max 4)", 
                player_list, 
                max_selections=4,
                default=player_list[:2] if len(player_list) >= 2 else player_list
            )
        
        with col_ctrl2:
            formats = all_players['Format'].unique()
            selected_format = st.selectbox("Select Format", formats, key="fmt_cmp_elite")

        if not selected_players:
            st.warning("Please select at least one player to compare.")
            return

        # Metrics for radar and display
        bat_metrics = ['runs', 'average', 'strike_rate', '100s', '50s']
        bowl_metrics = ['wickets', 'bowling_average', 'economy']
        
        # Display Player Tiles in a responsive grid
        cols = st.columns(len(selected_players))
        
        radar_data = []

        for idx, player in enumerate(selected_players):
            with cols[idx]:
                p_fmt_data = all_players[(all_players['player'] == player) & (all_players['Format'] == selected_format)]
                
                if not p_fmt_data.empty:
                    p_row = p_fmt_data.iloc[0]
                    
                    # Battle Tile UI
                    st.markdown(f"""
                    <div class="battle-tile">
                        <img src="{p_row.get('image_url', 'https://via.placeholder.com/100?text=No+Img')}" class="battle-avatar">
                        <h4 style="margin: 5px 0; color: var(--primary-dark) !important;">{player}</h4>
                        <p style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 10px;">{p_row.get('Team', '-')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key Battle Stats
                    st.metric("Runs", int(p_row['runs']))
                    st.metric("Avg", round(p_row['average'], 1))
                    if p_row['wickets'] > 0:
                        st.metric("Wickets", int(p_row['wickets']))
                    
                    # Collect data for radar
                    radar_vals = [p_row[m] for m in bat_metrics]
                    radar_data.append((player, radar_vals))
                else:
                    st.error(f"No {selected_format} data for {player}")

        st.markdown("---")
        
        # Advanced Radar Comparison
        if len(radar_data) > 0:
            st.subheader("📊 Statistical Overlap (Radar)")
            fig = go.Figure()
            
            colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
            
            for i, (name, vals) in enumerate(radar_data):
                # Normalize values for radar comparison if they are in vastly different scales
                # For now, just plot them directly
                fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=[m.replace('_', ' ').capitalize() for m in bat_metrics],
                    fill='toself',
                    name=name,
                    line_color=colors[i % len(colors)]
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max([max(v) for n, v in radar_data]) * 1.1])
                ),
                showlegend=True,
                height=600,
                template="plotly_white",
                title=f"{selected_format} Radar Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Comparison Table (Elite Glass style)
            st.subheader("📋 In-Depth Metric Breakdown")
            comp_df = all_players[(all_players['player'].isin(selected_players)) & (all_players['Format'] == selected_format)]
            display_cols = ['player', 'Team', 'matches', 'Innings', 'runs', 'average', 'strike_rate', 'wickets', 'bowling_average', 'economy']
            st.dataframe(
                comp_df[display_cols].set_index('player'),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Battle Arena error: {e}")
        import traceback
        st.code(traceback.format_exc())
