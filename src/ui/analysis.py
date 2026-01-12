import streamlit as st
import plotly.express as px
import pandas as pd

def render_player_analysis(all_players):
    st.markdown("---")
    st.header("🔍 Player Search & Analysis")
    
    player_list = sorted(all_players['player'].dropna().unique().tolist())
    selected_player = st.selectbox("Search Player", player_list, key='player_search_box')

    if selected_player:
        player_data = all_players[all_players['player'] == selected_player]
        if not player_data.empty:
            player_row = player_data.iloc[0]
            col_img, col_info = st.columns([1, 2])

            with col_img:
                img_url = player_row.get('image_url', "https://via.placeholder.com/150?text=No+Image")
                st.image(img_url, width=180)

            with col_info:
                st.markdown(f"### {player_row.get('player','Unknown')}")
                st.markdown(f"**Team:** {player_row.get('Team','-')}")
                st.markdown(f"**Role:** {player_row.get('role','-')}")
                st.markdown(f"**Batting Position:** {player_row.get('batting_position','-')}")

            if 'Format' in player_data.columns:
                formats = player_data['Format'].unique()
                for fmt in formats:
                    fmt_data = player_data[player_data['Format'] == fmt]
                    st.markdown(f"#### 🏏 {fmt} Format")
                    cols = st.columns(6)
                    cols[0].metric("Matches", int(fmt_data['matches'].sum()))
                    cols[1].metric("Runs", int(fmt_data['runs'].sum()))
                    cols[2].metric("Average", round(fmt_data['average'].mean(), 2))
                    cols[3].metric("SR", round(fmt_data['strike_rate'].mean(), 2))
                    cols[4].metric("Wickets", int(fmt_data['wickets'].sum()))
                    cols[5].metric("Bowling Avg", round(fmt_data['bowling_average'].mean(), 2))
                    st.markdown("---")

                # Format Comparison Chart
                st.subheader("📈 Runs Comparison Across Formats")
                format_stats = player_data.groupby('Format')[['runs', 'matches', 'average', 'strike_rate']].mean().reset_index()
                fig = px.bar(format_stats, x='Format', y='runs', color='Format', title=f"Runs by Format for {selected_player}")
                st.plotly_chart(fig, use_container_width=True)
