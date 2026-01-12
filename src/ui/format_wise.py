import streamlit as st
import plotly.express as px
import pandas as pd
from ..config import FORMATS
from ..utils import sort_players

def render_format_analysis(batsmen, all_rounders, bowlers_data, wicket_keepers):
    for fmt in FORMATS:
        st.markdown(f"## 🏏 {fmt} Format Analysis")
        st.markdown("---")

        # Filter pre-classified data for this format
        fmt_batsmen = batsmen[batsmen['Format'] == fmt]
        fmt_all_rounders = all_rounders[all_rounders['Format'] == fmt]
        fmt_bowlers = bowlers_data[bowlers_data['Format'] == fmt]
        fmt_wicket_keepers = wicket_keepers[wicket_keepers['Format'] == fmt]

        if fmt_batsmen.empty and fmt_all_rounders.empty and fmt_bowlers.empty:
            st.info(f"No data available for {fmt} format.")
            continue

        # For charts, we use these format-specific subsets
        filtered_batsmen = fmt_batsmen[fmt_batsmen['matches'] > 10]
        filtered_all_rounders = fmt_all_rounders[fmt_all_rounders['matches'] > 10]
        filtered_bowlers = fmt_bowlers[fmt_bowlers['matches'] > 10]

        # --- Top 3 Visualizations ---
        col1, col2, col3 = st.columns(3)
        with col1:
            if not filtered_batsmen.empty:
                top_runs = sort_players(filtered_batsmen, top_n=10, by='runs', ascending=False)
                fig = px.bar(top_runs, x='player', y='runs', color='Team', title=f"🏆 Top 10 Run Scorers - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'top_runs_{fmt}')
        
        with col2:
            if not filtered_batsmen.empty:
                # Scatter usually doesn't need "higher to lower" order on an axis in the same way, but let's keep it sorted by avg
                top_scatter = sort_players(filtered_batsmen, top_n=100, by='average', ascending=False)
                fig = px.scatter(top_scatter, x='average', y='strike_rate', color='Team', 
                                 size='matches', hover_name='player', title=f"📈 Avg vs SR - {fmt}")
                st.plotly_chart(fig, use_container_width=True, key=f'avg_sr_{fmt}')

        with col3:
            if not fmt_wicket_keepers.empty:
                top_wk = sort_players(fmt_wicket_keepers, top_n=10, by='average', ascending=False)
                fig = px.scatter(top_wk, x='average', y='strike_rate', color='Team',
                                 size='matches', hover_name='player', title=f"📊 WKs: Avg vs SR - {fmt}")
                st.plotly_chart(fig, use_container_width=True, key=f'wk_scatter_{fmt}')

        # --- Detailed Batting Analysis ---
        st.markdown("---")
        if not filtered_batsmen.empty:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                top_avg = sort_players(filtered_batsmen, top_n=10, by='average', ascending=False)
                fig = px.bar(top_avg, x='player', y='average', color='Team', title=f"Top 10 Batters by Average - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'bat_avg_{fmt}')
            
            with col_b2:
                top_sr = sort_players(filtered_batsmen, top_n=10, by='strike_rate', ascending=False)
                fig = px.bar(top_sr, x='player', y='strike_rate', color='Team', title=f"Top 10 Batters by Strike Rate - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'bat_sr_{fmt}')

        # --- All-Rounders Analysis ---
        st.markdown("---")
        if not filtered_all_rounders.empty:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                top_wickets_ar = sort_players(filtered_all_rounders, top_n=10, by='wickets', ascending=False)
                fig = px.bar(top_wickets_ar, x='player', y='wickets', color='Team', title=f"Top 10 All-Rounders by Wickets - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'wickets_ar_{fmt}')
            
            with col_a2:
                top_bowling_avg_ar = sort_players(filtered_all_rounders, top_n=10, by='bowling_average', ascending=True) # Lower is better
                fig = px.bar(top_bowling_avg_ar, x='player', y='bowling_average', color='Team', title=f"Top 10 All-Rounders by Bowling Avg (Lower Better) - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'bowling_avg_ar_{fmt}')

        # --- Bowlers Analysis ---
        st.markdown("---")
        if not filtered_bowlers.empty:
            st.subheader(f"⚡ Top 10 Bowlers - {fmt}")
            col_bw1, col_bw2 = st.columns(2)
            with col_bw1:
                top_wickets_bw = sort_players(filtered_bowlers, top_n=10, by='wickets', ascending=False)
                fig = px.bar(top_wickets_bw, x='player', y='wickets', color='Team', title=f"Top 10 Bowlers by Wickets - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'wickets_bw_{fmt}')
            
            with col_bw2:
                top_bowling_avg_bw = sort_players(filtered_bowlers, top_n=10, by='bowling_average', ascending=True) # Lower is better
                fig = px.bar(top_bowling_avg_bw, x='player', y='bowling_average', color='Team', title=f"Top 10 Bowlers by Bowling Avg (Lower Better) - {fmt}")
                fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f'bowling_avg_bw_{fmt}')
