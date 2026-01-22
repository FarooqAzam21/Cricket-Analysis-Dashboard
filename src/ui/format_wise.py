import streamlit as st
import plotly.express as px
import pandas as pd
from ..config import FORMATS
from ..utils import sort_players

def render_format_analysis(batsmen, all_rounders, bowlers_data, wicket_keepers):
    # Create tabs for each format
    tabs = st.tabs([f"🏏 {fmt}" for fmt in FORMATS])
    
    for i, fmt in enumerate(FORMATS):
        with tabs[i]:
            st.markdown(f"## {fmt} Format Analysis")
            st.markdown("---")

            # Filter pre-classified data for this format
            fmt_batsmen = batsmen[batsmen['Format'] == fmt]
            fmt_all_rounders = all_rounders[all_rounders['Format'] == fmt]
            fmt_bowlers = bowlers_data[bowlers_data['Format'] == fmt]
            fmt_wicket_keepers = wicket_keepers[wicket_keepers['Format'] == fmt]

            if fmt_batsmen.empty and fmt_all_rounders.empty and fmt_bowlers.empty:
                st.info(f"No data available for {fmt} format.")
                continue

            # --- Apply Stricter Filters for Charts ---
            # 1. Batting Filter: 50+ Matches AND 1000+ Runs
            filtered_batsmen = fmt_batsmen[(fmt_batsmen['matches'] >= 50) & (fmt_batsmen['runs'] >= 1000)]
            
            # 2. Bowling Filter: 50+ Matches AND 50+ Wickets
            filtered_bowlers = fmt_bowlers[(fmt_bowlers['matches'] >= 50) & (fmt_bowlers['wickets'] >= 50)]
            
            # 3. All-Rounder Filter: 50+ Matches AND (1000+ Runs OR 50+ Wickets)
            filtered_all_rounders = fmt_all_rounders[(fmt_all_rounders['matches'] >= 50) & 
                                                    ((fmt_all_rounders['runs'] >= 1000) | (fmt_all_rounders['wickets'] >= 50))]
            
            # 4. Strike Rate Filter for Scatter Plots: SR > 100
            sr_filtered_batsmen = filtered_batsmen[filtered_batsmen['strike_rate'] > 100]

            # --- Top 3 Visualizations ---
            col1, col2, col3 = st.columns(3)
            with col1:
                if not filtered_batsmen.empty:
                    top_runs = sort_players(filtered_batsmen, top_n=10, by='runs', ascending=False)
                    fig = px.bar(top_runs, x='player', y='runs', color='Team', title=f"🏆 Top 10 Run Scorers (50+ Matches, 1000+ Runs)")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'top_runs_{fmt}')
                else:
                    st.warning("No players found with 50+ Matches & 1000+ Runs")
            
            with col2:
                # Use SR > 100 filter for the scatter plot
                plot_data = sr_filtered_batsmen if not sr_filtered_batsmen.empty else filtered_batsmen
                if not plot_data.empty:
                    top_scatter = sort_players(plot_data, top_n=100, by='average', ascending=False)
                    title = "📈 Avg vs SR" + (" (SR > 100)" if not sr_filtered_batsmen.empty else "")
                    fig = px.scatter(top_scatter, x='average', y='strike_rate', color='Team', 
                                     size='matches', hover_name='player', title=title)
                    st.plotly_chart(fig, use_container_width=True, key=f'avg_sr_{fmt}')

            with col3:
                if not fmt_wicket_keepers.empty:
                    # For WKs we relax the match filter slightly (25+) to show more talent, but still keep it elite
                    wk_elite = fmt_wicket_keepers[fmt_wicket_keepers['matches'] >= 25]
                    if not wk_elite.empty:
                        top_wk = sort_players(wk_elite, top_n=10, by='average', ascending=False)
                        fig = px.scatter(top_wk, x='average', y='strike_rate', color='Team',
                                         size='matches', hover_name='player', title=f"📊 WKs: Avg vs SR (25+ Matches)")
                        st.plotly_chart(fig, use_container_width=True, key=f'wk_scatter_{fmt}')

            # --- Detailed Batting Analysis ---
            st.markdown("---")
            if not filtered_batsmen.empty:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    top_avg = sort_players(filtered_batsmen, top_n=10, by='average', ascending=False)
                    fig = px.bar(top_avg, x='player', y='average', color='Team', title=f"Top 10 Batters by Average (50+ Matches)")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'bat_avg_{fmt}')
                
                with col_b2:
                    # Strike rate specific top 10
                    top_sr = sort_players(sr_filtered_batsmen if not sr_filtered_batsmen.empty else filtered_batsmen, 
                                          top_n=10, by='strike_rate', ascending=False)
                    fig = px.bar(top_sr, x='player', y='strike_rate', color='Team', title=f"Top 10 Batters by Strike Rate (SR > 100)")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'bat_sr_{fmt}')

            # --- All-Rounders Analysis ---
            st.markdown("---")
            if not filtered_all_rounders.empty:
                st.subheader(f"🔄 Top All-Rounders (50+ Matches, 50+ Wickets)")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    top_wickets_ar = sort_players(filtered_all_rounders, top_n=10, by='wickets', ascending=False)
                    fig = px.bar(top_wickets_ar, x='player', y='wickets', color='Team', title=f"Top 10 All-Rounders by Wickets")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'wickets_ar_{fmt}')
                
                with col_a2:
                    top_bowling_avg_ar = sort_players(filtered_all_rounders, top_n=10, by='bowling_average', ascending=True)
                    fig = px.bar(top_bowling_avg_ar, x='player', y='bowling_average', color='Team', title=f"Top 10 All-Rounders by Bowling Avg")
                    fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'bowling_avg_ar_{fmt}')

            # --- Bowlers Analysis ---
            st.markdown("---")
            if not filtered_bowlers.empty:
                st.subheader(f"⚡ Top 10 Bowlers (50+ Matches, 50+ Wickets)")
                col_bw1, col_bw2 = st.columns(2)
                with col_bw1:
                    top_wickets_bw = sort_players(filtered_bowlers, top_n=10, by='wickets', ascending=False)
                    fig = px.bar(top_wickets_bw, x='player', y='wickets', color='Team', title=f"Top 10 Bowlers by Wickets")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'wickets_bw_{fmt}')
                
                with col_bw2:
                    top_bowling_avg_bw = sort_players(filtered_bowlers, top_n=10, by='bowling_average', ascending=True)
                    fig = px.bar(top_bowling_avg_bw, x='player', y='bowling_average', color='Team', title=f"Top 10 Bowlers by Bowling Avg")
                    fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f'bowling_avg_bw_{fmt}')
