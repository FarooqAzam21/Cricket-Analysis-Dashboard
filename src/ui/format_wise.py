import streamlit as st
import plotly.express as px
import pandas as pd
from ..config import FORMATS
from ..utils import sort_players

def render_format_analysis(batsmen, all_rounders, bowlers_data, wicket_keepers):
    # Debug check and fallback
    if batsmen is None or batsmen.empty:
        st.error(f"❌ No batsmen data. Type: {type(batsmen)}, Shape: {batsmen.shape if hasattr(batsmen, 'shape') else 'N/A'}")
        st.info("⚠️ Falling back to showing all players data")
        # If batsmen is empty, use all_rounders or bowlers_data as fallback
        if bowlers_data is not None and not bowlers_data.empty:
            batsmen = bowlers_data
        elif all_rounders is not None and not all_rounders.empty:
            batsmen = all_rounders
        else:
            st.error("❌ No player data available at all. Please check CSV files.")
            st.stop()
    
    # --- Interactive Filter Section ---
    st.markdown("### 🛠️ Global Chart Filters")
    with st.expander("Adjust Analysis Thresholds (Applies to all charts)", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_matches = st.number_input("Min Matches", min_value=0, value=10, step=5)
            min_sr = st.number_input("Min Strike Rate (Batting)", min_value=0, value=80, step=10)
        with col_f2:
            min_runs = st.number_input("Min Runs", min_value=0, value=500, step=100)
            include_teams = st.multiselect("Filter Teams", options=sorted(batsmen['Team'].unique()), default=None)
        with col_f3:
            min_wickets = st.number_input("Min Wickets", min_value=0, value=10, step=5)
            top_n_charts = st.slider("Show Top N Players", 5, 20, 10)

    # Create tabs for each format
    tabs = st.tabs([f"🏏 {fmt}" for fmt in FORMATS])
    
    for i, fmt in enumerate(FORMATS):
        with tabs[i]:
            st.markdown(f"## {fmt} Format Analysis")
            st.markdown("---")

            # Filter data for this format
            fmt_batsmen = batsmen[batsmen['Format'] == fmt]
            fmt_all_rounders = all_rounders[all_rounders['Format'] == fmt]
            fmt_bowlers = bowlers_data[bowlers_data['Format'] == fmt]
            fmt_wicket_keepers = wicket_keepers[wicket_keepers['Format'] == fmt]

            # Apply Team Filter if selected
            if include_teams and len(include_teams) > 0:
                fmt_batsmen = fmt_batsmen[fmt_batsmen['Team'].isin(include_teams)]
                fmt_all_rounders = fmt_all_rounders[fmt_all_rounders['Team'].isin(include_teams)]
                fmt_bowlers = fmt_bowlers[fmt_bowlers['Team'].isin(include_teams)]
                fmt_wicket_keepers = fmt_wicket_keepers[fmt_wicket_keepers['Team'].isin(include_teams)]

            # Check if any data exists for this format
            if fmt_batsmen.empty and fmt_all_rounders.empty and fmt_bowlers.empty:
                st.warning(f"❌ No data available for {fmt} format. Please check the data.")
                continue
            
            # Check if all data is filtered out
            filtered_batsmen = fmt_batsmen[(fmt_batsmen['matches'] >= min_matches) & (fmt_batsmen['runs'] >= min_runs)]
            filtered_bowlers = fmt_bowlers[(fmt_bowlers['matches'] >= min_matches) & (fmt_bowlers['wickets'] >= min_wickets)]
            filtered_all_rounders = fmt_all_rounders[(fmt_all_rounders['matches'] >= min_matches) & 
                                                    ((fmt_all_rounders['runs'] >= min_runs) | (fmt_all_rounders['wickets'] >= min_wickets))]
            
            if filtered_batsmen.empty and filtered_bowlers.empty and filtered_all_rounders.empty:
                with st.expander(f"ℹ️ No data available for {fmt} format with selected filters - Click to adjust", expanded=True):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.warning(f"**Current Filters:**")
                        st.write(f"🔹 Min Matches: **{min_matches}**")
                        st.write(f"🔹 Min Runs: **{min_runs}**")
                        st.write(f"🔹 Min Wickets: **{min_wickets}**")
                        st.write(f"🔹 Min Strike Rate: **{min_sr}**")
                    with col_info2:
                        st.info(f"**{fmt} Format Data Available:**")
                        st.write(f"📊 Batsmen: **{len(fmt_batsmen)}**")
                        st.write(f"⚡ Bowlers: **{len(fmt_bowlers)}**")
                        st.write(f"🔄 All-Rounders: **{len(fmt_all_rounders)}**")
                        st.write(f"👥 Teams: **{len(include_teams) if include_teams else 'All'}**")
                    st.error("🔧 Try reducing filter values above to see more data!")
                continue

            # --- Apply Interactive Filters ---
            filtered_batsmen = fmt_batsmen[(fmt_batsmen['matches'] >= min_matches) & (fmt_batsmen['runs'] >= min_runs)]
            filtered_bowlers = fmt_bowlers[(fmt_bowlers['matches'] >= min_matches) & (fmt_bowlers['wickets'] >= min_wickets)]
            filtered_all_rounders = fmt_all_rounders[(fmt_all_rounders['matches'] >= min_matches) & 
                                                    ((fmt_all_rounders['runs'] >= min_runs) | (fmt_all_rounders['wickets'] >= min_wickets))]
            
            # Special filter for SR scatter
            sr_filtered_batsmen = filtered_batsmen[filtered_batsmen['strike_rate'] >= min_sr]

            # --- Top 3 Visualizations ---
            col1, col2, col3 = st.columns(3)
            with col1:
                if not filtered_batsmen.empty:
                    top_runs = sort_players(filtered_batsmen, top_n=top_n_charts, by='runs', ascending=False)
                    fig = px.bar(top_runs, x='player', y='runs', color='Team', 
                                 title=f"🏆 Top {top_n_charts} Run Scorers ({min_matches}+ M, {min_runs}+ R)")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'top_runs_{fmt}')
                else:
                    st.warning(f"No players found with {min_matches}+ Matches & {min_runs}+ Runs")
            
            with col2:
                plot_data = sr_filtered_batsmen if not sr_filtered_batsmen.empty else filtered_batsmen
                if not plot_data.empty:
                    top_scatter = sort_players(plot_data, top_n=100, by='average', ascending=False)
                    title = f"📈 Avg vs SR (SR >= {min_sr})" if not sr_filtered_batsmen.empty else "📈 Avg vs SR"
                    fig = px.scatter(top_scatter, x='average', y='strike_rate', color='Team', 
                                     size='matches', hover_name='player', title=title)
                    st.plotly_chart(fig, width="stretch", key=f'avg_sr_{fmt}')

            with col3:
                if not fmt_wicket_keepers.empty:
                    # Relaxed match filter for WKs but still follows global min if possible
                    wk_min = max(20, min_matches // 2)
                    wk_elite = fmt_wicket_keepers[fmt_wicket_keepers['matches'] >= wk_min]
                    if not wk_elite.empty:
                        top_wk = sort_players(wk_elite, top_n=top_n_charts, by='average', ascending=False)
                        fig = px.scatter(top_wk, x='average', y='strike_rate', color='Team',
                                         size='matches', hover_name='player', title=f"📊 WKs: Avg vs SR ({wk_min}+ M)")
                        st.plotly_chart(fig, width="stretch", key=f'wk_scatter_{fmt}')

            # --- Detailed Batting Analysis ---
            st.markdown("---")
            if not filtered_batsmen.empty:
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    top_avg = sort_players(filtered_batsmen, top_n=top_n_charts, by='average', ascending=False)
                    fig = px.bar(top_avg, x='player', y='average', color='Team', 
                                 title=f"Top {top_n_charts} Batters by Average ({min_matches}+ M)")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'bat_avg_{fmt}')
                
                with col_b2:
                    plot_data_sr = sr_filtered_batsmen if not sr_filtered_batsmen.empty else filtered_batsmen
                    top_sr = sort_players(plot_data_sr, top_n=top_n_charts, by='strike_rate', ascending=False)
                    fig = px.bar(top_sr, x='player', y='strike_rate', color='Team', 
                                 title=f"Top {top_n_charts} by Strike Rate (SR >= {min_sr})")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'bat_sr_{fmt}')

            # --- All-Rounders Analysis ---
            st.markdown("---")
            if not filtered_all_rounders.empty:
                st.subheader(f"🔄 Top All-Rounders ({min_matches}+ M, {min_wickets}+ W)")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    top_wickets_ar = sort_players(filtered_all_rounders, top_n=top_n_charts, by='wickets', ascending=False)
                    fig = px.bar(top_wickets_ar, x='player', y='wickets', color='Team', title=f"Top {top_n_charts} All-Rounders by Wickets")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'wickets_ar_{fmt}')
                
                with col_a2:
                    top_bowling_avg_ar = sort_players(filtered_all_rounders, top_n=top_n_charts, by='bowling_average', ascending=True)
                    fig = px.bar(top_bowling_avg_ar, x='player', y='bowling_average', color='Team', title=f"Top {top_n_charts} by Bowling Avg")
                    fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'bowling_avg_ar_{fmt}')

            # --- Bowlers Analysis ---
            st.markdown("---")
            if not filtered_bowlers.empty:
                st.subheader(f"⚡ Top Bowlers ({min_matches}+ M, {min_wickets}+ W)")
                col_bw1, col_bw2 = st.columns(2)
                with col_bw1:
                    top_wickets_bw = sort_players(filtered_bowlers, top_n=top_n_charts, by='wickets', ascending=False)
                    fig = px.bar(top_wickets_bw, x='player', y='wickets', color='Team', title=f"Top {top_n_charts} Bowlers by Wickets")
                    fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'wickets_bw_{fmt}')
                
                with col_bw2:
                    top_bowling_avg_bw = sort_players(filtered_bowlers, top_n=top_n_charts, by='bowling_average', ascending=True)
                    fig = px.bar(top_bowling_avg_bw, x='player', y='bowling_average', color='Team', title=f"Top {top_n_charts} by Bowling Avg")
                    fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_tickangle=45)
                    st.plotly_chart(fig, width="stretch", key=f'bowling_avg_bw_{fmt}')
