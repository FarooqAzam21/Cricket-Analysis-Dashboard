import streamlit as st
import pandas as pd
from ..ai_features import find_similar_players
from ..database import save_scout_feedback

def render_smart_scout(all_players):
    st.markdown("---")
    st.header("🔎 AI Smart Scout (Player Similarity)")
    st.info("Select a player to find others with a similar statistical profile. Rate the results to help improve recommendations!")
    
    # Cache player list in session state to avoid recomputing
    if 'cached_player_list' not in st.session_state:
        st.session_state.cached_player_list = sorted(all_players['player'].unique().tolist())
    
    player_list = st.session_state.cached_player_list
    selected_player = st.selectbox("Select Player to Find Look-alikes", player_list)
    
    if selected_player:
        # Filter by format to make comparison more accurate
        player_formats = all_players[all_players['player'] == selected_player]['Format'].unique()
        selected_fmt = st.selectbox("Select Format", player_formats)
        
        fmt_data = all_players[all_players['Format'] == selected_fmt].reset_index(drop=True)
        
        if st.button("Find Similar Players"):
            with st.spinner("Finding similar players..."):
                similar_df, scores = find_similar_players(fmt_data, selected_player)
            
            # Store results in session state for feedback
            st.session_state.scout_results = {
                'source_player': selected_player,
                'format': selected_fmt,
                'similar_players': similar_df,
                'scores': scores
            }
            
        # Display results if they exist in session state
        if 'scout_results' in st.session_state and st.session_state.scout_results['source_player'] == selected_player:
            similar_df = st.session_state.scout_results['similar_players']
            scores = st.session_state.scout_results['scores']
            
            if not similar_df.empty:
                st.subheader(f"Top 5 Players Similar to {selected_player} in {selected_fmt}")
                
                # Show results
                for i, (_, row) in enumerate(similar_df.iterrows()):
                    match_score = round((1 - scores[i]) * 100, 2)
                    
                    # Main container for each result
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        
                        with col1:
                            if row.get('image_url') and str(row['image_url']) != 'nan' and row['image_url'] != "":
                                st.image(row['image_url'], width=100)
                            else:
                                st.image("https://via.placeholder.com/100?text=No+Img", width=100)
                        
                        with col2:
                            st.markdown(f"#### {row['player']} ({row['Team']})")
                            st.markdown(f"**Role:** `{row.get('role', 'N/A')}` | **Match:** {match_score}%")
                            
                            cols = st.columns(4)
                            role_l = str(row.get('role', '')).lower()
                            
                            if 'bowler' in role_l or 'spinner' in role_l or 'fast' in role_l:
                                # Bowler Metrics
                                cols[0].metric("Matches", row.get('matches', 0))
                                cols[1].metric("Runs", row.get('runs', 0))
                                cols[2].metric("Wickets", row.get('wickets', 0))
                                cols[3].metric("Bowl Avg", row.get('bowling_average', 0))
                            else:
                                # Batsman Metrics
                                cols[0].metric("Runs", row.get('runs', 0))
                                cols[1].metric("Average", row.get('average', 0))
                                cols[2].metric("SR", row.get('strike_rate', 0))
                                cols[3].metric("Wickets", row.get('wickets', 0))
                        
                        with col3:
                            st.markdown("**Rate this match:**")
                            feedback_key = f"feedback_{selected_player}_{row['player']}_{i}"
                            
                            col_up, col_down = st.columns(2)
                            with col_up:
                                if st.button("👍", key=f"up_{feedback_key}", help="Good match"):
                                    save_scout_feedback(
                                        st.session_state.username,
                                        selected_player,
                                        row['player'],
                                        selected_fmt,
                                        "good"
                                    )
                                    st.success("✓")
                            
                            with col_down:
                                if st.button("👎", key=f"down_{feedback_key}", help="Bad match"):
                                    save_scout_feedback(
                                        st.session_state.username,
                                        selected_player,
                                        row['player'],
                                        selected_fmt,
                                        "bad"
                                    )
                                    st.error("✓")
                        
                        st.markdown("---")
            else:
                st.error("Could not find similar players.")
