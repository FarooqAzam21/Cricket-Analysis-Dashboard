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
                    img_url = row.get('image_url', "https://via.placeholder.com/100?text=No+Img")
                    
                    # ELITE RESULT CARD
                    st.markdown(f"""
                    <div class="elite-card">
                        <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 20px;">
                            <img src="{img_url}" style="width: 80px; height: 80px; border-radius: 40px; object-fit: cover; border: 3px solid var(--primary); box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                            <div style="flex-grow: 1;">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div>
                                        <h4 style="margin: 0; color: var(--primary-dark) !important;">{row['player']}</h4>
                                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.7; font-weight: 600;">{row['Team']} | {row.get('role', 'N/A')}</p>
                                    </div>
                                    <div style="background: var(--primary); color: #e2e8f0; padding: 4px 12px; border-radius: 20px; font-weight: 800; font-size: 0.8rem;">
                                        {match_score}% MATCH
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; background: rgba(0,0,0,0.02); padding: 15px; border-radius: 12px;">
                            <div style="text-align: center;">
                                <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; font-weight: 800;">Matches</div>
                                <div style="font-size: 1.1rem; font-weight: 800; color: var(--primary-dark);">{int(row.get('matches', 0))}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; font-weight: 800;">Runs</div>
                                <div style="font-size: 1.1rem; font-weight: 800; color: var(--primary-dark);">{int(row.get('runs', 0))}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; font-weight: 800;">{"Avg" if 'batter' in str(row.get('role','')).lower() or 'batsman' in str(row.get('role','')).lower() else "Bowl Avg"}</div>
                                <div style="font-size: 1.1rem; font-weight: 800; color: var(--primary-dark);">{row.get('average', 0) if 'batter' in str(row.get('role','')).lower() or 'batsman' in str(row.get('role','')).lower() else row.get('bowling_average', 0):.1f}</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.6; font-weight: 800;">Wickets</div>
                                <div style="font-size: 1.1rem; font-weight: 800; color: var(--primary-dark);">{int(row.get('wickets', 0))}</div>
                            </div>
                        </div>
                        
                        <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 15px;">
                            <div style="font-size: 0.8rem; opacity: 0.6; align-self: center;">Help improve results:</div>
                    """, unsafe_allow_html=True)
                    
                    # Feedback buttons continue using Streamlit components for logic
                    col_fb1, col_fb2 = st.columns([8, 1]) # Shift buttons to the right
                    with col_fb2:
                        fb_c1, fb_c2 = st.columns(2)
                        with fb_c1:
                            if st.button("👍", key=f"up_{selected_player}_{row['player']}_{i}"):
                                save_scout_feedback(st.session_state.username, selected_player, row['player'], selected_fmt, "good")
                                st.toast("Feedback saved!", icon="✅")
                        with fb_c2:
                            if st.button("👎", key=f"down_{selected_player}_{row['player']}_{i}"):
                                save_scout_feedback(st.session_state.username, selected_player, row['player'], selected_fmt, "bad")
                                st.toast("Feedback noted!", icon="❌")
                    
                    st.markdown("</div>", unsafe_allow_html=True) # Close elite-card
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.error("Could not find similar players.")
