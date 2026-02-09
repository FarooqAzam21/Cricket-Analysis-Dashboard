import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import (
    get_tournament, get_tournament_teams, get_group_standings, 
    get_tournament_matches, get_db_connection, get_leaderboard
)

def show_tournament_home():
    """Display tournament home page with matches and standings"""
    
    st.title("🏆 T20 World Cup 2024")
    
    # Get tournament (hardcoded for now, can be dynamic later)
    # In production, you'd select from available tournaments
    conn = get_db_connection()
    tournaments = conn.execute("SELECT * FROM tournaments ORDER BY id DESC").fetchall()
    conn.close()
    
    if not tournaments:
        st.info("📋 No active tournaments. Check back soon!")
        return
    
    # Select tournament if multiple
    if len(tournaments) > 1:
        tournament_options = {t['name']: t['id'] for t in tournaments}
        tournament_name = st.selectbox("Select Tournament", tournament_options.keys())
        tournament_id = tournament_options[tournament_name]
    else:
        tournament_id = tournaments[0]['id']
        st.header(tournaments[0]['name'])
    
    tournament = get_tournament(tournament_id)
    
    if not tournament:
        st.error("Tournament not found")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Matches", "Group Standings", "Knockout", "Leaderboard"])
    
    # ========== MATCHES TAB ==========
    with tab1:
        from src.database import get_tournament_stats
        
        st.header("📅 Tournament Pulse")
        matches = get_tournament_matches(tournament_id)
        
        if not matches:
            st.info("No matches scheduled yet")
        else:
            # --- UPCOMING MATCHES (HORIZONTAL) ---
            upcoming = [m for m in matches if m['status'] != 'completed']
            if upcoming:
                st.subheader("🔥 Upcoming Matches")
                up_cols = st.columns(min(len(upcoming), 3))
                for i, match in enumerate(upcoming[:3]):
                    with up_cols[i]:
                        all_teams = get_tournament_teams(tournament_id)
                        t1 = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), "T1")
                        t2 = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), "T2")
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); text-align: center; margin-bottom: 20px;">
                            <div style="font-size: 0.8rem; opacity: 0.6; margin-bottom: 5px;">📅 {match['match_date']}</div>
                            <div style="font-weight: bold; font-size: 1.1rem;">{t1}</div>
                            <div style="color: #238636; font-size: 0.8rem; margin: 4px 0;">VS</div>
                            <div style="font-weight: bold; font-size: 1.1rem;">{t2}</div>
                            <div style="font-size: 0.75rem; opacity: 0.5; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 5px;">{match['stage'].upper()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if len(upcoming) > 3:
                    with st.expander("View All Upcoming Matches"):
                        for match in upcoming[3:]:
                            st.write(f"📅 {match['match_date']} | {match['stage']} | {match['team1_id']} vs {match['team2_id']}")
            
            st.divider()
            
            # --- COMPLETED MATCHES & TOURNAMENT STATS ---
            st.subheader("📊 Live Tournament Standings & Stats")
            
            s_col1, s_col2 = st.columns([2, 1])
            
            with s_col1:
                st.markdown("### 🎯 Top Run Scorers")
                df_runs = get_tournament_stats(tournament_id, 'runs')
                if not df_runs.empty:
                    st.dataframe(df_runs, width="stretch", hide_index=True)
                else:
                    st.info("Batting stats will appear after match results are entered.")
                
                st.markdown("### 🎲 Top Wicket Takers")
                df_wkts = get_tournament_stats(tournament_id, 'wickets')
                if not df_wkts.empty:
                    st.dataframe(df_wkts, width="stretch", hide_index=True)
            
            with s_col2:
                st.markdown("### 🔥 Most Sixes")
                df_sixes = get_tournament_stats(tournament_id, 'sixes')
                if not df_sixes.empty:
                    st.dataframe(df_sixes, width="stretch", hide_index=True)
                
                st.markdown("### 🏁 Recent Results")
                completed = [m for m in matches if m['status'] == 'completed']
                for match in completed[-3:]:
                    all_teams = get_tournament_teams(tournament_id)
                    t1 = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), "T1")
                    t2 = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), "T2")
                    winner = t1 if match['winner_id'] == match['team1_id'] else t2
                    st.markdown(f"""
                    <div style="font-size: 0.85rem; padding: 10px; border-left: 3px solid #238636; background: rgba(35, 134, 54, 0.05); margin-bottom: 8px;">
                        <b>{t1}</b> vs <b>{t2}</b><br/>
                        <span style="color: #238636;">Winner: {winner}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========== GROUP STANDINGS TAB ==========
    with tab2:
        st.header("🗂️ Group Standings")
        
        groups = ['A', 'B', 'C', 'D']
        
        col1, col2 = st.columns(2)
        
        for idx, group in enumerate(groups):
            with col1 if idx % 2 == 0 else col2:
                st.subheader(f"Group {group}")
                
                standings = get_group_standings(tournament_id, group)
                
                if standings:
                    standings_data = []
                    for rank, team in enumerate(standings, 1):
                        standings_data.append({
                            'Rank': rank,
                            'Team': team['team_name'],
                            'Played': team['matches_played'],
                            'Won': team['wins'],
                            'Lost': team['losses'],
                            'Points': team['points']
                        })
                    
                    st.dataframe(
                        pd.DataFrame(standings_data),
                        hide_index=True,
                        width="stretch"
                    )
                else:
                    st.info("No standings data available")
    
    # ========== KNOCKOUT TAB ==========
    with tab3:
        st.header("🏅 Knockout Stage")
        
        knockout_matches = [m for m in matches if m['stage'] != 'group']
        
        if knockout_matches:
            # Organize by stage
            semi_finals = [m for m in knockout_matches if m['stage'] == 'semi-final']
            finals = [m for m in knockout_matches if m['stage'] == 'final']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Semi-Finals")
                for match in semi_finals:
                    all_teams = get_tournament_teams(tournament_id)
                    team1 = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), f"Team {match['team1_id']}")
                    team2 = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), f"Team {match['team2_id']}")
                    
                    st.write(f"{team1} vs {team2}")
                    if match['status'] == 'completed':
                        st.success(f"Winner: {next((t['team_name'] for t in all_teams if t['id'] == match['winner_id']), 'TBD')}")
            
            with col2:
                st.subheader("Final")
                for match in finals:
                    all_teams = get_tournament_teams(tournament_id)
                    team1 = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), f"Team {match['team1_id']}")
                    team2 = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), f"Team {match['team2_id']}")
                    
                    st.write(f"{team1} vs {team2}")
                    if match['status'] == 'completed':
                        all_teams = get_tournament_teams(tournament_id)
                        winner = next((t['team_name'] for t in all_teams if t['id'] == match['winner_id']), 'TBD')
                        st.success(f"🏆 Champion: {winner}")
        else:
            st.info("Knockout stage matches will appear here after group stage completes")
    
    # ========== LEADERBOARD TAB ==========
    with tab4:
        st.header("🏆 Fantasy Leaderboard")
        
        leaderboard = get_leaderboard(tournament_id)
        
        if leaderboard:
            leaderboard_data = []
            for rank, entry in enumerate(leaderboard, 1):
                leaderboard_data.append({
                    'Rank': rank,
                    'User': entry['username'],
                    'Total Points': entry['total_points'],
                    'Teams Created': entry['fantasy_teams_created']
                })
            
            st.dataframe(
                pd.DataFrame(leaderboard_data),
                hide_index=True,
                width="stretch"
            )
        else:
            st.info("Leaderboard will appear here once users create fantasy teams")

if __name__ == "__main__":
    show_tournament_home()
