import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    get_tournament, get_tournament_teams, get_group_standings, 
    get_tournament_matches
)

def show_tournament_home():
    """Display tournament home page with matches and standings"""
    
    st.title("🏆 T20 World Cup 2024")
    
    # Get tournament (hardcoded for now, can be dynamic later)
    # In production, you'd select from available tournaments
    from database import get_db_connection
    conn = get_db_connection()
    tournaments = conn.execute("SELECT * FROM tournaments WHERE status != 'archived'").fetchall()
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
        st.header("📅 Matches")
        
        matches = get_tournament_matches(tournament_id)
        
        if not matches:
            st.info("No matches scheduled yet")
        else:
            # Filter tabs for match stages
            col1, col2, col3 = st.columns(3)
            with col1:
                show_group = st.checkbox("Group Stage", value=True)
            with col2:
                show_knockout = st.checkbox("Knockout", value=True)
            
            # Display matches
            for match in matches:
                # Skip based on filters
                if match['stage'] == 'group' and not show_group:
                    continue
                if match['stage'] != 'group' and not show_knockout:
                    continue
                
                # Get team names
                all_teams = get_tournament_teams(tournament_id)
                team1_name = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), f"Team {match['team1_id']}")
                team2_name = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), f"Team {match['team2_id']}")
                
                # Match card
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 3])
                    
                    with col1:
                        st.markdown(f"### {team1_name}")
                    
                    with col2:
                        if match['status'] == 'completed':
                            if match['winner_id'] == match['team1_id']:
                                st.markdown(f"### **{match['team1_score']}**")
                            else:
                                st.markdown(f"### {match['team1_score']}")
                        else:
                            st.markdown("### vs")
                    
                    with col3:
                        st.markdown(f"### {team2_name}")
                    
                    # Score line
                    if match['status'] == 'completed':
                        col1, col2, col3 = st.columns([3, 1, 3])
                        with col1:
                            st.write("")
                        with col2:
                            st.write("")
                        with col3:
                            if match['winner_id'] == match['team2_id']:
                                st.markdown(f"### **{match['team2_score']}**")
                            else:
                                st.markdown(f"### {match['team2_score']}")
                    
                    # Match details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status_color = "🟢" if match['status'] == 'completed' else "🔵"
                        st.caption(f"{status_color} {match['status'].upper()}")
                    with col2:
                        st.caption(f"📅 {match['match_date']}")
                    with col3:
                        stage_label = "Group" if match['stage'] == 'group' else match['stage'].replace('-', ' ').title()
                        st.caption(f"📍 {stage_label}")
                    
                    # Result message
                    if match['status'] == 'completed':
                        winner_team = next((t['team_name'] for t in all_teams if t['id'] == match['winner_id']), "")
                        st.success(f"🎉 {winner_team} won!", icon="✓")
                    
                    st.divider()
    
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
                        use_container_width=True
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
        
        from database import get_leaderboard
        
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
                use_container_width=True
            )
        else:
            st.info("Leaderboard will appear here once users create fantasy teams")

if __name__ == "__main__":
    show_tournament_home()
