import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    get_tournament, get_tournament_matches, get_tournament_teams,
    save_fantasy_team, get_user_fantasy_teams, get_db_connection, fetch_all_players_from_db
)

def get_upcoming_and_scheduled_matches(tournament_id):
    """Get all scheduled/upcoming matches (not completed)"""
    matches = get_tournament_matches(tournament_id)
    return [m for m in matches if m['status'] != 'completed']

def get_match_squads(match_id):
    """Get all players from both teams in a match"""
    all_players = fetch_all_players_from_db()
    
    match = None
    conn = get_db_connection()
    match = conn.execute("SELECT * FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
    conn.close()
    
    if not match:
        return pd.DataFrame()
    
    # Get both teams' players from database
    # Filter by team and format = 'T20'
    team1_id = match['team1_id']
    team2_id = match['team2_id']
    
    all_teams = get_tournament_teams(match['tournament_id'])
    team1_name = next((t['team_name'] for t in all_teams if t['id'] == team1_id), None)
    team2_name = next((t['team_name'] for t in all_teams if t['id'] == team2_id), None)
    
    # Filter players from both teams
    squad = all_players[
        ((all_players['team'] == team1_name) | (all_players['team'] == team2_name)) &
        (all_players['format'] == 'T20')
    ].copy()
    
    return squad, team1_name, team2_name

def calculate_fantasy_points(players_data, match_results):
    """Calculate fantasy points based on match performance"""
    # Scoring system:
    # Batting: 1 point per run, 2 points per 4, 3 points per 6
    # Bowling: 20 points per wicket, 4 points per economy <7, etc.
    # Fielding: 10 points per catch, 15 points per runout, 25 points per stumping
    # Bonuses: 50 points for 50, 100 points for 100
    
    points = 0
    
    # This is a placeholder - in production, you'd integrate with live APIs
    # For now, we'll use a formula based on player stats
    
    return points

def show_fantasy_cricket():
    """Fantasy cricket team builder interface"""
    
    st.title("🏏 Fantasy Cricket")
    
    # Get active tournaments
    conn = get_db_connection()
    tournaments = conn.execute("SELECT * FROM tournaments WHERE status IN ('planning', 'active')").fetchall()
    conn.close()
    
    if not tournaments:
        st.info("No tournaments available")
        return
    
    # Select tournament
    tournament_options = {t['name']: t['id'] for t in tournaments}
    tournament_name = st.selectbox("Select Tournament", tournament_options.keys())
    tournament_id = tournament_options[tournament_name]
    
    tournament = get_tournament(tournament_id)
    
    # Get upcoming/scheduled matches (not completed yet)
    upcoming_matches = get_upcoming_and_scheduled_matches(tournament_id)
    
    if not upcoming_matches:
        st.warning("⏳ No upcoming matches. Check back when matches are scheduled!")
        return
    
    # Select match
    st.subheader("Select Match to Create Fantasy Team")
    all_teams = get_tournament_teams(tournament_id)
    
    match_options = {}
    for match in upcoming_matches:
        team1 = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), f"Team {match['team1_id']}")
        team2 = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), f"Team {match['team2_id']}")
        status_badge = "🔴 Scheduled" if match['status'] == 'scheduled' else "🟡 Live"
        match_options[f"{team1} vs {team2} ({match['match_date']}) {status_badge}"] = match['id']
    
    selected_match_display = st.selectbox("Pick a Match", match_options.keys())
    match_id = match_options[selected_match_display]
    
    # Get match details
    selected_match = next((m for m in upcoming_matches if m['id'] == match_id), None)
    
    if not selected_match:
        st.error("Match not found")
        return
    
    team1 = next((t['team_name'] for t in all_teams if t['id'] == selected_match['team1_id']), "Team 1")
    team2 = next((t['team_name'] for t in all_teams if t['id'] == selected_match['team2_id']), "Team 2")
    
    # Display match info
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.write(f"### {team1}")
        if selected_match['status'] == 'completed':
            st.metric("Runs", selected_match['team1_score'])
        else:
            st.info("Match not started yet")
    with col2:
        st.write("**VS**")
        st.write(f"📅 {selected_match['match_date']}")
    with col3:
        st.write(f"### {team2}")
        if selected_match['status'] == 'completed':
            st.metric("Runs", selected_match['team2_score'])
        else:
            st.info("Match not started yet")
    
    st.divider()
    
    # Get squads
    squad_df, team1_name, team2_name = get_match_squads(match_id)
    
    if squad_df.empty:
        st.error("Could not load player squads")
        return
    
    # Team builder
    st.subheader("🏗️ Build Your Fantasy Team (11 Players)")
    
    # Create team selection
    team_selections = {}
    cols = st.columns(3)
    
    st.write("**Select 11 players from both teams**")
    
    # Separate by teams
    team1_players = squad_df[squad_df['team'] == team1_name].copy()
    team2_players = squad_df[squad_df['team'] == team2_name].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"#### {team1_name}")
        team1_selected = st.multiselect(
            f"Select {team1_name} players",
            options=team1_players['player'].tolist(),
            key=f"team1_{match_id}",
            max_selections=11
        )
    
    with col2:
        st.write(f"#### {team2_name}")
        team2_selected = st.multiselect(
            f"Select {team2_name} players",
            options=team2_players['player'].tolist(),
            key=f"team2_{match_id}",
            max_selections=11
        )
    
    all_selected = team1_selected + team2_selected
    
    # Display player count
    st.info(f"Selected Players: {len(all_selected)}/11")
    
    if len(all_selected) == 11:
        st.success("✅ Full team selected!")
        
        # Show selected team preview
        st.subheader("Your Fantasy Team Preview")
        selected_df = squad_df[squad_df['player'].isin(all_selected)][
            ['player', 'team', 'role', 'runs', 'wickets', 'strike_rate', 'average']
        ].copy()
        
        selected_df = selected_df.sort_values('team')
        st.dataframe(selected_df, hide_index=True, use_container_width=True)
        
        # Positions
        st.subheader("Assign Batting Positions (1-11)")
        position_assignments = {}
        
        cols = st.columns(3)
        for idx, player in enumerate(all_selected):
            col = cols[idx % 3]
            with col:
                position = st.number_input(
                    f"{player}",
                    min_value=1,
                    max_value=11,
                    value=idx + 1,
                    key=f"position_{player}_{match_id}"
                )
                position_assignments[player] = position
        
        # Captain selection
        st.subheader("📍 Select Captain & Vice-Captain")
        col1, col2 = st.columns(2)
        
        with col1:
            captain = st.selectbox("Captain", all_selected, key=f"captain_{match_id}")
        
        with col2:
            vice_captain = st.selectbox(
                "Vice-Captain",
                [p for p in all_selected if p != captain],
                key=f"vice_captain_{match_id}"
            )
        
        # Submit fantasy team
        if st.button("🚀 Submit Fantasy Team", key=f"submit_{match_id}"):
            try:
                # Get user ID
                conn = get_db_connection()
                user = conn.execute(
                    "SELECT id FROM users WHERE username = ?",
                    (st.session_state.username,)
                ).fetchone()
                conn.close()
                
                if not user:
                    st.error("User not found")
                    return
                
                user_id = user['id']
                
                # Save fantasy team
                players_json = json.dumps({
                    'players': all_selected,
                    'positions': position_assignments,
                    'captain': captain,
                    'vice_captain': vice_captain
                })
                
                # Get captain and vice captain IDs
                captain_id = next((p['id'] for p in squad_df.itertuples()), None)
                vice_captain_id = next((p['id'] for p in squad_df.itertuples()), None)
                
                fantasy_team_id = save_fantasy_team(
                    user_id,
                    tournament_id,
                    match_id,
                    players_json,
                    captain_id,
                    vice_captain_id
                )
                
                st.success("🎉 Fantasy team submitted successfully!")
                st.balloons()
                
                # Show points estimation
                st.info(f"Your team will be scored based on player performance.")
                
            except Exception as e:
                st.error(f"Error submitting fantasy team: {e}")
    
    elif len(all_selected) > 11:
        st.warning(f"⚠️ You've selected {len(all_selected)} players. Maximum is 11.")
    
    # Show previous teams
    st.divider()
    st.subheader("Your Previous Fantasy Teams")
    
    try:
        conn = get_db_connection()
        user = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (st.session_state.username,)
        ).fetchone()
        conn.close()
        
        if user:
            previous_teams = get_user_fantasy_teams(user['id'], tournament_id)
            
            if previous_teams:
                for team in previous_teams:
                    with st.expander(f"Team from match {team['match_id']} ({team['created_at']})"):
                        team_data = json.loads(team['players_json'])
                        st.write(f"**Players:** {', '.join(team_data['players'])}")
                        st.write(f"**Captain:** {team_data['captain']}")
                        st.write(f"**Vice-Captain:** {team_data['vice_captain']}")
            else:
                st.info("You haven't created any fantasy teams yet")
    except Exception as e:
        st.error(f"Error loading previous teams: {e}")

if __name__ == "__main__":
    show_fantasy_cricket()
