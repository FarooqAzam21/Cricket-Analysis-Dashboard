import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from itertools import combinations
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    create_tournament, get_tournament, add_team_to_tournament, get_tournament_teams,
    create_tournament_match, get_tournament_matches, update_match_result
)

def check_admin_access():
    """Check if user has admin access"""
    if 'username' not in st.session_state or st.session_state.username != 'admin':
        st.error("⛔ Unauthorized Access. Admin panel is only for administrators.")
        st.stop()

def get_t20_teams():
    """Return list of T20 World Cup teams"""
    teams = {
        'Group A': ['India', 'Pakistan', 'Afghanistan', 'Australia', 'Sri Lanka'],
        'Group B': ['England', 'West Indies', 'South Africa', 'New Zealand', 'Bangladesh'],
        'Group C': ['USA', 'Ireland', 'Zimbabwe', 'Kenya', 'UAE'],
        'Group D': ['Canada', 'Netherlands', 'Scotland', 'Oman', 'Namibia']
    }
    return teams

def auto_generate_group_stage_matches(tournament_id, teams_dict):
    """Generate round-robin matches for group stage"""
    match_counter = 1
    base_date = datetime.now()
    
    for group, group_teams in teams_dict.items():
        group_letter = group.split()[-1]  # Extract 'A', 'B', 'C', 'D'
        
        # Get team IDs for this group
        all_tournament_teams = get_tournament_teams(tournament_id)
        group_team_ids = {
            t['team_name']: t['id'] 
            for t in all_tournament_teams 
            if t['group_letter'] == group_letter
        }
        
        # Generate round-robin matches
        team_list = list(group_team_ids.items())
        for team1_name, team2_name in combinations(range(len(team_list)), 2):
            team1_db = team_list[team1_name]
            team2_db = team_list[team2_name]
            
            match_date = (base_date + timedelta(days=match_counter)).strftime("%Y-%m-%d")
            create_tournament_match(
                tournament_id,
                team1_db[1],  # team1_id
                team2_db[1],  # team2_id
                match_date,
                'group',
                group_letter
            )
            match_counter += 1

def generate_knockout_matches(tournament_id, teams_dict):
    """Generate knockout matches based on group standings"""
    # This will be implemented after group stage completes
    # Top 2 teams from each group advance to knockouts
    # Semi-finals, Finals
    pass

def show_admin_panel():
    """Main admin panel interface"""
    check_admin_access()
    
    st.title("🏆 T20 World Cup Fantasy Admin Panel")
    
    tab1, tab2, tab3 = st.tabs(["Create Tournament", "Manage Matches", "Update Scores"])
    
    # ========== TAB 1: CREATE TOURNAMENT ==========
    with tab1:
        st.header("Create Tournament")
        
        col1, col2 = st.columns(2)
        with col1:
            tournament_name = st.text_input("Tournament Name", value="T20 World Cup 2024")
            start_date = st.date_input("Start Date")
        
        with col2:
            end_date = st.date_input("End Date", value=start_date + timedelta(days=30))
            auto_setup = st.checkbox("Auto-setup with standard T20 teams & groups")
        
        if st.button("Create Tournament", key="create_tournament"):
            try:
                tournament_id = create_tournament(
                    tournament_name,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )
                
                if tournament_id:
                    st.success(f"✅ Tournament created with ID: {tournament_id}")
                    st.session_state.current_tournament_id = tournament_id
                    
                    if auto_setup:
                        with st.spinner("Setting up teams and groups..."):
                            teams_dict = get_t20_teams()
                            
                            # Add all teams to tournament with groups
                            for group, group_teams in teams_dict.items():
                                group_letter = group.split()[-1]
                                for team_name in group_teams:
                                    add_team_to_tournament(tournament_id, team_name, group_letter)
                            
                            # Generate group stage matches
                            auto_generate_group_stage_matches(tournament_id, teams_dict)
                            
                        st.success("✅ All 20 teams added to 4 groups with group stage matches generated!")
                        st.balloons()
            except Exception as e:
                st.error(f"Error creating tournament: {e}")
        
        # Display tournament structure
        st.markdown("### Tournament Structure (T20 Format)")
        teams_dict = get_t20_teams()
        
        for group, group_teams in teams_dict.items():
            with st.expander(f"{group} ({len(group_teams)} Teams)"):
                st.write(group_teams)
    
    # ========== TAB 2: MANAGE MATCHES ==========
    with tab2:
        st.header("Manage Matches")
        
        # Select tournament
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1)
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                # Get matches
                matches = get_tournament_matches(tournament_id)
                
                if matches:
                    # Filter by stage
                    stage_filter = st.selectbox("Filter by Stage", ["All", "group", "semi-final", "final"])
                    
                    if stage_filter != "All":
                        matches = [m for m in matches if m['stage'] == stage_filter]
                    
                    # Display matches in a table
                    match_data = []
                    for m in matches:
                        match_data.append({
                            'ID': m['id'],
                            'Date': m['match_date'],
                            'Team 1 ID': m['team1_id'],
                            'Team 2 ID': m['team2_id'],
                            'Stage': m['stage'],
                            'Status': m['status'],
                            'Winner': m['winner_id'] if m['winner_id'] else '-'
                        })
                    
                    st.dataframe(pd.DataFrame(match_data), use_container_width=True)
                else:
                    st.warning("No matches found for this tournament")
            else:
                st.error("Tournament not found")
    
    # ========== TAB 3: UPDATE SCORES ==========
    with tab3:
        st.header("Update Match Scores")
        
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="score_tournament")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                # Get incomplete matches
                matches = get_tournament_matches(tournament_id)
                incomplete_matches = [m for m in matches if m['status'] != 'completed']
                
                if incomplete_matches:
                    # Select match
                    match_options = {
                        f"Match {m['id']}: Team {m['team1_id']} vs Team {m['team2_id']} ({m['match_date']})": m['id']
                        for m in incomplete_matches
                    }
                    
                    selected_match_display = st.selectbox("Select Match", match_options.keys())
                    match_id = match_options[selected_match_display]
                    
                    # Get match details
                    match = next((m for m in matches if m['id'] == match_id), None)
                    
                    if match:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            team1_score = st.number_input(f"Team {match['team1_id']} Runs", min_value=0, step=1)
                        
                        with col2:
                            st.markdown("**VS**")
                        
                        with col3:
                            team2_score = st.number_input(f"Team {match['team2_id']} Runs", min_value=0, step=1)
                        
                        # Select winner
                        winner_options = {
                            f"Team {match['team1_id']}": match['team1_id'],
                            f"Team {match['team2_id']}": match['team2_id'],
                            "No Result": None
                        }
                        
                        winner_display = st.selectbox("Match Winner", winner_options.keys())
                        winner_id = winner_options[winner_display]
                        
                        if st.button("Update Score", key="update_score"):
                            try:
                                update_match_result(match_id, winner_id, team1_score, team2_score)
                                st.success("✅ Match score updated successfully!")
                                st.balloons()
                            except Exception as e:
                                st.error(f"Error updating score: {e}")
                else:
                    st.info("✅ All matches completed!")
            else:
                st.error("Tournament not found")

if __name__ == "__main__":
    show_admin_panel()
