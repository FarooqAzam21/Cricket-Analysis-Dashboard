import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from itertools import combinations
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    create_tournament, get_tournament, add_team_to_tournament, get_tournament_teams,
    create_tournament_match, get_tournament_matches, update_match_result,
    delete_tournament, update_team_squad, get_team_details, fetch_all_players_from_db,
    get_db_connection, update_match_date, get_group_stage_matches
)

def check_admin_access():
    """Check if user has admin access"""
    if 'username' not in st.session_state or st.session_state.username != 'admin':
        st.error("⛔ Unauthorized Access. Admin panel is only for administrators.")
        st.stop()

def show_admin_panel():
    """Main admin panel interface"""
    check_admin_access()
    
    st.title("🏆 T20 World Cup Fantasy Admin Panel")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Create Tournament", 
        "Add Teams to Groups", 
        "Add Players to Teams",
        "Schedule Matches",
        "Manage Matches", 
        "Update Scores"
    ])
    
    # ========== TAB 1: CREATE TOURNAMENT ==========
    with tab1:
        st.header("Create Tournament")
        
        col1, col2 = st.columns(2)
        with col1:
            tournament_name = st.text_input("Tournament Name", value="T20 World Cup 2024")
            start_date = st.date_input("Start Date")
        
        with col2:
            end_date = st.date_input("End Date", value=start_date + timedelta(days=30))
        
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
                    st.info("📝 Next: Go to 'Add Teams to Groups' tab to add your 20 teams")
            except Exception as e:
                st.error(f"Error creating tournament: {e}")
    
    # ========== TAB 2: ADD TEAMS TO GROUPS ==========
    with tab2:
        st.header("Add Teams to Groups")
        
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="add_teams_tournament")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                # Select group
                group_letter = st.selectbox("Select Group", ["A", "B", "C", "D"])
                
                # Get teams already in this group
                existing_teams = get_tournament_teams(tournament_id)
                existing_in_group = [t['team_name'] for t in existing_teams if t['group_letter'] == group_letter]
                
                st.subheader(f"Group {group_letter}")
                if existing_in_group:
                    st.write(f"Teams already in Group {group_letter}:")
                    for team in existing_in_group:
                        st.write(f"  ✅ {team}")
                else:
                    st.write(f"No teams yet in Group {group_letter}")
                
                # Add new team
                team_name = st.text_input(f"Add team to Group {group_letter}")
                
                if st.button(f"Add Team to Group {group_letter}", key=f"add_team_{group_letter}"):
                    if team_name:
                        try:
                            team_id = add_team_to_tournament(tournament_id, team_name, group_letter)
                            st.success(f"✅ {team_name} added to Group {group_letter}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding team: {e}")
                    else:
                        st.warning("Please enter a team name")
                
                # Show all teams
                st.divider()
                st.subheader("All Teams in Tournament")
                all_tournament_teams = get_tournament_teams(tournament_id)
                
                if all_tournament_teams:
                    groups_dict = {}
                    for team in all_tournament_teams:
                        group = team['group_letter']
                        if group not in groups_dict:
                            groups_dict[group] = []
                        groups_dict[group].append(team['team_name'])
                    
                    for group in ['A', 'B', 'C', 'D']:
                        teams = groups_dict.get(group, [])
                        st.write(f"**Group {group}** ({len(teams)}/5)")
                        for team in teams:
                            st.write(f"  • {team}")
            else:
                st.error("Tournament not found")
    
    # ========== TAB 3: ADD PLAYERS TO TEAMS ==========
    with tab3:
        st.header("Add Players to Teams")
        
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="add_players_tournament")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                # Get all teams
                all_teams = get_tournament_teams(tournament_id)
                
                if all_teams:
                    # Select team
                    team_options = {f"{t['team_name']} (Group {t['group_letter']})": t['id'] for t in all_teams}
                    team_display = st.selectbox("Select Team", team_options.keys())
                    team_id = team_options[team_display]
                    
                    team_details = get_team_details(team_id)
                    
                    # Get players from database
                    all_players_df = fetch_all_players_from_db()
                    
                    if all_players_df is not None and not all_players_df.empty:
                        # Filter T20 format
                        t20_players = all_players_df[all_players_df['format'] == 'T20'].copy()
                        
                        if not t20_players.empty:
                            # Multi-select players
                            player_list = t20_players['player'].unique().tolist()
                            
                            # Show current squad if exists
                            current_squad = []
                            if team_details['squad']:
                                try:
                                    current_squad = json.loads(team_details['squad'])
                                except:
                                    current_squad = []
                            
                            st.write(f"**Current Squad** ({len(current_squad)} players):")
                            if current_squad:
                                for idx, player in enumerate(current_squad, 1):
                                    st.write(f"  {idx}. {player}")
                            else:
                                st.write("  No players yet")
                            
                            st.divider()
                            
                            # Select players
                            selected_players = st.multiselect(
                                "Select 15 players for squad",
                                options=player_list,
                                default=current_squad,
                                max_selections=15,
                                key=f"squad_selector_{team_id}"
                            )
                            
                            st.info(f"Selected: {len(selected_players)}/15 players")
                            
                            if st.button(f"Save Squad for {team_details['team_name']}", key=f"save_squad_{team_id}"):
                                if len(selected_players) > 0:
                                    try:
                                        squad_json = json.dumps(selected_players)
                                        update_team_squad(team_id, squad_json)
                                        st.success(f"✅ Squad updated with {len(selected_players)} players")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error saving squad: {e}")
                                else:
                                    st.warning("Please select at least one player")
                        else:
                            st.warning("No T20 format players found in database")
                    else:
                        st.error("Could not load players from database")
                else:
                    st.warning("No teams found. Add teams first in 'Add Teams to Groups' tab")
            else:
                st.error("Tournament not found")
    
    # ========== TAB 4: SCHEDULE MATCHES ==========
    with tab4:
        st.header("Schedule Matches")
        
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="schedule_tournament")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                all_teams = get_tournament_teams(tournament_id)
                
                if all_teams:
                    # Create match schedule
                    st.subheader("Create Group Stage Matches")
                    
                    # Group teams by group
                    groups_dict = {}
                    for team in all_teams:
                        group = team['group_letter']
                        if group not in groups_dict:
                            groups_dict[group] = []
                        groups_dict[group].append(team)
                    
                    # Generate matches for each group
                    if st.button("Auto-Generate Group Stage Matches", key="auto_gen_matches"):
                        try:
                            match_counter = 1
                            base_date = datetime.strptime(tournament['start_date'], "%Y-%m-%d")
                            
                            for group_letter in ['A', 'B', 'C', 'D']:
                                group_teams = groups_dict.get(group_letter, [])
                                
                                # Round-robin: each team plays every other team once
                                for i in range(len(group_teams)):
                                    for j in range(i + 1, len(group_teams)):
                                        team1 = group_teams[i]
                                        team2 = group_teams[j]
                                        
                                        match_date = (base_date + timedelta(days=match_counter)).strftime("%Y-%m-%d")
                                        
                                        create_tournament_match(
                                            tournament_id,
                                            team1['id'],
                                            team2['id'],
                                            match_date,
                                            'group',
                                            group_letter
                                        )
                                        match_counter += 1
                            
                            st.success("✅ Group stage matches scheduled!")
                            st.info("📍 Matches will be played in round-robin format within each group")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Error scheduling matches: {e}")
                    
                    # Edit generated matches
                    st.divider()
                    st.subheader("Edit Match Schedule")
                    
                    group_matches = get_group_stage_matches(tournament_id)
                    
                    if group_matches:
                        st.write(f"Found {len(group_matches)} group stage matches. Edit dates and numbers below:")
                        
                        # Create editable table
                        match_data = []
                        for idx, m in enumerate(group_matches, 1):
                            team1 = next((t['team_name'] for t in all_teams if t['id'] == m['team1_id']), f"Team {m['team1_id']}")
                            team2 = next((t['team_name'] for t in all_teams if t['id'] == m['team2_id']), f"Team {m['team2_id']}")
                            
                            match_data.append({
                                'Match #': idx,
                                'ID': m['id'],
                                'Team 1': team1,
                                'Team 2': team2,
                                'Current Date': m['match_date'],
                                'Group': m['group_letter']
                            })
                        
                        st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)
                        
                        st.write("**Edit Match Dates:**")
                        
                        # Create columns for editing
                        edit_cols = st.columns([2, 2, 1])
                        
                        with edit_cols[0]:
                            match_to_edit = st.selectbox(
                                "Select Match to Edit",
                                options=[f"Match {idx}: {md['Team 1']} vs {md['Team 2']}" for idx, md in enumerate(match_data, 1)],
                                key="edit_match_select"
                            )
                        
                        match_idx = int(match_to_edit.split(':')[0].split()[1]) - 1
                        selected_match = group_matches[match_idx]
                        
                        with edit_cols[1]:
                            new_date = st.date_input(
                                "New Date",
                                value=datetime.strptime(selected_match['match_date'], "%Y-%m-%d").date(),
                                key="new_match_date"
                            )
                        
                        with edit_cols[2]:
                            if st.button("Update Date", key="update_date_btn"):
                                try:
                                    update_match_date(selected_match['id'], new_date.strftime("%Y-%m-%d"))
                                    st.success(f"✅ Match date updated to {new_date}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating date: {e}")
                    else:
                        st.info("No group stage matches generated yet. Click 'Auto-Generate Group Stage Matches' first.")
                    
                    st.divider()
                    st.subheader("Create Knockout Matches")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Semi-Final 1**")
                        sf1_team1 = st.selectbox("SF1 Team 1", [t['team_name'] for t in all_teams], key="sf1_t1")
                        sf1_team2 = st.selectbox("SF1 Team 2", [t['team_name'] for t in all_teams if t['team_name'] != sf1_team1], key="sf1_t2")
                        sf1_date = st.date_input("SF1 Date", key="sf1_date")
                    
                    with col2:
                        st.write("**Semi-Final 2**")
                        sf2_team1 = st.selectbox("SF2 Team 1", [t['team_name'] for t in all_teams], key="sf2_t1")
                        sf2_team2 = st.selectbox("SF2 Team 2", [t['team_name'] for t in all_teams if t['team_name'] != sf2_team1], key="sf2_t2")
                        sf2_date = st.date_input("SF2 Date", key="sf2_date")
                    
                    final_date = st.date_input("Final Date")
                    
                    if st.button("Create Knockout Matches", key="create_knockout"):
                        try:
                            sf1_t1_id = next(t['id'] for t in all_teams if t['team_name'] == sf1_team1)
                            sf1_t2_id = next(t['id'] for t in all_teams if t['team_name'] == sf1_team2)
                            sf2_t1_id = next(t['id'] for t in all_teams if t['team_name'] == sf2_team1)
                            sf2_t2_id = next(t['id'] for t in all_teams if t['team_name'] == sf2_team2)
                            
                            create_tournament_match(tournament_id, sf1_t1_id, sf1_t2_id, sf1_date.strftime("%Y-%m-%d"), 'semi-final')
                            create_tournament_match(tournament_id, sf2_t1_id, sf2_t2_id, sf2_date.strftime("%Y-%m-%d"), 'semi-final')
                            
                            st.success("✅ Knockout matches scheduled!")
                            st.info("Final will use winners from semi-finals")
                        except Exception as e:
                            st.error(f"Error creating knockout: {e}")
                else:
                    st.warning("No teams found. Add teams first in 'Add Teams to Groups' tab")
            else:
                st.error("Tournament not found")
    
    # ========== TAB 5: MANAGE MATCHES ==========
    with tab5:
        st.header("Manage Matches")
        
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="manage_tournament")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                matches = get_tournament_matches(tournament_id)
                
                if matches:
                    col1, col2 = st.columns(2)
                    with col1:
                        stage_filter = st.selectbox("Filter by Stage", ["All", "group", "semi-final", "final"], key="stage_filter")
                    with col2:
                        status_filter = st.selectbox("Filter by Status", ["All", "scheduled", "completed"], key="status_filter")
                    
                    # Filter matches
                    filtered_matches = matches
                    if stage_filter != "All":
                        filtered_matches = [m for m in filtered_matches if m['stage'] == stage_filter]
                    if status_filter != "All":
                        filtered_matches = [m for m in filtered_matches if m['status'] == status_filter]
                    
                    if filtered_matches:
                        all_teams = get_tournament_teams(tournament_id)
                        match_data = []
                        for m in filtered_matches:
                            team1 = next((t['team_name'] for t in all_teams if t['id'] == m['team1_id']), f"Team {m['team1_id']}")
                            team2 = next((t['team_name'] for t in all_teams if t['id'] == m['team2_id']), f"Team {m['team2_id']}")
                            
                            match_data.append({
                                'ID': m['id'],
                                'Team 1': team1,
                                'Team 2': team2,
                                'Date': m['match_date'],
                                'Stage': m['stage'].title(),
                                'Status': m['status'].title(),
                            })
                        
                        st.dataframe(pd.DataFrame(match_data), use_container_width=True)
                    else:
                        st.info("No matches found with selected filters")
                else:
                    st.warning("No matches found. Create matches in 'Schedule Matches' tab")
            else:
                st.error("Tournament not found")
    
    # ========== TAB 6: UPDATE SCORES ==========
    with tab6:
        st.header("Update Match Scores")
        
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="score_tournament")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.info(f"Tournament: {tournament['name']}")
                
                matches = get_tournament_matches(tournament_id)
                incomplete_matches = [m for m in matches if m['status'] != 'completed']
                
                if incomplete_matches:
                    all_teams = get_tournament_teams(tournament_id)
                    match_options = {}
                    
                    for m in incomplete_matches:
                        team1 = next((t['team_name'] for t in all_teams if t['id'] == m['team1_id']), f"Team {m['team1_id']}")
                        team2 = next((t['team_name'] for t in all_teams if t['id'] == m['team2_id']), f"Team {m['team2_id']}")
                        match_options[f"{team1} vs {team2} ({m['match_date']})"] = m['id']
                    
                    selected_match_display = st.selectbox("Select Match", match_options.keys())
                    match_id = match_options[selected_match_display]
                    match = next(m for m in incomplete_matches if m['id'] == match_id)
                    
                    col1, col2, col3 = st.columns([3, 1, 3])
                    
                    with col1:
                        team1_name = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), f"Team {match['team1_id']}")
                        st.write(f"### {team1_name}")
                        team1_score = st.number_input(f"{team1_name} Runs", min_value=0, step=1, key="team1_score")
                    
                    with col2:
                        st.markdown("**VS**")
                    
                    with col3:
                        team2_name = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), f"Team {match['team2_id']}")
                        st.write(f"### {team2_name}")
                        team2_score = st.number_input(f"{team2_name} Runs", min_value=0, step=1, key="team2_score")
                    
                    # Select winner
                    winner_options = {
                        team1_name: match['team1_id'],
                        team2_name: match['team2_id'],
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
    
    # ========== DELETE TOURNAMENT ==========
    st.divider()
    st.subheader("⚠️ Danger Zone")
    
    delete_tournament_id = st.number_input("Tournament ID to Delete", min_value=1, step=1, key="delete_id")
    
    if delete_tournament_id:
        tournament = get_tournament(delete_tournament_id)
        if tournament:
            st.warning(f"⚠️ This will permanently delete '{tournament['name']}' and all related data")
            
            confirm = st.checkbox(f"I confirm deletion of '{tournament['name']}'")
            
            if confirm and st.button("🗑️ Delete Tournament", key="delete_tournament"):
                try:
                    if delete_tournament(delete_tournament_id):
                        st.success(f"✅ Tournament '{tournament['name']}' deleted successfully")
                        st.balloons()
                    else:
                        st.error("Failed to delete tournament")
                except Exception as e:
                    st.error(f"Error deleting tournament: {e}")

if __name__ == "__main__":
    show_admin_panel()

