import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from itertools import combinations
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import (
    create_tournament, get_tournament, add_team_to_tournament, get_tournament_teams,
    create_tournament_match, get_tournament_matches, update_match_result,
    delete_tournament, update_team_squad, get_team_details, fetch_all_players_from_db,
    get_db_connection, update_match_date, get_group_stage_matches, calculate_and_save_fantasy_points,
    update_team_name, update_match_number, add_player_performance, get_match_performances,
    calculate_team_strength, get_team_strength_rating, calculate_updated_fantasy_scores,
    promote_to_super8, get_tournament_stats, delete_match,
    save_playing_xi, get_playing_xi
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
    
    # ========== TOURNAMENTS OVERVIEW ==========
    st.divider()
    st.subheader("📋 Existing Tournaments")
    
    conn = get_db_connection()
    all_tournaments = conn.execute("SELECT * FROM tournaments ORDER BY id DESC").fetchall()
    conn.close()
    
    if all_tournaments:
        tourn_data = []
        for t in all_tournaments:
            tourn_data.append({
                'ID': t['id'],
                'Tournament': t['name'],
                'Start Date': t['start_date'],
                'End Date': t['end_date'],
                'Status': t['status']
            })
        
        st.dataframe(pd.DataFrame(tourn_data), use_container_width=True)
        st.info(f"💡 Use the Tournament ID from the table above to manage tournaments in the tabs below")
    else:
        st.warning("No tournaments found. Create one in Tab 1 below.")
    
    st.divider()
    
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0d1117;
            padding: 10px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            white-space: pre-wrap;
            background-color: #161b22;
            border-radius: 8px;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 0 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #238636 !important;
            color: white !important;
            border-color: #2ea043 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🆕 Create Tournament", 
        "👥 Groups", 
        "👤 Add Players",
        "📅 Schedule",
        "⚙️ Manage Matches", 
        "🏏 Update Scores",
        "🏆 Super 8",
        "📊 Tournament Stats",
        "🛠️ Master Player Control"
    ])
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = tabs
    
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
                    team_id_map = {}
                    for team in all_tournament_teams:
                        group = team['group_letter']
                        if group not in groups_dict:
                            groups_dict[group] = []
                        groups_dict[group].append(team['team_name'])
                        team_id_map[team['team_name']] = team['id']
                    
                    for idx, group in enumerate(['A', 'B', 'C', 'D']):
                        teams = groups_dict.get(group, [])
                        st.write(f"**Group {group}** ({len(teams)}/5)")
                        for team_idx, team in enumerate(teams):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"  • {team}")
                            with col2:
                                # Use unique key with tournament_id, group and team index
                                if st.button("✏️ Edit", key=f"edit_team_{tournament_id}_{group}_{team_idx}"):
                                    st.session_state[f"editing_team_{team_id_map[team]}"] = True
                    
                    # Edit team name section
                    st.divider()
                    st.subheader("🔧 Edit Team Name")
                    
                    edit_team_options = {f"{t['team_name']} (Group {t['group_letter']})": t['id'] for t in all_tournament_teams}
                    
                    if edit_team_options:
                        edit_team_display = st.selectbox("Select Team to Rename", edit_team_options.keys(), key="edit_team_select")
                        edit_team_id = edit_team_options[edit_team_display]
                        edit_team_old_name = edit_team_display.split(' (')[0]
                        
                        new_team_name = st.text_input("New Team Name", value=edit_team_old_name, key="new_team_name_input")
                        
                        if st.button("Update Team Name", key="update_team_name_btn"):
                            if new_team_name and new_team_name != edit_team_old_name:
                                try:
                                    update_team_name(edit_team_id, new_team_name)
                                    st.success(f"✅ Team renamed from '{edit_team_old_name}' to '{new_team_name}'")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating team name: {e}")
                            else:
                                st.warning("Please enter a different team name")
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
                            
                            # Display team strength
                            if selected_players:
                                team_strength = calculate_team_strength(selected_players, tournament_id)
                                
                                # Color coded strength indicator
                                if team_strength >= 70:
                                    st.success(f"⚡ Team Strength: {team_strength}/100 - STRONG 🟢")
                                elif team_strength >= 50:
                                    st.warning(f"⚡ Team Strength: {team_strength}/100 - MEDIUM 🟡")
                                else:
                                    st.error(f"⚡ Team Strength: {team_strength}/100 - WEAK 🔴")
                            
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
                    st.subheader("🔧 Edit Match Schedule")
                    
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
                        
                        # Date/Time editing moved to Manage Matches tab for consistency
                        
                        # Edit match numbers for same-date matches
                        st.divider()
                        st.write("**Edit Match Number (for multiple matches on same date):**")
                        
                        # Group matches by date to show which have multiple matches
                        date_groups = {}
                        for idx, m in enumerate(group_matches, 1):
                            date = m['match_date']
                            if date not in date_groups:
                                date_groups[date] = []
                            date_groups[date].append({'index': idx, 'match': m, 'id': m['id']})
                        
                        # Show dates with multiple matches
                        multi_match_dates = {date: matches for date, matches in date_groups.items() if len(matches) > 1}
                        
                        if multi_match_dates:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                date_to_edit = st.selectbox(
                                    "Select Date with Multiple Matches",
                                    options=list(multi_match_dates.keys()),
                                    key="select_date_for_numbers"
                                )
                                
                                st.write(f"**Matches on {date_to_edit}:**")
                                for m_info in multi_match_dates[date_to_edit]:
                                    team1 = next((t['team_name'] for t in all_teams if t['id'] == m_info['match']['team1_id']), f"Team {m_info['match']['team1_id']}")
                                    team2 = next((t['team_name'] for t in all_teams if t['id'] == m_info['match']['team2_id']), f"Team {m_info['match']['team2_id']}")
                                    st.write(f"  Match {m_info['index']}: {team1} vs {team2}")
                            
                            with col2:
                                selected_match_for_number = st.selectbox(
                                    "Select Match",
                                    options=[f"Match {m['index']}" for m in multi_match_dates[date_to_edit]],
                                    key="select_match_for_number"
                                )
                                selected_match_number = int(selected_match_for_number.split()[1])
                            
                            with col3:
                                new_match_number = st.number_input(
                                    "New Match #",
                                    min_value=1,
                                    value=selected_match_number,
                                    key="new_match_number"
                                )
                                
                                if st.button("Update Number", key="update_number_btn"):
                                    try:
                                        match_to_update = multi_match_dates[date_to_edit][selected_match_number - 1]
                                        update_match_number(match_to_update['id'], new_match_number)
                                        st.success(f"✅ Match number updated to {new_match_number}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating match number: {e}")
                        else:
                            st.info("ℹ️ No multiple matches on same date yet")
                    else:
                        st.info("No group stage matches generated yet. Click 'Auto-Generate Group Stage Matches' first.")
                        
                    # Custom Single Match Entry
                    st.divider()
                    st.subheader("➕ Add Single Match Manually")
                    sc_col1, sc_col2 = st.columns(2)
                    with sc_col1:
                        m_team1 = st.selectbox("Team 1", [t['team_name'] for t in all_teams], key="man_t1")
                        m_team2 = st.selectbox("Team 2", [t['team_name'] for t in all_teams if t['team_name'] != m_team1], key="man_t2")
                    with sc_col2:
                        m_date = st.date_input("Match Date", key="man_date")
                        m_stage = st.selectbox("Stage", ["group", "super8", "semi-final", "final"], key="man_stage")
                        
                    if st.button("Schedule Manual Match"):
                        t1_id = next(t['id'] for t in all_teams if t['team_name'] == m_team1)
                        t2_id = next(t['id'] for t in all_teams if t['team_name'] == m_team2)
                        create_tournament_match(
                            tournament_id, t1_id, t2_id, 
                            m_date.strftime("%Y-%m-%d"), m_stage
                        )
                        st.success(f"Match Scheduled for {m_date}!")
                        st.rerun()
                    
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
                            
                            match_dict = dict(m)
                            match_data.append({
                                'ID': match_dict['id'],
                                'Team 1': team1,
                                'Team 2': team2,
                                'Date': match_dict['match_date'],
                                'Time': match_dict.get('match_time', '10:00'),
                                'Stage': match_dict['stage'].title(),
                                'Status': match_dict['status'].title(),
                            })
                        
                        st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)
                        
                        # Create a mapping for descriptive selectboxes
                        match_map = {
                            f"{m['Team 1']} vs {m['Team 2']} ({m['Date']} at {m['Time']})": m['ID'] 
                            for m in match_data
                        }
                        
                        st.subheader("Match Control")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Reset Match**")
                            match_display_reset = st.selectbox("Select Match to Reset", list(match_map.keys()), key="reset_match_select")
                            match_to_reset = match_map[match_display_reset]
                            if st.button("🚨 Reset Selected Match Result", type="secondary"):
                                from ..database import reset_match_result
                                success, msg = reset_match_result(match_to_reset)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        with col2:
                            st.write("**Update Schedule (Time/Date)**")
                            match_display_update = st.selectbox("Select Match to Update", list(match_map.keys()), key="update_match_select")
                            match_to_update = match_map[match_display_update]
                            sel_m = next(m for m in filtered_matches if m['id'] == match_to_update)
                            sel_m_dict = dict(sel_m)
                            
                            new_date = st.date_input("New Date", value=datetime.strptime(sel_m_dict['match_date'], "%Y-%m-%d").date(), key="manage_date")
                            new_time = st.time_input("New Time", value=datetime.strptime(sel_m_dict.get('match_time', '10:00'), "%H:%M").time(), key="manage_time")
                            
                            if st.button("Update Match Schedule"):
                                from ..database import update_match_date, update_match_time
                                try:
                                    update_match_date(match_to_update, new_date.strftime("%Y-%m-%d"))
                                    update_match_time(match_to_update, new_time.strftime("%H:%M"))
                                    st.success("✅ Match schedule updated!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            
                            st.write("---")
                            st.write("**Delete Match**")
                            if st.button("🗑️ Delete Selected Match", type="primary", key="del_m_btn"):
                                success, msg = delete_match(match_to_update)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                    else:
                        st.info("No matches found with selected filters")
                else:
                    st.warning("No matches found. Create matches in 'Schedule Matches' tab")
            else:
                st.error("Tournament not found")
    
    # ========== TAB 6: UPDATE SCORES ==========
    with tab6:
        st.header("Update Match Scores & Player Performance")
        
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
                    
                    # --- TIME CHECK ---
                    m_dict = dict(match)
                    match_dt_str = f"{m_dict['match_date']} {m_dict.get('match_time', '10:00')}"
                    match_dt = datetime.strptime(match_dt_str, "%Y-%m-%d %H:%M")
                    
                    if datetime.now() < match_dt:
                        st.warning(f"⏳ **Match Result Entry Locked**")
                        st.info(f"This match is scheduled for **{match_dt_str}**. Results can only be entered after the match starts.")
                        st.stop()
                    
                    # Step 1: Update Match Score
                    st.subheader("Step 1: Match Result")
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
                    
                    st.divider()
                    
                    # Step 2: Add Player Performance
                    st.subheader("Step 2: Player Performance Tracking")
                    
                    # Get completed matches for performance entry
                    completed_matches = [m for m in matches if m['status'] == 'completed']
                    
                    if completed_matches:
                        perf_match_options = {}
                        for m in completed_matches:
                            team1 = next((t['team_name'] for t in all_teams if t['id'] == m['team1_id']), f"Team {m['team1_id']}")
                            team2 = next((t['team_name'] for t in all_teams if t['id'] == m['team2_id']), f"Team {m['team2_id']}")
                            perf_match_options[f"{team1} vs {team2} ({m['match_date']})"] = m['id']
                        
                        perf_selected_match = st.selectbox("Select Match for Performance Entry", perf_match_options.keys(), key="perf_select_match")
                        perf_match_id = perf_match_options[perf_selected_match]
                        
                        perf_match = next(m for m in completed_matches if m['id'] == perf_match_id)
                        perf_team1_name = next((t['team_name'] for t in all_teams if t['id'] == perf_match['team1_id']), f"Team {perf_match['team1_id']}")
                        perf_team2_name = next((t['team_name'] for t in all_teams if t['id'] == perf_match['team2_id']), f"Team {perf_match['team2_id']}")
                        
                        # Display existing performances
                        existing_perfs = get_match_performances(perf_match_id)
                        if existing_perfs:
                            st.info(f"📊 Already recorded performances for this match:")
                            perf_df = pd.DataFrame([{
                                'Player': p['player_name'],
                                'Team': perf_team1_name if p['team_id'] == perf_match['team1_id'] else perf_team2_name,
                                'Runs': p['runs'],
                                'Balls': p['balls_faced'],
                                'Fours': p['fours'],
                                'Sixes': p['sixes'],
                                'Wickets': p['wickets']
                            } for p in existing_perfs])
                            st.dataframe(perf_df, use_container_width=True)
                        
                            st.subheader("Add New Player Performance")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                perf_team = st.selectbox("Select Team", [perf_team1_name, perf_team2_name], key="perf_team_select")
                                perf_team_id = perf_match['team1_id'] if perf_team == perf_team1_name else perf_match['team2_id']
                            
                            with col2:
                                # Get team players
                                team_details = get_team_details(perf_team_id)
                                team_players = team_details['squad'].split(',') if team_details and team_details['squad'] else []
                                
                                if team_players:
                                    # Multiselect for players who played
                                    playing_players = st.multiselect(
                                        "Select Playing 11 (Exactly 11 required)", 
                                        team_players, 
                                        default=get_playing_xi(perf_match_id, perf_team_id),
                                        key="perf_playing_select",
                                        help="Finalize your 11 players for this match before entering scores"
                                    )
                                    
                                    if st.button("📢 Save/Finalize Playing 11", key=f"save_xi_{perf_team_id}"):
                                        if len(playing_players) > 0:
                                            success, msg = save_playing_xi(perf_match_id, perf_team_id, playing_players)
                                            if success:
                                                st.success(f"✅ {perf_team} Playing 11 saved!")
                                            else:
                                                st.error(msg)
                                        else:
                                            st.warning("Please select at least one player.")

                                    if len(playing_players) != 11 and len(playing_players) > 0:
                                        st.warning(f"⚠️ Selected {len(playing_players)} players. Pro Tip: International matches usually have exactly 11.")
                                else:
                                    st.error("No players found for this team")
                                    playing_players = []
                            
                            if playing_players:
                                # We'll use a data editor or dynamic fields for multiple players
                                st.write("📝 **Enter Match Performance Details**")
                                
                                for p_name in playing_players:
                                    with st.expander(f"👤 {p_name} - Stats", expanded=True):
                                        p_col1, p_col2, p_col3 = st.columns(3)
                                        
                                        with p_col1:
                                            st.write("**Batting**")
                                            p_runs = st.number_input("Runs", min_value=0, step=1, key=f"runs_{p_name}")
                                            p_balls = st.number_input("Balls Faced", min_value=0, step=1, key=f"balls_{p_name}")
                                            p_not_out = st.checkbox("Not Out", key=f"no_{p_name}")
                                            # Simple SR display
                                            if p_balls > 0:
                                                p_sr = (p_runs / p_balls) * 100
                                                st.caption(f"SR: {p_sr:.2f}")
                                            
                                        with p_col2:
                                            st.write("**Boundaries**")
                                            p_fours = st.number_input("4s", min_value=0, step=1, key=f"4s_{p_name}")
                                            p_sixes = st.number_input("6s", min_value=0, step=1, key=f"6s_{p_name}")
                                            
                                        with p_col3:
                                            st.write("**Bowling**")
                                            p_overs = st.number_input("Overs", min_value=0.0, max_value=4.0, step=0.1, key=f"overs_{p_name}")
                                            p_cons = st.number_input("Runs Conceded", min_value=0, step=1, key=f"cons_{p_name}")
                                            p_wkts = st.number_input("Wickets", min_value=0, step=1, key=f"wkts_{p_name}")
                                            # Simple Econ display
                                            if p_overs > 0:
                                                p_econ = p_cons / p_overs
                                                st.caption(f"Econ: {p_econ:.2f}")
                                        
                                        if st.button(f"Save {p_name}", key=f"save_{p_name}"):
                                            try:
                                                # 1. Save to Database for match stats and fantasy
                                                add_player_performance(
                                                    perf_match_id, p_name, perf_team_id,
                                                    p_runs, p_balls, p_fours, p_sixes,
                                                    p_wkts, p_overs, p_cons, p_not_out
                                                )
                                                
                                                # 2. Synchronize to wc_players.csv
                                                from ..database import update_wc_csv_stats
                                                match_stats = {
                                                    'runs': p_runs,
                                                    'balls': p_balls,
                                                    'fours': p_fours,
                                                    'sixes': p_sixes,
                                                    'wickets': p_wkts,
                                                    'overs': p_overs,
                                                    'runs_conceded': p_cons,
                                                    'is_not_out': p_not_out
                                                }
                                                csv_success, csv_msg = update_wc_csv_stats(p_name, match_stats)
                                                
                                                if csv_success:
                                                    st.success(f"✅ Saved for {p_name} and updated CSV!")
                                                else:
                                                    st.warning(f"✅ Saved for {p_name}, but CSV update note: {csv_msg}")
                                                    
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                        
                        # Recalculate fantasy scores
                        if st.button("🔄 Recalculate Fantasy Points (After all performances recorded)", key="recalc_fantasy"):
                            try:
                                calculate_updated_fantasy_scores(tournament_id)
                                st.success("✅ Fantasy points recalculated based on performance data!")
                                st.balloons()
                            except Exception as e:
                                st.error(f"Error recalculating fantasy points: {e}")
                    else:
                        st.info("No completed matches yet. Complete some matches first to record performances.")
                else:
                    st.info("✅ All matches completed!")
            else:
                st.error("Tournament not found")

    # ========== TAB 7: SUPER 8 & STAGES ==========
    with tab7:
        st.header("🏆 Stage Progression Management")
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="stage_man_id")
        
        if tournament_id:
            tournament = get_tournament(tournament_id)
            if tournament:
                st.subheader("Promote to Super 8")
                st.write("This will take the Top 2 teams from each of the 4 groups (A, B, C, D) based on Points and NRR, then schedule the Super 8 round-robin matches.")
                
                if st.button("✨ Advance to Super 8", type="primary"):
                    success, msg = promote_to_super8(tournament_id)
                    if success:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(f"Cannot Advance: {msg}")
                
                st.divider()
                st.subheader("Knockout Progression")
                st.info("Once Super 8 is complete, manually schedule the Semi-Finals using the 'Schedule Matches' tab.")

    # ========== TAB 8: TOURNAMENT STATS ==========
    with tab8:
        st.header("📊 Live Tournament Statistics")
        tournament_id = st.number_input("Tournament ID", min_value=1, step=1, key="stats_man_id")
        
        if tournament_id:
            s_tab1, s_tab2, s_tab3 = st.tabs(["Top Scorers", "Top Bowlers", "Most Sixes"])
            
            with s_tab1:
                df = get_tournament_stats(tournament_id, 'runs')
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No batting data recorded yet.")
            
            with s_tab2:
                df = get_tournament_stats(tournament_id, 'wickets')
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No bowling data recorded yet.")
                    
            with s_tab3:
                df = get_tournament_stats(tournament_id, 'sixes')
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No sixes data recorded yet.")
    
    # ========== TEAM STRENGTH DISPLAY ==========
    st.divider()
    st.subheader("⚡ AI Team Strength Analysis")
    
    strength_tournament_id = st.number_input("Tournament ID for Team Strength", min_value=1, step=1, key="strength_tournament")
    
    if strength_tournament_id:
        strength_tournament = get_tournament(strength_tournament_id)
        if strength_tournament:
            strength_teams = get_tournament_teams(strength_tournament_id)
            
            if strength_teams:
                st.write(f"**{strength_tournament['name']}** - Team Strength Ratings")
                
                strength_data = []
                for team in strength_teams:
                    strength = get_team_strength_rating(strength_tournament_id, team['id'])
                    strength_data.append({
                        'Team': team['team_name'],
                        'Group': team['group_letter'],
                        'Players': len(team['squad'].split(',')) if team['squad'] else 0,
                        'Strength': strength,
                        'Rating': '🟢 Strong' if strength >= 70 else '🟡 Medium' if strength >= 50 else '🔴 Weak'
                    })
                
                strength_df = pd.DataFrame(strength_data).sort_values('Strength', ascending=False)
                st.dataframe(strength_df, use_container_width=True)
                
                # Visual representation
                st.bar_chart(strength_df.set_index('Team')['Strength'])
            else:
                st.info("No teams found in this tournament")
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
    
    # ========== TAB 9: PLAYER MASTER CONTROL ==========
    with tab9:
        st.header("🛠️ Global Player Master Control")
        from .player_management import render_player_management
        render_player_management()
        
        st.divider()
        st.subheader("➕ Global Database Additions")
        with st.expander("Register New Professional Player"):
            new_p_name = st.text_input("Full Player Name")
            new_p_team = st.text_input("National Side")
            new_p_role = st.selectbox("Playing Role", ["Batsman", "Bowler", "All-rounder", "Wicket-keeper"], key="master_role")
            new_p_fmt = st.selectbox("Primary Format", ["T20", "ODI", "Test"], key="master_fmt")
            
            if st.button("Commit to Master List", type="primary"):
                try:
                    conn = get_db_connection()
                    conn.execute("""
                        INSERT INTO players (player, team, format, role, runs, wickets, average, strike_rate, matches, innings)
                        VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 0)
                    """, (new_p_name, new_p_team, new_p_fmt, new_p_role))
                    conn.commit()
                    st.success(f"Successfully added {new_p_name} to world database!")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    conn.close()

if __name__ == "__main__":
    show_admin_panel()

