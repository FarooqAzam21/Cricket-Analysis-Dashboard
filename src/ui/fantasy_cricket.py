import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import (
    get_tournament, get_tournament_matches, get_tournament_teams,
    save_fantasy_team, get_user_fantasy_teams, get_db_connection, 
    fetch_all_players_from_db, get_match_playing_xi
)

def get_upcoming_and_scheduled_matches(tournament_id):
    """Get all scheduled/upcoming matches (not completed)"""
    matches = get_tournament_matches(tournament_id)
    return [m for m in matches if m['status'] != 'completed']

def safe_float(value, default=0):
    """Safely convert value to float, handling '-' and empty strings"""
    try:
        if value is None or value == '' or value == '-':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def clean_name(name):
    """Clean player name from JSON artifacts like quotes and brackets"""
    if not name: return ""
    name = str(name).strip()
    # Remove JSON artifacts if they exist
    for char in ['[', ']', '"', "'"]:
        name = name.replace(char, '')
    return name.strip()

def load_role_based_players():
    """Load players from role-specific CSV files with stats and pricing"""
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    role_map = {}
    stats_map = {}  # Store player stats for valuation
    
    files_to_load = [
        ('odi_batsman.csv', 'Batsman'),
        ('odi_all_rounders.csv', 'All-rounder'),
        ('odi_bowler.csv', 'Bowler')
    ]
    
    for filename, default_role in files_to_load:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Process all formats but prioritize T20 for stats
            # We sort by format so T20 entries come last and overwrite Odi if both exist
            if 'Format' in df.columns:
                df['format_lower'] = df['Format'].str.lower().str.strip()
                # Sort so 't20' comes last
                df = df.sort_values(by='format_lower', ascending=True) 

            for _, row in df.iterrows():
                player_name = clean_name(row.get('player', ''))
                if not player_name: continue
                
                # Determine role
                role = default_role
                if filename == 'odi_batsman.csv':
                    csv_role = str(row.get('role', '')).lower()
                    if 'keeper' in csv_role or 'wicket' in csv_role:
                        role = 'Wicket-keeper'
                
                # Update maps
                # If player already exists in role_map, we only update if this row is T20
                if player_name not in role_map or row.get('format_lower') == 't20':
                    role_map[player_name] = role
                    stats_map[player_name] = {
                        'batting_avg': safe_float(row.get('average', 0)),
                        'strike_rate': safe_float(row.get('strike_rate', 0)),
                        'bowling_avg': safe_float(row.get('bowling_average', row.get('average', 0)) if role == 'Bowler' else row.get('bowling_average', 0)),
                        'image': row.get('image_url', '')
                    }
    
    return role_map, stats_map

def calculate_player_price(player_name, role, stats_map):
    """
    Calculate player price based on their stats
    Tiers: 300K (Premium), 200K (Good), 100K (Average), 50K (Budget)
    """
    stats = stats_map.get(player_name, {'batting_avg': 0, 'strike_rate': 0, 'bowling_avg': 0})
    
    bat_avg = stats.get('batting_avg', 0)
    strike_rate = stats.get('strike_rate', 0)
    bowl_avg = stats.get('bowling_avg', 0)
    
    # Scoring algorithm based on role
    score = 0
    
    if role in ['Batsman', 'Wicket-keeper']:
        # For batsmen: 70% batting avg, 30% strike rate
        score = (bat_avg * 0.8) + (strike_rate / 10 * 0.3)
    elif role == 'Bowler':
        # For bowlers: Lower avg is better (inverse scoring)
        if bowl_avg > 0:
            score = max(0, 100 - bowl_avg)*0.98  # Lower average = higher score
        else:
            score = 20  # Default for missing stats
    elif role == 'All-rounder':
        # Balanced scoring for all-rounders
        batting_score = bat_avg * 0.4
        bowl_score = max(0, 50 - bowl_avg) * 0.4 if bowl_avg > 0 else 10
        sr_score = strike_rate / 10 * 0.2
        score = batting_score + bowl_score + sr_score
    else:
        score = 25  # Default
    
    # Categorize into price tiers
    if score >= 80:
        return 300_000  # Premium: 300K
    elif score >= 40:
        return 200_000  # Good: 200K
    elif score >= 25:
        return 100_000  # Average: 100K
    else:
        return 50_000   # Budget: 50K

def get_match_squads_curated(match_id):
    """Get only the players announced in Playing 11 from admin panel with pricing"""
    # Load role mappings and stats
    role_map, stats_map = load_role_based_players()
    
    # Get match and team details
    conn = get_db_connection()
    match = conn.execute("SELECT team1_id, team2_id, tournament_id FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
    
    if not match:
        conn.close()
        return pd.DataFrame(), "Team 1", "Team 2", False
    
    # Get team details with squads
    team1 = conn.execute("SELECT * FROM tournament_teams WHERE id = ?", (match['team1_id'],)).fetchone()
    team2 = conn.execute("SELECT * FROM tournament_teams WHERE id = ?", (match['team2_id'],)).fetchone()
    conn.close()
    
    team1_name = team1['team_name'] if team1 else "Team 1"
    team2_name = team2['team_name'] if team2 else "Team 2"
    
    # Check if Playing 11 has been announced
    playing_xi_names = get_match_playing_xi(match_id)
    
    if playing_xi_names:
        # Use only the announced Playing 11
        players_data = []
        for raw_name in playing_xi_names:
            player_name = clean_name(raw_name)
            if not player_name: continue
            
            # Determine which team this player belongs to
            team1_squad = [clean_name(p) for p in (team1['squad'].split(',') if team1 and team1['squad'] else [])]
            team2_squad = [clean_name(p) for p in (team2['squad'].split(',') if team2 and team2['squad'] else [])]
            
            if player_name in team1_squad:
                team = team1_name
            elif player_name in team2_squad:
                team = team2_name
            else:
                continue  # Skip if player not in either squad
            
            # Get role from role_map
            role = role_map.get(player_name, 'All-rounder')
            price = calculate_player_price(player_name, role, stats_map)
            player_stats = stats_map.get(player_name, {})
            
            players_data.append({
                'player': player_name,
                'team': team,
                'role': role,
                'price': price,
                'format': 'T20',
                'image': player_stats.get('image', ''),
                'batting_avg': player_stats.get('batting_avg', 0),
                'strike_rate': player_stats.get('strike_rate', 0),
                'bowling_avg': player_stats.get('bowling_avg', 0)
            })
        
        squad_df = pd.DataFrame(players_data)
        return squad_df, team1_name, team2_name, True
    else:
        # No Playing 11 announced - use full squads from tournament teams
        players_data = []
        
        # Team 1 squad
        if team1 and team1['squad']:
            team1_squad = [clean_name(p) for p in team1['squad'].split(',')]
            
            for player_name in team1_squad:
                if not player_name: continue
                role = role_map.get(player_name, 'All-rounder')
                price = calculate_player_price(player_name, role, stats_map)
                player_stats = stats_map.get(player_name, {})
                players_data.append({
                    'player': player_name,
                    'team': team1_name,
                    'role': role,
                    'price': price,
                    'format': 'T20',
                    'image': player_stats.get('image', ''),
                    'batting_avg': player_stats.get('batting_avg', 0),
                    'strike_rate': player_stats.get('strike_rate', 0),
                    'bowling_avg': player_stats.get('bowling_avg', 0)
                })
        
        # Team 2 squad
        if team2 and team2['squad']:
            team2_squad = [clean_name(p) for p in team2['squad'].split(',')]
            
            for player_name in team2_squad:
                if not player_name: continue
                role = role_map.get(player_name, 'All-rounder')
                price = calculate_player_price(player_name, role, stats_map)
                player_stats = stats_map.get(player_name, {})
                players_data.append({
                    'player': player_name,
                    'team': team2_name,
                    'role': role,
                    'price': price,
                    'format': 'T20',
                    'image': player_stats.get('image', ''),
                    'batting_avg': player_stats.get('batting_avg', 0),
                    'strike_rate': player_stats.get('strike_rate', 0),
                    'bowling_avg': player_stats.get('bowling_avg', 0)
                })
        
        squad_df = pd.DataFrame(players_data)
        return squad_df, team1_name, team2_name, False

def get_match_squads_full(match_id):
    """Legacy fallback - not needed anymore but kept for compatibility"""
    return get_match_squads_curated(match_id)[:3]

def show_fantasy_cricket():
    """Fantasy cricket team builder interface with improved design"""
    
    # Custom CSS for horizontal match list
    st.markdown("""
        <style>
        .match-card {
            background: #1a1c24;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border: 1px solid #333;
            transition: all 0.3s ease;
            color: #e2e8f0 !important;
        }
        .match-card:hover {
            border-color: #2ecc71;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
        }
        .match-team { font-weight: bold; font-size: 1.1rem; color: #e2e8f0; }
        .match-vs { color: #f39c12; font-weight: bold; margin: 5px 0; }
        .match-date { font-size: 0.75rem; color: #95a5a6; margin-top: 5px; }
        /* Custom button styling within cards */
        div.stButton > button.create-team-btn {
            width: 100%;
            background-color: #2ecc71 !important;
            color: #e2e8f0 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-weight: bold !important;
            margin-top: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏏 Fantasy Cricket League")
    
    # Get active tournaments
    conn = get_db_connection()
    tournaments = conn.execute("SELECT * FROM tournaments ORDER BY id DESC").fetchall()
    conn.close()
    
    if not tournaments:
        st.info("No tournaments available")
        return
    
    # Select tournament (simplified)
    t_id = tournaments[0]['id'] if len(tournaments) > 0 else None
    if not t_id: return

    # Get upcoming matches
    upcoming_matches = get_upcoming_and_scheduled_matches(t_id)
    all_teams = get_tournament_teams(t_id)
    
    st.write("### 📅 Upcoming Fixtures")
    
    # Horizontal Match Selection Area
    if upcoming_matches:
        cols = st.columns(len(upcoming_matches) if len(upcoming_matches) < 5 else 5)
        
        # Track selected match in session state
        if 'selected_match_id' not in st.session_state:
            st.session_state.selected_match_id = None

        for idx, match in enumerate(upcoming_matches[:5]): # Show up to 5 horizontally
            with cols[idx]:
                t1 = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), "T1")
                t2 = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), "T2")
                
                # Manual HTML-ish layout for the box
                # Unified Match Card with Button
                st.markdown(f"""
                    <div class="match-card">
                        <div class="match-team">{t1}</div>
                        <div class="match-vs">VS</div>
                        <div class="match-team">{t2}</div>
                        <div class="match-date">📅 {match['match_date']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("CREATE TEAM ➜", key=f"btn_{match['id']}", width="stretch"):
                    st.session_state.selected_match_id = match['id']
                    st.rerun()

    if st.session_state.selected_match_id:
        match_id = st.session_state.selected_match_id
        selected_match = next((m for m in upcoming_matches if m['id'] == match_id), None)
        
        if not selected_match:
            st.error("Match not found")
            return

        st.divider()
        st.write(f"## 🛠️ Team Builder: {selected_match['match_date']}")
        
        # Get squads (Filtered by Admin's Playing 11)
        res = get_match_squads_curated(match_id)
        if len(res) == 4:
            squad_df, team1_name, team2_name, is_announced = res
        else:
            squad_df, team1_name, team2_name = res
            is_announced = False

        if not is_announced:
            st.markdown("""
                <div style="background: rgba(243, 156, 18, 0.1); border: 1px solid #f39c12; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
                    <span style="color: #f39c12; font-weight: bold;">⚠️ Squad Unofficial</span>
                    <p style="color: #dcdde1; font-size: 0.9rem; margin-top: 5px; margin-bottom: 0;">
                        Official Playing 11 contains all squad players. Final lineups may vary after the toss.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
                    <span style="color: #2ecc71; font-weight: bold;">✅ Lineup Confirmed</span>
                    <p style="color: #dcdde1; font-size: 0.9rem; margin-top: 5px; margin-bottom: 0;">
                        Showing the official 22 players announced in Admin Panel.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Build UI
        if f"selected_players_{match_id}" not in st.session_state:
            st.session_state[f"selected_players_{match_id}"] = []
        
        selected_players = st.session_state[f"selected_players_{match_id}"]

        # Budget Display
        TOTAL_BUDGET = 1_000_000
        sel_df_temp = squad_df[squad_df['player'].isin(selected_players)]
        spent_budget = sel_df_temp['price'].sum() if not sel_df_temp.empty else 0
        remaining = TOTAL_BUDGET - spent_budget
        
        # Team distribution
        team_counts = sel_df_temp['team'].value_counts().to_dict() if not sel_df_temp.empty else {}
        
        # Budget bar
        budget_pct = (spent_budget / TOTAL_BUDGET) * 100
        budget_color = "#2ecc71" if budget_pct <= 100 else "#e74c3c"
        
        st.markdown(f"""
        <div style="background: #1a1c24; padding: 20px; border-radius: 15px; border-left: 5px solid #2ecc71; margin-bottom: 15px;">
            <h4 style="margin-top: 0; color: #e2e8f0;">🛡️ Selection Rules</h4>
            <ul style="font-size: 0.9rem; color: #e0e0e0; margin: 0; padding-left: 20px;">
                <li>Select 11 players (2+ Batsmen, 2+ Bowlers, 1+ Wicket-keeper)</li>
                <li>Budget: <b>1M</b> total (Player prices: 300K/200K/100K/50K)</li>
                <li>Maximum <b>8 players</b> from the same team</li>
            </ul>
        </div>
        
        <div style="background: #1a1c24; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: #e2e8f0; font-weight: bold;">💰 Budget</span>
                <span style="color: {budget_color}; font-weight: bold;">{spent_budget:,} / {TOTAL_BUDGET:,}</span>
            </div>
            <div style="background: #34495e; border-radius: 10px; height: 20px; overflow: hidden;">
                <div style="background: {budget_color}; height: 100%; width: {min(budget_pct, 100)}%; transition: all 0.3s;"></div>
            </div>
            <div style="color: #95a5a6; font-size: 0.8rem; margin-top: 5px;">Remaining: {remaining:,}</div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for Roles
        tab_bat, tab_bowl, tab_ar, tab_wk = st.tabs(["🏏 Batsmen", "🎳 Bowlers", "🛡️ All-rounders", "🧤 Wicket-keepers"])
        
        def render_compact_selection(role_filter, player_df):
            c1, c2 = st.columns(2)
            teams = [team1_name, team2_name]
            role_players = player_df[player_df['role'].str.contains(role_filter, case=False, na=False)]
            
            for i, team in enumerate(teams):
                with [c1, c2][i]:
                    st.markdown(f"<h5 style='color: #e2e8f0;'>{team}</h5>", unsafe_allow_html=True)
                    team_role_players = role_players[role_players['team'] == team]
                    if team_role_players.empty:
                        st.caption("No players in this role")
                    for _, p in team_role_players.iterrows():
                        p_name = p['player']
                        p_price = p['price']
                        p_team = p['team']
                        sel = p_name in selected_players
                        
                        # Format price
                        price_display = f"{int(p_price/1000)}K"
                        
                        # Check constraints before allowing selection
                        can_select = True
                        error_msg = ""
                        
                        if not sel:
                            # Budget check
                            temp_df = squad_df[squad_df['player'].isin(selected_players + [p_name])]
                            temp_spent = temp_df['price'].sum()
                            if temp_spent > TOTAL_BUDGET:
                                can_select = False
                                error_msg = "💸 Over budget!"
                            
                            # Team limit check (max 8 from same team)
                            temp_team_counts = temp_df['team'].value_counts().to_dict()
                            if temp_team_counts.get(p_team, 0) > 8:
                                can_select = False
                                error_msg = "⚠️ Max 8 from same team!"
                            
                            # Squad size check
                            if len(selected_players) >= 11:
                                can_select = False
                                error_msg = "11 players max!"
                        
                        # Button label with price
                        if sel:
                            label = f"✅ {p_name} ({price_display})"
                        else:
                            label = f"➕ {p_name} ({price_display})"
                        
                        if st.button(label, key=f"fsel_{p_name}_{match_id}", disabled=(not can_select and not sel)):
                            if sel:
                                selected_players.remove(p_name)
                            elif can_select:
                                selected_players.append(p_name)
                            elif error_msg:
                                st.error(error_msg)
                            st.rerun()

        with tab_bat: render_compact_selection('Batsman', squad_df)
        with tab_bowl: render_compact_selection('Bowler', squad_df)
        with tab_ar: render_compact_selection('All-rounder', squad_df)
        with tab_wk: render_compact_selection('Wicket-keeper', squad_df)

        # Validation & Submit
        st.divider()
        sel_df = squad_df[squad_df['player'].isin(selected_players)]
        bat_c = len(sel_df[sel_df['role'].str.contains('Batsman', case=False)])
        bowl_c = len(sel_df[sel_df['role'].str.contains('Bowler', case=False)])
        wk_c = len(sel_df[sel_df['role'].str.contains('Wicket-keeper', case=False)])
        
        # Team distribution
        team_distribution = sel_df['team'].value_counts().to_dict() if not sel_df.empty else {}
        max_from_team = max(team_distribution.values()) if team_distribution else 0
        
        # Metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Batsmen", f"{bat_c}/2+", delta=bat_c-2 if bat_c>=2 else bat_c-2)
        col2.metric("Bowlers", f"{bowl_c}/2+", delta=bowl_c-2 if bowl_c>=2 else bowl_c-2)
        col3.metric("WKs", f"{wk_c}/1+", delta=wk_c-1 if wk_c>=1 else wk_c-1)
        col4.metric("Squad", f"{len(selected_players)}/11")
        col5.metric("Max Team", f"{max_from_team}/8", delta=0 if max_from_team <= 8 else max_from_team-8)
        
        # Show team distribution if players selected
        if selected_players and team_distribution:
            st.markdown("<p style='color: #95a5a6; font-size: 0.85rem; margin-top: 10px;'>Team Distribution:</p>", unsafe_allow_html=True)
            dist_cols = st.columns(len(team_distribution))
            for idx, (team, count) in enumerate(team_distribution.items()):
                with dist_cols[idx]:
                    color = "#e74c3c" if count > 8 else "#2ecc71"
                    st.markdown(f"<div style='text-align: center; color: {color}; font-size: 0.9rem;'><b>{team}</b>: {count}</div>", unsafe_allow_html=True)

        is_valid = (bat_c>=2 and bowl_c>=2 and wk_c>=1 and len(selected_players)==11 and 
                   spent_budget <= TOTAL_BUDGET and max_from_team <= 8)

        # Captain/Vice-Captain Selection (always show if team has players)
        if len(selected_players) > 0:
            st.divider()
            st.markdown("### 👑 Leadership Selection")
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                cap = st.selectbox("Captain (2x)", selected_players, key=f"c_{match_id}")
            with c_col2:
                if len(selected_players) > 1:
                    vcap = st.selectbox("Vice-Captain (1.5x)", [p for p in selected_players if p != cap], key=f"vc_{match_id}")
                else:
                    vcap = cap
                    st.caption("Need at least 2 players for vice-captain")
            
            # Save button (always visible, but disabled if invalid)
            save_btn_label = "🚀 Submit Fantasy Squad" if is_valid else "⚠️ Complete Your Squad"
            save_disabled = not is_valid
            
            if st.button(save_btn_label, type="primary", disabled=save_disabled, width="stretch"):
                try:
                    conn = get_db_connection()
                    user = conn.execute("SELECT id FROM users WHERE username = ?", (st.session_state.username,)).fetchone()
                    if user:
                        players_json = json.dumps({'players': selected_players, 'captain': cap, 'vice_captain': vcap})
                        save_fantasy_team(user['id'], t_id, match_id, players_json)
                        st.success("✅ Team saved for the match!")
                        st.balloons()
                    conn.close()
                except Exception as e: st.error(f"Error: {e}")
        
        if is_valid:
            st.success("✨ Your team is balanced and ready to submit!")
        elif len(selected_players) > 0:
            st.info("💡 Keep picking until you satisfy all requirements!")
            if selected_players:
                # Professional team display instead of array format
                st.markdown("<h4 style='color: #e2e8f0; margin-top: 20px;'>Current Lineup</h4>", unsafe_allow_html=True)
                
                # Group by roles for display
                lineup_df = squad_df[squad_df['player'].isin(selected_players)]
                
                roles_order = ['Batsman', 'All-rounder', 'Bowler', 'Wicket-keeper']
                for role in roles_order:
                    role_players = lineup_df[lineup_df['role'].str.contains(role, case=False, na=False)]
                    if not role_players.empty:
                        st.markdown(f"<p style='color: #f39c12; font-weight: bold; margin: 10px 0 5px 0;'>{role}s</p>", unsafe_allow_html=True)
                        cols = st.columns(min(len(role_players), 4))
                        for idx, (_, player) in enumerate(role_players.iterrows()):
                            with cols[idx % 4]:
                                st.markdown(f"""
                                <div style='background: #2c3e50; padding: 12px; border-radius: 10px; margin: 5px 0; text-align: center; border: 2px solid #34495e;'>
                                    <div style='font-size: 0.95rem; font-weight: bold; color: #e2e8f0; margin-bottom: 4px;'>{player['player']}</div>
                                    <div style='font-size: 0.75rem; color: #3498db; margin-bottom: 6px;'>{player['team']}</div>
                                    <div style='font-size: 0.7rem; color: #2ecc71; font-weight: 600;'>{int(player['price']/1000)}K</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                if st.button("🗑️ Clear All", key="clear_lineup"):
                    st.session_state[f"selected_players_{match_id}"] = []

    # Previous Teams Section (Persistent)
    st.divider()
    st.subheader("📜 Your Saved Teams")
    try:
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username = ?", (st.session_state.username,)).fetchone()
        conn.close()
        if user:
            prev = get_user_fantasy_teams(user['id'], t_id)
            if prev:
                for team in prev:
                    match_info = next((m for m in upcoming_matches if m['id'] == team['match_id']), None)
                    date_str = match_info['match_date'] if match_info else "Unknown Date"
                    with st.expander(f"Match ID: {team['match_id']} | Date: {date_str}"):
                        data = json.loads(team['players_json'])
                        st.write(f"**Players:** {', '.join(data['players'])}")
                        st.write(f"**C/VC:** {data['captain']} (C), {data['vice_captain']} (VC)")
            else:
                st.caption("No teams saved yet.")
    except Exception as e: st.error(str(e))

if __name__ == "__main__":
    show_fantasy_cricket()
