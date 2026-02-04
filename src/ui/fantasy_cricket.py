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

def get_match_squads_curated(match_id):
    """Get only the players announced in Playing 11 from admin panel"""
    all_players = fetch_all_players_from_db()
    playing_xi_names = get_match_playing_xi(match_id)
    
    if not playing_xi_names:
        # Fallback to full squads if no playing 11 announced yet
        # But we should probably warn the user
        return get_match_squads_full(match_id), False
    
    # Filter global players list by these names
    squad = all_players[all_players['player'].isin(playing_xi_names)].copy()
    
    # Get team names for header
    conn = get_db_connection()
    match = conn.execute("SELECT team1_id, team2_id, tournament_id FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
    conn.close()
    
    all_teams = get_tournament_teams(match['tournament_id'])
    team1_name = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), "Team 1")
    team2_name = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), "Team 2")
    
    return squad, team1_name, team2_name, True

def get_match_squads_full(match_id):
    """Fallback: Get all players from both teams"""
    all_players = fetch_all_players_from_db()
    conn = get_db_connection()
    match = conn.execute("SELECT team1_id, team2_id, tournament_id FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
    conn.close()
    
    all_teams = get_tournament_teams(match['tournament_id'])
    team1_name = next((t['team_name'] for t in all_teams if t['id'] == match['team1_id']), None)
    team2_name = next((t['team_name'] for t in all_teams if t['id'] == match['team2_id']), None)
    
    squad = all_players[
        ((all_players['team'] == team1_name) | (all_players['team'] == team2_name)) &
        (all_players['format'] == 'T20')
    ].copy()
    return squad, team1_name, team2_name

def show_fantasy_cricket():
    """Fantasy cricket team builder interface with improved design"""
    
    # Custom CSS for horizontal match list
    st.markdown("""
        <style>
        .match-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.2s;
            min-width: 200px;
        }
        .match-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.1);
        }
        .match-team { font-weight: bold; font-size: 1.1rem; }
        .match-vs { color: #f39c12; font-weight: bold; margin: 5px 0; }
        .match-info { font-size: 0.8rem; opacity: 0.7; }
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
                st.markdown(f"""
                    <div style="text-align: center; padding: 10px; border: 1px solid #333; border-radius: 10px; background: #0e1117; margin-bottom: 10px;">
                        <div style="font-size: 0.9rem; font-weight: bold;">{t1}</div>
                        <div style="color: #f39c12; font-weight: bold; margin: 2px 0;">VS</div>
                        <div style="font-size: 0.9rem; font-weight: bold;">{t2}</div>
                        <div style="font-size: 0.7rem; opacity: 0.6; margin-top: 5px;">{match['match_date']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("Create Team", key=f"btn_{match['id']}"):
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
            st.warning("⚠️ Official Playing 11 contains all squad players. Final lineups may vary.")
        else:
            st.success(f"✅ Showing the official 22 players announced in Admin Panel.")

        # Build UI
        if f"selected_players_{match_id}" not in st.session_state:
            st.session_state[f"selected_players_{match_id}"] = []
        
        selected_players = st.session_state[f"selected_players_{match_id}"]

        st.markdown(f"""
        <div style="background: #1a1c24; padding: 20px; border-radius: 15px; border-left: 5px solid #2ecc71; margin-bottom: 25px;">
            <h4 style="margin-top: 0;">🛡️ Selection Strategy</h4>
            <p style="font-size: 0.9rem; opacity: 0.8;">Select 11 players. Must include 2+ Batsmen, 2+ Bowlers, and 1+ Wicket-keeper.</p>
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
                    st.markdown(f"##### {team}")
                    team_role_players = role_players[role_players['team'] == team]
                    if team_role_players.empty:
                        st.caption("No players in this role")
                    for _, p in team_role_players.iterrows():
                        p_name = p['player']
                        sel = p_name in selected_players
                        label = f"✅ {p_name}" if sel else f"➕ {p_name}"
                        if st.button(label, key=f"fsel_{p_name}_{match_id}"):
                            if sel: selected_players.remove(p_name)
                            elif len(selected_players) < 11: selected_players.append(p_name)
                            else: st.error("11 players max!")
                            st.rerun()

        with tab_bat: render_compact_selection('Bat', squad_df)
        with tab_bowl: render_compact_selection('Bowl', squad_df)
        with tab_ar: render_compact_selection('All', squad_df)
        with tab_wk: render_compact_selection('Keeper', squad_df)

        # Validation & Submit
        st.divider()
        sel_df = squad_df[squad_df['player'].isin(selected_players)]
        bat_c = len(sel_df[sel_df['role'].str.contains('Bat', case=False)])
        bowl_c = len(sel_df[sel_df['role'].str.contains('Bowl', case=False)])
        wk_c = len(sel_df[sel_df['role'].str.contains('Keeper', case=False)])
        
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Batsmen", f"{bat_c}/2+", delta=bat_c-2 if bat_c>=2 else bat_c-2)
        v2.metric("Bowlers", f"{bowl_c}/2+", delta=bowl_c-2 if bowl_c>=2 else bowl_c-2)
        v3.metric("Wicketkeepers", f"{wk_c}/1+", delta=wk_c-1 if wk_c>=1 else wk_c-1)
        v4.metric("Squad Size", f"{len(selected_players)}/11")

        if bat_c>=2 and bowl_c>=2 and wk_c>=1 and len(selected_players)==11:
            st.success("✨ Your team is balanced and ready!")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                cap = st.selectbox("Captain (2x)", selected_players, key=f"c_{match_id}")
            with c_col2:
                vcap = st.selectbox("Vice-Captain (1.5x)", [p for p in selected_players if p != cap], key=f"vc_{match_id}")
            
            if st.button("🚀 Submit Fantasy Squad", type="primary"):
                try:
                    conn = get_db_connection()
                    user = conn.execute("SELECT id FROM users WHERE username = ?", (st.session_state.username,)).fetchone()
                    if user:
                        players_json = json.dumps({'players': selected_players, 'captain': cap, 'vice_captain': vcap})
                        save_fantasy_team(user['id'], t_id, match_id, players_json)
                        st.success("Team saved for the match!")
                        st.balloons()
                    conn.close()
                except Exception as e: st.error(f"Error: {e}")
        else:
            st.info("💡 Keep picking until you satisfy the requirements!")
            if selected_players:
                st.write("**Current Lineup:** " + ", ".join(selected_players))
                if st.button("Clear All"):
                    st.session_state[f"selected_players_{match_id}"] = []
                    st.rerun()

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
