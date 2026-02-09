import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import (
    get_db_connection, get_leaderboard, get_tournament
)

def show_leaderboard():
    """Display fantasy cricket leaderboard"""
    
    st.title("🏆 Fantasy Cricket Leaderboard")
    
    # Get tournaments
    conn = get_db_connection()
    tournaments = conn.execute("SELECT * FROM tournaments WHERE status IN ('planning', 'active', 'completed')").fetchall()
    conn.close()
    
    if not tournaments:
        st.info("No tournaments available")
        return
    
    # Select tournament
    tournament_options = {t['name']: t['id'] for t in tournaments}
    tournament_name = st.selectbox("Select Tournament", tournament_options.keys())
    tournament_id = tournament_options[tournament_name]
    
    tournament = get_tournament(tournament_id)
    
    if not tournament:
        st.error("Tournament not found")
        return
    
    st.info(f"📅 Tournament: {tournament['name']} | Status: {tournament['status'].upper()}")
    
    # Get leaderboard
    leaderboard = get_leaderboard(tournament_id)
    
    if not leaderboard:
        st.warning("No leaderboard data available yet")
        return
    
    # Prepare leaderboard data
    leaderboard_data = []
    for rank, entry in enumerate(leaderboard, 1):
        medal = ""
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        
        leaderboard_data.append({
            'Rank': f"{medal} {rank}" if medal else str(rank),
            'User': entry['username'],
            'Total Points': entry['total_points'],
            'Teams Created': entry['fantasy_teams_created'],
            'Avg Points': round(entry['total_points'] / max(entry['fantasy_teams_created'], 1), 2)
        })
    
    # Display leaderboard
    st.subheader("📊 Overall Rankings")
    
    df = pd.DataFrame(leaderboard_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Total Points": st.column_config.NumberColumn(format="%.1f"),
            "Avg Points": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    # Top performers
    col1, col2, col3 = st.columns(3)
    
    if len(leaderboard) >= 1:
        with col1:
            top_user = leaderboard[0]['username']
            top_points = leaderboard[0]['total_points']
            st.metric("🥇 Leader", top_user, f"{top_points} pts")
    
    if len(leaderboard) >= 2:
        with col2:
            runner_user = leaderboard[1]['username']
            runner_points = leaderboard[1]['total_points']
            diff = leaderboard[0]['total_points'] - runner_points
            st.metric("🥈 Runner-up", runner_user, f"{runner_points} pts", f"-{diff:.1f} pts")
    
    if len(leaderboard) >= 3:
        with col3:
            third_user = leaderboard[2]['username']
            third_points = leaderboard[2]['total_points']
            diff = leaderboard[0]['total_points'] - third_points
            st.metric("🥉 Third", third_user, f"{third_points} pts", f"-{diff:.1f} pts")
    
    # Awards section (if tournament is completed)
    if tournament['status'] == 'completed':
        st.divider()
        st.subheader("🎖️ Tournament Awards")
        
        col1, col2, col3 = st.columns(3)
        
        awards = [
            (col1, "🥇", "CHAMPION", "Top Scorer", 1),
            (col2, "🥈", "RUNNER-UP", "Second Place", 2),
            (col3, "🥉", "THIRD PLACE", "Third Place", 3)
        ]
        
        for col, medal, title, desc, rank in awards:
            with col:
                if rank <= len(leaderboard):
                    user_data = leaderboard[rank - 1]
                    st.metric(
                        f"{medal} {title}",
                        user_data['username'],
                        f"{user_data['total_points']} Points"
                    )
                    st.caption(desc)
    
    # User's position
    if 'username' in st.session_state and st.session_state.username:
        st.divider()
        st.subheader("Your Performance")
        
        user_entry = next(
            (entry for entry in leaderboard if entry['username'] == st.session_state.username),
            None
        )
        
        if user_entry:
            user_rank = leaderboard.index(user_entry) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Your Rank", f"#{user_rank}")
            
            with col2:
                st.metric("Total Points", f"{user_entry['total_points']:.1f}")
            
            with col3:
                st.metric("Teams Created", user_entry['fantasy_teams_created'])
            
            with col4:
                avg_points = round(user_entry['total_points'] / max(user_entry['fantasy_teams_created'], 1), 2)
                st.metric("Avg Per Team", avg_points)
            
            # Show how far from leader
            if user_rank > 1:
                leader_points = leaderboard[0]['total_points']
                gap = leader_points - user_entry['total_points']
                st.warning(f"📊 You're {gap:.1f} points behind the leader")
            else:
                st.success("🎉 You're the leader!")
        else:
            st.info("You haven't created any fantasy teams yet. Create a team to appear on the leaderboard!")

if __name__ == "__main__":
    show_leaderboard()
