
import sqlite3
import sys
import os

# Add project root (parent of scripts/) to path to use src.database
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import delete_match, get_db_connection

def revert_and_delete_match(match_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Get match details before deleting
    match = conn.execute("SELECT team1_id, team2_id, winner_id, status FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
    
    if not match:
        print(f"Error: Match {match_id} not found.")
        conn.close()
        return
    
    if match['status'] == 'completed':
        print(f"Match is completed. Reverting points for teams...")
        team1_id = match['team1_id']
        team2_id = match['team2_id']
        winner_id = match['winner_id']
        
        # Revert winner stats
        cursor.execute("""
            UPDATE tournament_teams 
            SET matches_played = MAX(0, matches_played - 1), 
                wins = MAX(0, wins - 1), 
                points = MAX(0, points - 2) 
            WHERE id = ?
        """, (winner_id,))
        
        # Revert loser stats
        loser_id = team1_id if winner_id == team2_id else team2_id
        cursor.execute("""
            UPDATE tournament_teams 
            SET matches_played = MAX(0, matches_played - 1), 
                losses = MAX(0, losses - 1) 
            WHERE id = ?
        """, (loser_id,))
        
        print(f"Points reverted for teams {winner_id} (winner) and {loser_id} (loser).")
    
    conn.commit()
    conn.close()
    
    # 2. Use the existing delete_match function to clean up related data
    success, msg = delete_match(match_id)
    if success:
        print(f"Success: {msg}")
    else:
        print(f"Error: {msg}")

if __name__ == "__main__":
    revert_and_delete_match(1)
