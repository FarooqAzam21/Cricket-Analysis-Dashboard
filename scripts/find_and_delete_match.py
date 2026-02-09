
import sqlite3
import os

DB_PATH = "cricket_dashboard.db"

def find_and_revert_match():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Search for Pakistan vs Netherlands match where Netherlands won
    # Using LIKE to be safe with naming variations
    query = """
    SELECT m.id, t1.team_name as t1_name, t2.team_name as t2_name, w.team_name as winner_name, 
           m.team1_id, m.team2_id, m.winner_id, m.status, m.tournament_id as t_id
    FROM tournament_matches m 
    JOIN tournament_teams t1 ON m.team1_id = t1.id 
    JOIN tournament_teams t2 ON m.team2_id = t2.id 
    LEFT JOIN tournament_teams w ON m.winner_id = w.id 
    WHERE ((t1.team_name LIKE '%Pakistan%' AND t2.team_name LIKE '%Netherlands%') 
       OR (t1.team_name LIKE '%Netherlands%' AND t2.team_name LIKE '%Pakistan%'))
      AND w.team_name LIKE '%Netherlands%'
    """
    
    matches = cursor.execute(query).fetchall()
    
    if not matches:
        print("No matching completed match found (Pak vs Ned where Ned won).")
        # Check if it was already deleted and we just need to verify points
        print("\nChecking current standings...")
        standings = cursor.execute("SELECT id, team_name, matches_played, wins, losses, points FROM tournament_teams WHERE team_name LIKE '%Pakistan%' OR team_name LIKE '%Netherlands%'").fetchall()
        for row in standings:
            print(dict(row))
        conn.close()
        return

    for m in matches:
        print(f"\nProcessing Match ID {m['id']}: {m['t1_name']} vs {m['t2_name']} (Winner: {m['winner_name']})")
        
        if m['status'] == 'completed':
            print("Reverting points...")
            winner_id = m['winner_id']
            loser_id = m['team1_id'] if winner_id == m['team2_id'] else m['team2_id']
            
            # Revert winner
            cursor.execute("UPDATE tournament_teams SET matches_played = MAX(0, matches_played - 1), wins = MAX(0, wins - 1), points = MAX(0, points - 2) WHERE id = ?", (winner_id,))
            # Revert loser
            cursor.execute("UPDATE tournament_teams SET matches_played = MAX(0, matches_played - 1), losses = MAX(0, losses - 1) WHERE id = ?", (loser_id,))
            
        # Clean up related data (as delete_match does)
        cursor.execute("DELETE FROM fantasy_scores WHERE fantasy_team_id IN (SELECT id FROM fantasy_teams WHERE match_id = ?)", (m['id'],))
        cursor.execute("DELETE FROM fantasy_teams WHERE match_id = ?", (m['id'],))
        cursor.execute("DELETE FROM match_player_performance WHERE match_id = ?", (m['id'],))
        cursor.execute("DELETE FROM tournament_matches WHERE id = ?", (m['id'],))
        
        print(f"Match {m['id']} and related data deleted.")

    conn.commit()
    conn.close()
    print("\nOperation complete.")

if __name__ == "__main__":
    find_and_revert_match()
