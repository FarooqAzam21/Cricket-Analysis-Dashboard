import sqlite3
import json
from datetime import datetime

def backup_tournaments():
    """Backup all tournaments, teams, and matches to a JSON file before running migrate_data.py"""
    try:
        conn = sqlite3.connect('cricket_dashboard.db')
        cursor = conn.cursor()
        
        # Get all tournaments
        cursor.execute('SELECT * FROM tournaments')
        tournaments = cursor.fetchall()
        
        if not tournaments:
            print("❌ No tournaments found to backup")
            return False
        
        # Get all tournament teams and matches
        cursor.execute('SELECT * FROM tournament_teams')
        tournament_teams = cursor.fetchall()
        
        cursor.execute('SELECT * FROM tournament_matches')
        tournament_matches = cursor.fetchall()
        
        cursor.execute('SELECT * FROM fantasy_teams')
        fantasy_teams = cursor.fetchall()
        
        # Create backup dictionary
        backup = {
            'tournaments': [list(row) for row in tournaments],
            'tournament_teams': [list(row) for row in tournament_teams],
            'tournament_matches': [list(row) for row in tournament_matches],
            'fantasy_teams': [list(row) for row in fantasy_teams],
            'backup_time': datetime.now().isoformat()
        }
        
        # Save to JSON file
        backup_file = f'tournament_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(backup_file, 'w') as f:
            json.dump(backup, f, indent=2)
        
        print(f"✅ Tournament backup successful!")
        print(f"   📁 Backup file: {backup_file}")
        print(f"   📊 Backed up:")
        print(f"      - {len(tournaments)} tournaments")
        print(f"      - {len(tournament_teams)} teams")
        print(f"      - {len(tournament_matches)} matches")
        print(f"      - {len(fantasy_teams)} fantasy teams")
        print(f"\n💡 Keep this file safe! You can restore from it if needed.")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def restore_tournaments_from_backup(backup_file):
    """Restore tournaments from backup file after migrate_data.py"""
    try:
        with open(backup_file, 'r') as f:
            backup = json.load(f)
        
        conn = sqlite3.connect('cricket_dashboard.db')
        cursor = conn.cursor()
        
        # Restore tournaments
        for tourn in backup['tournaments']:
            try:
                cursor.execute('''INSERT INTO tournaments 
                                (id, name, start_date, end_date, status, tournament_format)
                                VALUES (?, ?, ?, ?, ?, ?)''',
                             tourn[:6])
            except sqlite3.IntegrityError:
                pass
        
        # Restore tournament teams
        for team in backup['tournament_teams']:
            try:
                cursor.execute('''INSERT INTO tournament_teams 
                                (id, tournament_id, team_name, group_letter, squad, matches_played, wins, losses, points)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                             team[:9])
            except sqlite3.IntegrityError:
                pass
        
        # Restore matches
        for match in backup['tournament_matches']:
            try:
                cursor.execute('''INSERT INTO tournament_matches 
                                (id, tournament_id, team1_id, team2_id, match_date, status, stage, group_letter, match_number, team1_score, team2_score, winner_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                             match[:12])
            except sqlite3.IntegrityError:
                pass
        
        # Restore fantasy teams
        for fantasy in backup['fantasy_teams']:
            try:
                cursor.execute('''INSERT INTO fantasy_teams 
                                (id, user_id, tournament_id, team_name, team_composition, captain, vice_captain, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                             fantasy[:8])
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
        
        print(f"✅ Restore successful!")
        print(f"   📊 Restored:")
        print(f"      - {len(backup['tournaments'])} tournaments")
        print(f"      - {len(backup['tournament_teams'])} teams")
        print(f"      - {len(backup['tournament_matches'])} matches")
        print(f"      - {len(backup['fantasy_teams'])} fantasy teams")
        
        return True
        
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--restore':
        if len(sys.argv) > 2:
            restore_tournaments_from_backup(sys.argv[2])
        else:
            print("Usage: python backup_tournaments.py --restore <backup_file.json>")
    else:
        backup_tournaments()
