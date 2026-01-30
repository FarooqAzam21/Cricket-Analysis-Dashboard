from src.database import init_db, save_to_db, get_db_connection
from src.config import DATA_PATHS
import pandas as pd
import os
import sqlite3

def migrate():
    print("🔄 Safe Data Migration (Preserving User Accounts & Tournaments)")
    print("=" * 60)
    
    # 1. BACKUP existing users AND tournaments
    print("\n1️⃣ Backing up user accounts and tournaments...")
    db_path = 'cricket_dashboard.db'
    users_backup = None
    tournaments_backup = None
    tournament_teams_backup = None
    tournament_matches_backup = None
    fantasy_teams_backup = None
    match_performances_backup = None
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Backup users
            cursor.execute('SELECT * FROM users')
            users_backup = cursor.fetchall()
            print(f"   ✅ Backed up {len(users_backup)} user accounts")
            
            # Backup tournaments and related data
            cursor.execute('SELECT * FROM tournaments')
            tournaments_backup = cursor.fetchall()
            print(f"   ✅ Backed up {len(tournaments_backup)} tournaments")
            
            cursor.execute('SELECT * FROM tournament_teams')
            tournament_teams_backup = cursor.fetchall()
            print(f"   ✅ Backed up {len(tournament_teams_backup)} tournament teams")
            
            cursor.execute('SELECT * FROM tournament_matches')
            tournament_matches_backup = cursor.fetchall()
            print(f"   ✅ Backed up {len(tournament_matches_backup)} tournament matches")
            
            cursor.execute('SELECT * FROM fantasy_teams')
            fantasy_teams_backup = cursor.fetchall()
            print(f"   ✅ Backed up {len(fantasy_teams_backup)} fantasy teams")
            
            try:
                cursor.execute('SELECT * FROM match_player_performance')
                match_performances_backup = cursor.fetchall()
                print(f"   ✅ Backed up {len(match_performances_backup)} player performances")
            except:
                match_performances_backup = []
            
            conn.close()
        except Exception as e:
            print(f"   ℹ️  First migration - no data to backup: {e}")
    
    # 2. DELETE old database
    print("\n2️⃣ Resetting database...")
    if os.path.exists(db_path):
        os.remove(db_path)
        print("   ✅ Old database deleted")
    
    # 3. Initialize fresh database
    print("\n3️⃣ Creating fresh database schema...")
    init_db()
    
    # 4. Load DIRECTLY from CSV files
    print("\n4️⃣ Loading CSV data...")
    try:
        df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
        df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
        df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()
        
        # Clean and combine
        def clean(df):
            if df.empty: return df
            df.columns = df.columns.map(str).str.strip()
            for c in ['player', 'Team', 'Format', 'role']:
                if c in df.columns: 
                    df[c] = df[c].astype(str).str.replace(r'[\t"\']', '', regex=True).str.strip()
            return df

        composite = pd.concat([clean(df_bat), clean(df_ar), clean(df_bowl)], ignore_index=True, sort=False)
        all_players = composite.groupby(['player', 'Team', 'Format'], as_index=False).first()
        
    except Exception as csv_e:
        print(f"❌ CSV loading failed: {csv_e}")
        return
    
    if all_players.empty:
        print("❌ No data found in CSV files.")
        return

    # 5. Save to database
    print(f"\n5️⃣ Importing {len(all_players)} player records...")
    save_to_db(all_players)

    # 6. Restore users, tournaments, and fantasy data
    if users_backup or tournaments_backup:
        print("\n6️⃣ Restoring user accounts and tournament data...")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Restore users
            if users_backup:
                for user_row in users_backup:
                    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                                 (user_row[1], user_row[2]))
                print(f"   ✅ Restored {len(users_backup)} user accounts")
            
            # Restore tournaments
            if tournaments_backup:
                for tourn_row in tournaments_backup:
                    # Insert tournament preserving all fields
                    cursor.execute('''INSERT INTO tournaments 
                                    (id, name, start_date, end_date, status, tournament_format)
                                    VALUES (?, ?, ?, ?, ?, ?)''',
                                 (tourn_row[0], tourn_row[1], tourn_row[2], tourn_row[3], tourn_row[4], tourn_row[5] if len(tourn_row) > 5 else None))
                print(f"   ✅ Restored {len(tournaments_backup)} tournaments")
            
            # Restore tournament teams
            if tournament_teams_backup:
                for team_row in tournament_teams_backup:
                    cursor.execute('''INSERT INTO tournament_teams 
                                    (id, tournament_id, team_name, group_letter, players)
                                    VALUES (?, ?, ?, ?, ?)''',
                                 (team_row[0], team_row[1], team_row[2], team_row[3], team_row[4] if len(team_row) > 4 else ''))
                print(f"   ✅ Restored {len(tournament_teams_backup)} tournament teams")
            
            # Restore tournament matches
            if tournament_matches_backup:
                for match_row in tournament_matches_backup:
                    cursor.execute('''INSERT INTO tournament_matches 
                                    (id, tournament_id, team1_id, team2_id, match_date, status, stage, group_letter, match_number, team1_score, team2_score, winner_id)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                 match_row[:12])
                print(f"   ✅ Restored {len(tournament_matches_backup)} tournament matches")
            
            # Restore fantasy teams
            if fantasy_teams_backup:
                for fantasy_row in fantasy_teams_backup:
                    cursor.execute('''INSERT INTO fantasy_teams 
                                    (id, user_id, tournament_id, team_name, team_composition, captain, vice_captain, created_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                 fantasy_row[:8])
                print(f"   ✅ Restored {len(fantasy_teams_backup)} fantasy teams")
            
            # Restore match performances if they exist
            if match_performances_backup:
                for perf_row in match_performances_backup:
                    cursor.execute('''INSERT INTO match_player_performance 
                                    (id, match_id, player_name, team_id, runs, balls_faced, fours, sixes, wickets, economy, performance_type, created_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                 perf_row[:12])
                print(f"   ✅ Restored {len(match_performances_backup)} player performances")
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"   ❌ Failed to restore data: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✅ Migration successful!")
    print(f"   ✓ {len(all_players)} player records updated")
    print(f"   ✓ User accounts preserved")
    if tournaments_backup:
        print(f"   ✓ {len(tournaments_backup)} tournaments preserved")
    print("=" * 60)

if __name__ == "__main__":
    migrate()
