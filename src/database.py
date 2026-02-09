import sqlite3
import pandas as pd
import os

# Database Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "cricket_dashboard.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Players Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player TEXT,
            team TEXT,
            format TEXT,
            matches INTEGER,
            innings INTEGER,
            no INTEGER,
            runs INTEGER,
            wickets INTEGER,
            average REAL,
            strike_rate REAL,
            bowling_average REAL,
            economy REAL,
            hundreds INTEGER,
            fifties INTEGER,
            batting_position INTEGER,
            role TEXT,
            image_url TEXT,
            UNIQUE(player, team, format)
        )
    ''')
    
    # Create Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    
    # Create Scout Feedback Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scout_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            source_player TEXT,
            similar_player TEXT,
            format TEXT,
            rating TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Tournament Tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            status TEXT DEFAULT 'planning',
            start_date TEXT,
            end_date TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tournament_teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER,
            team_name TEXT,
            group_letter TEXT,
            squad TEXT,
            matches_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            points INTEGER DEFAULT 0,
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tournament_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id INTEGER,
            team1_id INTEGER,
            team2_id INTEGER,
            match_date TEXT,
            match_time TEXT DEFAULT '10:00',
            stage TEXT,
            group_letter TEXT,
            status TEXT DEFAULT 'scheduled',
            winner_id INTEGER,
            team1_score INTEGER,
            team2_score INTEGER,
            batting_first_id INTEGER,
            team1_overs REAL DEFAULT 20.0,
            team2_overs REAL DEFAULT 20.0,
            team1_all_out INTEGER DEFAULT 0,
            team2_all_out INTEGER DEFAULT 0,
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY(team1_id) REFERENCES tournament_teams(id),
            FOREIGN KEY(team2_id) REFERENCES tournament_teams(id)
        )
    ''')
    
    # Quick fix for existing databases
    try:
        cursor.execute("ALTER TABLE tournament_matches ADD COLUMN match_time TEXT DEFAULT '10:00'")
    except sqlite3.OperationalError: pass

    try:
        cursor.execute("ALTER TABLE tournament_matches ADD COLUMN batting_first_id INTEGER")
    except sqlite3.OperationalError: pass

    try:
        cursor.execute("ALTER TABLE tournament_matches ADD COLUMN team1_overs REAL DEFAULT 20.0")
    except sqlite3.OperationalError: pass

    try:
        cursor.execute("ALTER TABLE tournament_matches ADD COLUMN team2_overs REAL DEFAULT 20.0")
    except sqlite3.OperationalError: pass

    try:
        cursor.execute("ALTER TABLE tournament_matches ADD COLUMN team1_all_out INTEGER DEFAULT 0")
    except sqlite3.OperationalError: pass

    try:
        cursor.execute("ALTER TABLE tournament_matches ADD COLUMN team2_all_out INTEGER DEFAULT 0")
    except sqlite3.OperationalError: pass
    
    try:
        cursor.execute("ALTER TABLE match_player_performance ADD COLUMN is_not_out INTEGER DEFAULT 0")
    except sqlite3.OperationalError: pass
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fantasy_teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            tournament_id INTEGER,
            match_id INTEGER,
            players_json TEXT,
            captain_id INTEGER,
            vice_captain_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY(match_id) REFERENCES tournament_matches(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fantasy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fantasy_team_id INTEGER,
            total_score REAL DEFAULT 0,
            rank INTEGER,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(fantasy_team_id) REFERENCES fantasy_teams(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tournament_leaderboard (
            user_id INTEGER,
            tournament_id INTEGER,
            total_points REAL DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, tournament_id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_playing_xi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            team_id INTEGER,
            players_json TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(match_id, team_id),
            FOREIGN KEY(match_id) REFERENCES tournament_matches(id),
            FOREIGN KEY(team_id) REFERENCES tournament_teams(id)
        )
    ''')
    
    # Create match_player_performance Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_player_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            player_name TEXT,
            team_id INTEGER,
            runs INTEGER DEFAULT 0,
            balls_faced INTEGER DEFAULT 0,
            fours INTEGER DEFAULT 0,
            sixes INTEGER DEFAULT 0,
            wickets INTEGER DEFAULT 0,
            overs_bowled REAL DEFAULT 0,
            runs_conceded INTEGER DEFAULT 0,
            economy REAL DEFAULT 0,
            catches INTEGER DEFAULT 0,
            performance_type TEXT,
            is_not_out INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(match_id) REFERENCES tournament_matches(id),
            FOREIGN KEY(team_id) REFERENCES tournament_teams(id)
        )
    ''')
    
    # Ensure all columns exist (Migration/Repair)
    perf_cols = {
        'balls_faced': 'INTEGER DEFAULT 0',
        'fours': 'INTEGER DEFAULT 0',
        'sixes': 'INTEGER DEFAULT 0',
        'wickets': 'INTEGER DEFAULT 0',
        'overs_bowled': 'REAL DEFAULT 0',
        'runs_conceded': 'INTEGER DEFAULT 0',
        'economy': 'REAL DEFAULT 0',
        'catches': 'INTEGER DEFAULT 0',
        'is_not_out': 'INTEGER DEFAULT 0',
        'performance_type': 'TEXT'
    }
    for col, dtype in perf_cols.items():
        try:
            cursor.execute(f"ALTER TABLE match_player_performance ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError: pass

    conn.commit()
    conn.close()
    
    # Run data repair for corrupted JSON
    repair_corrupted_data()

def repair_corrupted_data():
    """Fix double-encoded JSON lists in match_playing_xi and tournament_teams"""
    try:
        import json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Fix match_playing_xi
        rows = cursor.execute("SELECT id, players_json FROM match_playing_xi").fetchall()
        for row in rows:
            p_json = row['players_json']
            try:
                data = json.loads(p_json)
                # If first element is a string that looks like a JSON list, it's double encoded
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str) and data[0].startswith('['):
                    inner_data = json.loads(data[0])
                    if isinstance(inner_data, list):
                        new_json = json.dumps(inner_data)
                        cursor.execute("UPDATE match_playing_xi SET players_json = ? WHERE id = ?", (new_json, row['id']))
            except: continue
            
        # 2. Fix tournament_teams squad
        rows = cursor.execute("SELECT id, squad FROM tournament_teams").fetchall()
        for row in rows:
            squad = row['squad']
            if not squad: continue
            try:
                data = json.loads(squad)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str) and data[0].startswith('['):
                    inner_data = json.loads(data[0])
                    if isinstance(inner_data, list):
                        new_json = json.dumps(inner_data)
                        cursor.execute("UPDATE tournament_teams SET squad = ? WHERE id = ?", (new_json, row['id']))
            except: continue
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Data Repair Error: {e}")
        return False

def save_scout_feedback(username, source_player, similar_player, format_type, rating):
    """Save user feedback on similarity results."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO scout_feedback (username, source_player, similar_player, format, rating) VALUES (?, ?, ?, ?, ?)",
            (username, source_player, similar_player, format_type, rating)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False
    finally:
        conn.close()

def get_feedback_stats():
    """Get statistics on feedback for model improvement."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            "SELECT source_player, similar_player, format, rating, COUNT(*) as count FROM scout_feedback GROUP BY source_player, similar_player, format, rating",
            conn
        )
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def add_user(username, password_hash):
    """Add a new user with hashed password."""
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    """Retrieve user details for authentication."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

# ==================== PLAYING XI MANAGEMENT ====================

def save_playing_xi(match_id, team_id, players_list):
    """Save the announced playing 11 for a team in a match"""
    try:
        import json
        conn = get_db_connection()
        cursor = conn.cursor()
        players_json = json.dumps(players_list)
        cursor.execute("""
            INSERT INTO match_playing_xi (match_id, team_id, players_json)
            VALUES (?, ?, ?)
            ON CONFLICT(match_id, team_id) DO UPDATE SET
            players_json = excluded.players_json,
            updated_at = CURRENT_TIMESTAMP
        """, (match_id, team_id, players_json))
        conn.commit()
        conn.close()
        return True, "Playing 11 saved successfully!"
    except Exception as e:
        return False, str(e)

def get_playing_xi(match_id, team_id):
    """Get the announced playing 11 for a team in a match"""
    try:
        import json
        conn = get_db_connection()
        row = conn.execute("SELECT players_json FROM match_playing_xi WHERE match_id = ? AND team_id = ?", 
                          (match_id, team_id)).fetchone()
        conn.close()
        if row:
            return json.loads(row['players_json'])
        return []
    except Exception as e:
        print(f"Error getting playing XI: {e}")
        return []

def get_match_playing_xi(match_id):
    """Get all players currently in playing 11 for both teams in a match"""
    try:
        conn = get_db_connection()
        match = conn.execute("SELECT team1_id, team2_id FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
        conn.close()
        
        if not match: return []
        
        xi1 = get_playing_xi(match_id, match['team1_id'])
        xi2 = get_playing_xi(match_id, match['team2_id'])
        # Filter out empty strings if any
        xi1 = [p for p in xi1 if p.strip()]
        xi2 = [p for p in xi2 if p.strip()]
        return list(set(xi1 + xi2)) # Return unique names
    except Exception as e:
        print(f"Error getting combined match playing XI: {e}")
        return []

def save_to_db(df):
    """Save/Update a dataframe to the players table."""
    conn = get_db_connection()
    # Normalize column names to match DB schema
    df_db = df.copy()
    
    # Map CSV names to DB names if necessary
    mapping = {
        '100s': 'hundreds',
        '50s': 'fifties',
        'NO': 'no'
    }
    df_db = df_db.rename(columns=mapping)
    
    # Keep only columns that exist in DB
    db_cols = ['player', 'team', 'format', 'matches', 'innings', 'no', 'runs', 'wickets', 'average', 
               'strike_rate', 'bowling_average', 'economy', 'hundreds', 'fifties', 
               'batting_position', 'role', 'image_url']
    
    # Ensure column casing matches exactly
    df_db.columns = [c.lower() for c in df_db.columns]
    
    # Use REPLACE to handle the UNIQUE constraint (UPSERT)
    for _, row in df_db.iterrows():
        placeholders = ', '.join(['?'] * len(db_cols))
        cols = ', '.join(db_cols)
        values = tuple(row.get(col) for col in db_cols)
        
        try:
            conn.execute(f"INSERT OR REPLACE INTO players ({cols}) VALUES ({placeholders})", values)
        except Exception as e:
            print(f"Error inserting {row.get('player')}: {e}")
            
    conn.commit()
    conn.close()

def fetch_all_players_from_db():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM players", conn)
    conn.close()
    return df
# ============ TOURNAMENT FUNCTIONS ============

def create_tournament(name, start_date, end_date):
    """Create a new tournament"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO tournaments (name, start_date, end_date, status) 
                         VALUES (?, ?, ?, ?)""",
                      (name, start_date, end_date, 'planning'))
        conn.commit()
        tournament_id = cursor.lastrowid
        conn.close()
        return tournament_id
    except Exception as e:
        print(f"Error creating tournament: {e}")
        return None

def get_tournament(tournament_id):
    """Get tournament details"""
    conn = get_db_connection()
    tournament = conn.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,)).fetchone()
    conn.close()
    return tournament

def add_team_to_tournament(tournament_id, team_name, group_letter):
    """Add a team to tournament"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tournament_teams (tournament_id, team_name, group_letter) VALUES (?, ?, ?)",
                  (tournament_id, team_name, group_letter))
    conn.commit()
    team_id = cursor.lastrowid
    conn.close()
    return team_id

def get_tournament_teams(tournament_id):
    """Get all teams in a tournament"""
    conn = get_db_connection()
    teams = conn.execute("SELECT * FROM tournament_teams WHERE tournament_id = ? ORDER BY group_letter, team_name", 
                        (tournament_id,)).fetchall()
    conn.close()
    return teams

def get_group_standings(tournament_id, group_letter):
    """Get standings for a specific group"""
    conn = get_db_connection()
    standings = conn.execute("""
        SELECT id, team_name, matches_played, wins, losses, points 
        FROM tournament_teams 
        WHERE tournament_id = ? AND group_letter = ? 
        ORDER BY points DESC, wins DESC
    """, (tournament_id, group_letter)).fetchall()
    conn.close()
    return standings

def create_tournament_match(tournament_id, team1_id, team2_id, match_date, stage, group_letter=None, match_time='10:00'):
    """Create a match in tournament"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tournament_matches (tournament_id, team1_id, team2_id, match_date, match_time, stage, group_letter, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (tournament_id, team1_id, team2_id, match_date, match_time, stage, group_letter, 'scheduled'))
    conn.commit()
    match_id = cursor.lastrowid
    conn.close()
    return match_id

def get_tournament_matches(tournament_id, stage=None, group_letter=None):
    """Get matches from tournament"""
    conn = get_db_connection()
    query = "SELECT * FROM tournament_matches WHERE tournament_id = ?"
    params = [tournament_id]
    
    if stage:
        query += " AND stage = ?"
        params.append(stage)
    if group_letter:
        query += " AND group_letter = ?"
        params.append(group_letter)
    
    query += " ORDER BY match_date ASC, match_time ASC"
    matches = conn.execute(query, params).fetchall()
    conn.close()
    return matches

def total_tournament_reset(tournament_id):
    """Reset all tournament progress, matches, performances and standings"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Reset matches
        cursor.execute("""
            UPDATE tournament_matches 
            SET status = 'scheduled', winner_id = NULL, team1_score = 0, team2_score = 0,
                batting_first_id = NULL, team1_overs = 20.0, team2_overs = 20.0,
                team1_all_out = 0, team2_all_out = 0
            WHERE tournament_id = ?
        """, (tournament_id,))
        
        # 2. Reset team standings
        cursor.execute("""
            UPDATE tournament_teams 
            SET matches_played = 0, wins = 0, losses = 0, points = 0
            WHERE tournament_id = ?
        """, (tournament_id,))
        
        # 3. Delete performances for these matches
        matches = conn.execute("SELECT id FROM tournament_matches WHERE tournament_id = ?", (tournament_id,)).fetchall()
        match_ids = [m['id'] for m in matches]
        
        if match_ids:
            placeholders = ','.join(['?'] * len(match_ids))
            cursor.execute(f"DELETE FROM match_player_performance WHERE match_id IN ({placeholders})", match_ids)
            cursor.execute(f"DELETE FROM match_playing_xi WHERE match_id IN ({placeholders})", match_ids)
            
            # Reset fantasy scores for these matches
            cursor.execute(f"DELETE FROM fantasy_scores WHERE fantasy_team_id IN (SELECT id FROM fantasy_teams WHERE match_id IN ({placeholders}))", match_ids)
        
        # 4. Reset leaderboard
        cursor.execute("UPDATE tournament_leaderboard SET total_points = 0 WHERE tournament_id = ?", (tournament_id,))
        
        conn.commit()
        conn.close()
        return True, "Tournament progress has been completely reset."
    except Exception as e:
        return False, str(e)

def update_match_result(match_id, winner_id, team1_score, team2_score, batting_first_id=None, t1_overs=20.0, t2_overs=20.0, t1_all_out=False, t2_all_out=False):
    """Update match result after completion with NRR data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update match status and NRR fields
    cursor.execute("""
        UPDATE tournament_matches 
        SET status = ?, winner_id = ?, team1_score = ?, team2_score = ?,
            batting_first_id = ?, team1_overs = ?, team2_overs = ?,
            team1_all_out = ?, team2_all_out = ?
        WHERE id = ?
    """, ('completed', winner_id, team1_score, team2_score, 
          batting_first_id, t1_overs, t2_overs, 
          1 if t1_all_out else 0, 1 if t2_all_out else 0, match_id))
    
    # Update team stats
    match = conn.execute("SELECT team1_id, team2_id FROM tournament_matches WHERE id = ?", (match_id,)).fetchone()
    team1_id, team2_id = match['team1_id'], match['team2_id']
    
    # Update teams
    if winner_id == team1_id:
        cursor.execute("""
            UPDATE tournament_teams SET matches_played = matches_played + 1, wins = wins + 1, points = points + 2 
            WHERE id = ?
        """, (team1_id,))
        cursor.execute("""
            UPDATE tournament_teams SET matches_played = matches_played + 1, losses = losses + 1 
            WHERE id = ?
        """, (team2_id,))
    else:
        cursor.execute("""
            UPDATE tournament_teams SET matches_played = matches_played + 1, losses = losses + 1 
            WHERE id = ?
        """, (team1_id,))
        cursor.execute("""
            UPDATE tournament_teams SET matches_played = matches_played + 1, wins = wins + 1, points = points + 2 
            WHERE id = ?
        """, (team2_id,))
    
    conn.commit()
    conn.close()
    
    # Calculate fantasy points for all users who created teams for this match
    calculate_and_save_fantasy_points(match_id)

def calculate_and_save_fantasy_points(match_id):
    """Calculate fantasy points for all teams based on match_player_performance"""
    try:
        conn = get_db_connection()
        
        # 1. Get all performances for this match
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM match_player_performance WHERE match_id = ?", (match_id,))
        perfs = cursor.fetchall()
        
        # Create a lookup dictionary: { 'player_name': {stats} }
        perf_map = {}
        for p in perfs:
            perf_map[p['player_name'].lower()] = dict(p)
            
        # 2. Get all fantasy teams for this match
        cursor.execute("SELECT id, players_json FROM fantasy_teams WHERE match_id = ?", (match_id,))
        fantasy_teams = cursor.fetchall()
        
        for ft in fantasy_teams:
            ft_id = ft['id']
            try:
                p_data = json.loads(ft['players_json'])
                players = p_data.get('players', [])
                captain = p_data.get('captain', '').lower()
                vice_captain = p_data.get('vice_captain', '').lower()
                
                total_ft_points = 0
                
                for p_name in players:
                    p_name_l = p_name.lower()
                    if p_name_l in perf_map:
                        p_perf = perf_map[p_name_l]
                        
                        # Calculate points
                        p_points = calculate_batsman_fantasy_points(
                            p_perf.get('runs', 0), 
                            p_perf.get('fours', 0), 
                            p_perf.get('sixes', 0), 
                            p_perf.get('balls_faced', 0)
                        )
                        
                        p_points += calculate_bowler_fantasy_points(
                            p_perf.get('wickets', 0),
                            p_perf.get('economy', 0),
                            p_perf.get('overs_bowled', 0)
                        )
                        
                        # Apply Captain/VC
                        if p_name_l == captain:
                            p_points *= 2
                        elif p_name_l == vice_captain:
                            p_points *= 1.5
                            
                        total_ft_points += p_points
                
                # Update Score in DB
                cursor.execute("""
                    INSERT INTO fantasy_scores (fantasy_team_id, total_score)
                    VALUES (?, ?)
                    ON CONFLICT(fantasy_team_id) DO UPDATE SET 
                    total_score = excluded.total_score,
                    updated_at = CURRENT_TIMESTAMP
                """, (ft_id, total_ft_points))
                
            except Exception as inner_e:
                print(f"Error for fantasy team {ft_id}: {inner_e}")
                
        conn.commit()
        # 3. Update Global Leaderboard for this Tournament
        cursor.execute("SELECT tournament_id FROM tournament_matches WHERE id = ?", (match_id,))
        tid_row = cursor.fetchone()
        if tid_row:
            tournament_id = tid_row['tournament_id']
            cursor.execute("SELECT DISTINCT user_id FROM fantasy_teams WHERE tournament_id = ?", (tournament_id,))
            users = cursor.fetchall()
            
            for u in users:
                user_id = u['user_id']
                cursor.execute("""
                    SELECT SUM(fs.total_score) 
                    FROM fantasy_scores fs
                    JOIN fantasy_teams ft ON fs.fantasy_team_id = ft.id
                    WHERE ft.user_id = ? AND ft.tournament_id = ?
                """, (user_id, tournament_id))
                total_pts = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                    INSERT INTO tournament_leaderboard (user_id, tournament_id, total_points)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, tournament_id) DO UPDATE SET
                    total_points = excluded.total_points,
                    updated_at = CURRENT_TIMESTAMP
                """, (user_id, tournament_id, total_pts))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        import traceback
        print(f"Global Fantasy Calc Error: {traceback.format_exc()}")
        return False

def save_fantasy_team(user_id, tournament_id, match_id, players_json, captain_id=None, vice_captain_id=None):
    """Save user's fantasy team"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fantasy_teams (user_id, tournament_id, match_id, players_json, captain_id, vice_captain_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, tournament_id, match_id, players_json, captain_id, vice_captain_id))
    conn.commit()
    fantasy_team_id = cursor.lastrowid
    conn.close()
    return fantasy_team_id

def get_user_fantasy_teams(user_id, tournament_id):
    """Get all fantasy teams created by user for tournament"""
    conn = get_db_connection()
    teams = conn.execute("""
        SELECT * FROM fantasy_teams 
        WHERE user_id = ? AND tournament_id = ?
        ORDER BY created_at DESC
    """, (user_id, tournament_id)).fetchall()
    conn.close()
    return teams

def update_fantasy_score(fantasy_team_id, total_score):
    """Update fantasy team score"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE fantasy_scores SET total_score = ?, updated_at = CURRENT_TIMESTAMP
        WHERE fantasy_team_id = ?
    """, (total_score, fantasy_team_id))
    conn.commit()
    conn.close()

def update_leaderboard(user_id, tournament_id, total_points):
    """Update leaderboard"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO leaderboard (user_id, tournament_id, total_points, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (user_id, tournament_id, total_points))
    conn.commit()
    conn.close()

def get_leaderboard(tournament_id):
    """Get tournament leaderboard"""
    conn = get_db_connection()
    leaderboard = conn.execute("""
        SELECT l.*, u.username FROM leaderboard l
        JOIN users u ON l.user_id = u.id
        WHERE l.tournament_id = ?
        ORDER BY l.total_points DESC
    """, (tournament_id,)).fetchall()
    conn.close()
    return leaderboard

def delete_tournament(tournament_id):
    """Delete tournament and all related data (cascade)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete in order of dependencies
        cursor.execute("DELETE FROM leaderboard WHERE tournament_id = ?", (tournament_id,))
        cursor.execute("DELETE FROM fantasy_scores WHERE fantasy_team_id IN (SELECT id FROM fantasy_teams WHERE tournament_id = ?)", (tournament_id,))
        cursor.execute("DELETE FROM fantasy_teams WHERE tournament_id = ?", (tournament_id,))
        cursor.execute("DELETE FROM tournament_matches WHERE tournament_id = ?", (tournament_id,))
        cursor.execute("DELETE FROM tournament_teams WHERE tournament_id = ?", (tournament_id,))
        cursor.execute("DELETE FROM tournaments WHERE id = ?", (tournament_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting tournament: {e}")
        return False

def update_team_squad(team_id, players_json):
    """Update team squad with player list"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tournament_teams SET squad = ? WHERE id = ?", (players_json, team_id))
    conn.commit()
    conn.close()

def get_team_details(team_id):
    """Get team details including squad"""
    conn = get_db_connection()
    team = conn.execute("SELECT * FROM tournament_teams WHERE id = ?", (team_id,)).fetchone()
    conn.close()
    return team

def update_match_date(match_id, new_date):
    """Update match date"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tournament_matches SET match_date = ? WHERE id = ?", (new_date, match_id))
    conn.commit()
    conn.close()

def update_match_time(match_id, new_time):
    """Update match time"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tournament_matches SET match_time = ? WHERE id = ?", (new_time, match_id))
    conn.commit()
    conn.close()

def delete_match(match_id):
    """Delete a match and its related data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Delete fantasy scores
        cursor.execute("DELETE FROM fantasy_scores WHERE fantasy_team_id IN (SELECT id FROM fantasy_teams WHERE match_id = ?)", (match_id,))
        # 2. Delete fantasy teams
        cursor.execute("DELETE FROM fantasy_teams WHERE match_id = ?", (match_id,))
        # 3. Delete performances
        cursor.execute("DELETE FROM match_player_performance WHERE match_id = ?", (match_id,))
        # 4. Delete match
        cursor.execute("DELETE FROM tournament_matches WHERE id = ?", (match_id,))
        
        conn.commit()
        conn.close()
        return True, "Match and all related data deleted successfully."
    except Exception as e:
        return False, str(e)

def get_group_stage_matches(tournament_id):
    """Get only group stage matches"""
    conn = get_db_connection()
    matches = conn.execute(
        "SELECT * FROM tournament_matches WHERE tournament_id = ? AND stage = 'group' ORDER BY match_date",
        (tournament_id,)
    ).fetchall()
    conn.close()
    return matches

def update_team_name(team_id, new_name):
    """Update tournament team name"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tournament_teams SET team_name = ? WHERE id = ?", (new_name, team_id))
    conn.commit()
    conn.close()

def update_match_number(match_id, new_match_number):
    """Update match number (sequential number for same date matches)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tournament_matches SET match_number = ? WHERE id = ?", (new_match_number, match_id))
    conn.commit()
    conn.close()

def calculate_nrr(team_id, tournament_id):
    """Calculate Net Run Rate for a team in a tournament"""
    conn = get_db_connection()
    
    # Get all completed matches for this team
    matches = conn.execute("""
        SELECT * FROM tournament_matches 
        WHERE tournament_id = ? AND status = 'completed'
        AND (team1_id = ? OR team2_id = ?)
    """, (tournament_id, team_id, team_id)).fetchall()
    
    runs_scored = 0
    runs_conceded = 0
    overs_faced = 0.0
    overs_bowled = 0.0
    
    def over_to_decimal(o):
        parts = str(o).split('.')
        ov = int(parts[0])
        balls = int(parts[1]) if len(parts) > 1 else 0
        return ov + (balls / 6.0)

    for match in matches:
        is_team1 = (match['team1_id'] == team_id)
        
        if is_team1:
            runs_scored += match['team1_score']
            runs_conceded += match['team2_score']
            # If all out, use full 20 overs
            overs_faced += 20.0 if match['team1_all_out'] else over_to_decimal(match['team1_overs'])
            overs_bowled += 20.0 if match['team2_all_out'] else over_to_decimal(match['team2_overs'])
        else:
            runs_scored += match['team2_score']
            runs_conceded += match['team1_score']
            overs_faced += 20.0 if match['team2_all_out'] else over_to_decimal(match['team2_overs'])
            overs_bowled += 20.0 if match['team1_all_out'] else over_to_decimal(match['team1_overs'])
    
    conn.close()
    
    # NRR = (Runs Scored / Overs Faced) - (Runs Conceded / Overs Bowled)
    if overs_faced == 0 or overs_bowled == 0:
        return 0.0
    
    nrr = (runs_scored / overs_faced) - (runs_conceded / overs_bowled)
    return round(nrr, 3)

def get_group_standings_with_nrr(tournament_id, group_letter):
    """Get group standings with NRR for qualification"""
    conn = get_db_connection()
    
    # Get all teams in the group
    teams = conn.execute("""
        SELECT * FROM tournament_teams 
        WHERE tournament_id = ? AND group_letter = ?
        ORDER BY points DESC, wins DESC
    """, (tournament_id, group_letter)).fetchall()
    
    standings = []
    for team in teams:
        nrr = calculate_nrr(team['id'], tournament_id)
        standings.append({
            'team_id': team['id'],
            'team_name': team['team_name'],
            'matches_played': team['matches_played'],
            'wins': team['wins'],
            'losses': team['losses'],
            'points': team['points'],
            'nrr': nrr
        })
    
    # Sort by points first, then NRR
    standings = sorted(standings, key=lambda x: (x['points'], x['nrr']), reverse=True)
    
    conn.close()
    return standings

def get_super8_qualified_teams(tournament_id):
    """Get top 2 teams from each group for Super 8 stage"""
    qualified_teams = []
    
    for group in ['A', 'B', 'C', 'D']:
        standings = get_group_standings_with_nrr(tournament_id, group)
        if len(standings) >= 2:
            qualified_teams.append(standings[0])
            qualified_teams.append(standings[1])
    
    return qualified_teams

def create_super8_matches(tournament_id):
    """Create Super 8 matches from qualified teams"""
    try:
        qualified = get_super8_qualified_teams(tournament_id)
        
        if len(qualified) < 8:
            return False
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get tournament details
        tournament = conn.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,)).fetchone()
        
        # Create round-robin for Super 8 (8 teams = 28 matches)
        from datetime import datetime, timedelta
        base_date = datetime.strptime(tournament['start_date'], "%Y-%m-%d")
        
        # Assume group stage ends and Super 8 starts 20 days after tournament start
        super8_start_date = base_date + timedelta(days=20)
        match_counter = 1
        
        for i in range(len(qualified)):
            for j in range(i + 1, len(qualified)):
                team1_id = qualified[i]['team_id']
                team2_id = qualified[j]['team_id']
                match_date = (super8_start_date + timedelta(days=match_counter)).strftime("%Y-%m-%d")
                
                cursor.execute("""
                    INSERT INTO tournament_matches 
                    (tournament_id, team1_id, team2_id, match_date, stage, match_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (tournament_id, team1_id, team2_id, match_date, 'super8', match_counter))
                
                match_counter += 1
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating Super 8 matches: {e}")
        return False

def update_tournament_format(tournament_id, stages):
    """Update tournament format/stages (flexible)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    stages_json = json.dumps(stages)
    cursor.execute("""
        UPDATE tournaments SET tournament_format = ? WHERE id = ?
    """, (stages_json, tournament_id))
    conn.commit()
    conn.close()

def get_tournament_format(tournament_id):
    """Get tournament format/stages"""
    conn = get_db_connection()
    tournament = conn.execute("SELECT tournament_format FROM tournaments WHERE id = ?", (tournament_id,)).fetchone()
    conn.close()
    
    return ['group', 'semi-final', 'final']
    
def update_wc_csv_stats(player_name, match_stats, team_name="Unknown"):
    """
    Update wc_players.csv with match performance data.
    match_stats = { 'runs': X, 'balls': X, 'fours': X, 'sixes': X, 'wickets': X, 'overs': X, 'runs_conceded': X, 'catches': X }
    """
    try:
        from .config import DATA_PATHS
        csv_path = DATA_PATHS.get("wc_players")
        if not csv_path or not os.path.exists(csv_path):
            return False, "CSV file not found"
            
        df = pd.read_csv(csv_path)
        
        # Ensure helper columns exist for accurate aggregate calculation
        required_cols = ['player', 'Team', 'Format', 'matches', 'Innings', 'NO', 'runs', 'wickets', 
                         'average', 'bowling_average', 'strike_rate', 'HS', '100s', '50s', 
                         'batting_position', 'image_url', 'role', 'economy',
                         'total_balls_faced', 'total_overs_bowled', 'total_runs_conceded', 
                         'total_fours', 'total_sixes', 'total_catches']
        
        for col in required_cols:
            if col not in df.columns:
                if col in ['player', 'Team', 'Format', 'role', 'HS', 'image_url']:
                    df[col] = ""
                else:
                    df[col] = 0

        # Find player (case insensitive)
        player_mask = df['player'].str.lower() == player_name.lower()
        
        if not player_mask.any():
            # Add new player row
            new_row = {col: (0 if df[col].dtype in ['int64', 'float64'] else "") for col in df.columns}
            new_row.update({
                'player': player_name,
                'Team': team_name,
                'Format': 'T20',
                'matches': 0,
                'Innings': 0,
                'runs': 0,
                'wickets': 0,
                'HS': '0'
            })
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            player_mask = df['player'] == player_name # Refresh mask
            
        idx = df[player_mask].index[0]
        
        # Update basics
        df.at[idx, 'matches'] = safe_int(df.at[idx, 'matches']) + 1
        
        # Batting updates
        if match_stats.get('balls', 0) > 0 or match_stats.get('runs', 0) > 0:
            df.at[idx, 'Innings'] = safe_int(df.at[idx, 'Innings']) + 1
            df.at[idx, 'runs'] = safe_int(df.at[idx, 'runs']) + match_stats['runs']
            df.at[idx, 'total_balls_faced'] = safe_int(df.at[idx, 'total_balls_faced']) + match_stats['balls']
            df.at[idx, 'total_fours'] = safe_int(df.at[idx, 'total_fours']) + match_stats.get('fours', 0)
            df.at[idx, 'total_sixes'] = safe_int(df.at[idx, 'total_sixes']) + match_stats.get('sixes', 0)
            
            if match_stats.get('is_not_out', False):
                df.at[idx, 'NO'] = safe_int(df.at[idx, 'NO']) + 1
            
            # Recalculate Batting Avg and SR
            total_runs = float(df.at[idx, 'runs'])
            innings = float(df.at[idx, 'Innings'])
            no = safe_float(df.at[idx, 'NO'])
            balls = float(df.at[idx, 'total_balls_faced'])
            
            if (innings - no) > 0:
                df.at[idx, 'average'] = round(total_runs / (innings - no), 2)
            if balls > 0:
                df.at[idx, 'strike_rate'] = round((total_runs / balls) * 100, 2)
                
            # HS update
            current_hs = str(df.at[idx, 'HS']) if pd.notna(df.at[idx, 'HS']) else '0'
            # Remove '*' if present
            clean_hs = int(current_hs.replace('*', '')) if current_hs else 0
            if match_stats['runs'] > clean_hs:
                df.at[idx, 'HS'] = match_stats['runs']
        
        # Bowling updates
        if match_stats.get('overs', 0) > 0 or match_stats.get('wickets', 0) > 0:
            df.at[idx, 'wickets'] = safe_int(df.at[idx, 'wickets']) + match_stats['wickets']
            df.at[idx, 'total_runs_conceded'] = safe_int(df.at[idx, 'total_runs_conceded']) + match_stats['runs_conceded']
            df.at[idx, 'total_overs_bowled'] = safe_float(df.at[idx, 'total_overs_bowled']) + match_stats['overs']
            
            wkts = float(df.at[idx, 'wickets'])
            cons = float(df.at[idx, 'total_runs_conceded'])
            ovrs = float(df.at[idx, 'total_overs_bowled'])
            
            if wkts > 0:
                df.at[idx, 'bowling_average'] = round(cons / wkts, 2)
            if ovrs > 0:
                df.at[idx, 'economy'] = round(cons / ovrs, 2)
        
        # Fielding updates
        if match_stats.get('catches', 0) > 0:
            df.at[idx, 'total_catches'] = safe_int(df.at[idx, 'total_catches']) + match_stats['catches']
                
        # Save back
        df.to_csv(csv_path, index=False)
        return True, "CSV updated successfully"
    except Exception as e:
        import traceback
        print(f"CSV Sync Error: {traceback.format_exc()}")
        return False, str(e)

def update_batch_wc_csv_stats(match_data_list):
    """
    Update wc_players.csv for multiple players at once.
    match_data_list = [ {'player_name': name, 'match_stats': {stats}, 'team_name': team}, ... ]
    """
    try:
        from .config import DATA_PATHS
        csv_path = DATA_PATHS.get("wc_players")
        if not csv_path or not os.path.exists(csv_path):
            return False, "CSV file not found"
            
        df = pd.read_csv(csv_path)
        
        # Ensure helper columns exist
        required_cols = ['player', 'Team', 'Format', 'matches', 'Innings', 'NO', 'runs', 'wickets', 
                         'average', 'bowling_average', 'strike_rate', 'HS', '100s', '50s', 
                         'batting_position', 'image_url', 'role', 'economy',
                         'total_balls_faced', 'total_overs_bowled', 'total_runs_conceded', 
                         'total_fours', 'total_sixes', 'total_catches']
        
        for col in required_cols:
            if col not in df.columns:
                if col in ['player', 'Team', 'Format', 'role', 'HS', 'image_url']:
                    df[col] = ""
                else:
                    df[col] = 0

        updated_count = 0
        for entry in match_data_list:
            player_name = entry['player_name']
            match_stats = entry['match_stats']
            team_name = entry.get('team_name', 'Unknown')
            
            player_mask = df['player'].str.lower() == player_name.lower()
            
            if not player_mask.any():
                new_row = {col: (0 if df[col].dtype in ['int64', 'float64'] else "") for col in df.columns}
                new_row.update({
                    'player': player_name,
                    'Team': team_name,
                    'Format': 'T20',
                    'matches': 0,
                    'Innings': 0,
                    'runs': 0,
                    'wickets': 0,
                    'HS': '0'
                })
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                player_mask = df['player'] == player_name
            
            idx = df[player_mask].index[0]
            
            # Update basics
            df.at[idx, 'matches'] = safe_int(df.at[idx, 'matches']) + 1
            
            # Batting updates
            if match_stats.get('balls', 0) > 0 or match_stats.get('runs', 0) > 0:
                df.at[idx, 'Innings'] = safe_int(df.at[idx, 'Innings']) + 1
                df.at[idx, 'runs'] = safe_int(df.at[idx, 'runs']) + match_stats['runs']
                df.at[idx, 'total_balls_faced'] = safe_int(df.at[idx, 'total_balls_faced']) + match_stats['balls']
                df.at[idx, 'total_fours'] = safe_int(df.at[idx, 'total_fours']) + match_stats.get('fours', 0)
                df.at[idx, 'total_sixes'] = safe_int(df.at[idx, 'total_sixes']) + match_stats.get('sixes', 0)
                
                if match_stats.get('is_not_out', False):
                    df.at[idx, 'NO'] = safe_int(df.at[idx, 'NO']) + 1
                
                # Recalculate
                total_runs = float(df.at[idx, 'runs'])
                innings = float(df.at[idx, 'Innings'])
                no = safe_float(df.at[idx, 'NO'])
                balls = float(df.at[idx, 'total_balls_faced'])
                
                if (innings - no) > 0:
                    df.at[idx, 'average'] = round(total_runs / (innings - no), 2)
                if balls > 0:
                    df.at[idx, 'strike_rate'] = round((total_runs / balls) * 100, 2)
                
                current_hs = str(df.at[idx, 'HS']) if pd.notna(df.at[idx, 'HS']) else '0'
                clean_hs = int(current_hs.replace('*', '')) if current_hs else 0
                if match_stats['runs'] > clean_hs:
                    df.at[idx, 'HS'] = match_stats['runs']
            
            # Bowling updates
            if match_stats.get('overs', 0) > 0 or match_stats.get('wickets', 0) > 0:
                df.at[idx, 'wickets'] = safe_int(df.at[idx, 'wickets']) + match_stats['wickets']
                df.at[idx, 'total_runs_conceded'] = safe_int(df.at[idx, 'total_runs_conceded']) + match_stats['runs_conceded']
                df.at[idx, 'total_overs_bowled'] = safe_float(df.at[idx, 'total_overs_bowled']) + match_stats['overs']
                
                wkts = float(df.at[idx, 'wickets'])
                cons = float(df.at[idx, 'total_runs_conceded'])
                ovrs = float(df.at[idx, 'total_overs_bowled'])
                
                if wkts > 0:
                    df.at[idx, 'bowling_average'] = round(cons / wkts, 2)
                if ovrs > 0:
                    df.at[idx, 'economy'] = round(cons / ovrs, 2)
            
            # Fielding
            if match_stats.get('catches', 0) > 0:
                df.at[idx, 'total_catches'] = safe_int(df.at[idx, 'total_catches']) + match_stats['catches']
            
            updated_count += 1
                
        df.to_csv(csv_path, index=False)
        return True, f"Batch CSV update successful for {updated_count} players."
    except Exception as e:
        import traceback
        print(f"Batch CSV Sync Error: {traceback.format_exc()}")
        return False, str(e)

def safe_int(v):
    try: return int(v) if pd.notna(v) else 0
    except: return 0

def safe_float(v):
    try: return float(v) if pd.notna(v) else 0.0
    except: return 0.0

# ==================== PERFORMANCE TRACKING ====================

def add_player_performance(match_id, player_name, team_id, runs=0, balls_faced=0, fours=0, sixes=0, wickets=0, overs_bowled=0, runs_conceded=0, economy=0, catches=0, is_not_out=False):
    """Add player performance for a match with auto-calculated details"""
    try:
        # Debug Logging
        with open("db_debug.log", "a") as f:
            f.write(f"Attempting Add Perf: match={match_id}, player={player_name}, team={team_id}\n")
            f.write(f"Params: runs={runs}, balls={balls_faced}, wkts={wickets}, overs={overs_bowled}, catches={catches}\n")

        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Auto-calculate economy if not provided
        if economy == 0 and overs_bowled > 0:
            economy = round(runs_conceded / overs_bowled, 2)
            
        # Determine performance type (can be both)
        performance_type = []
        if balls_faced > 0 or runs > 0: performance_type.append('batsman')
        if overs_bowled > 0 or wickets > 0: performance_type.append('bowler')
        perf_type_str = ','.join(performance_type) if performance_type else 'all'
        
        cursor.execute("""
            INSERT INTO match_player_performance 
            (match_id, player_name, team_id, runs, balls_faced, fours, sixes, wickets, overs_bowled, runs_conceded, economy, catches, performance_type, is_not_out)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (match_id, player_name, team_id, runs, balls_faced, fours, sixes, wickets, overs_bowled, runs_conceded, economy, catches, perf_type_str, 1 if is_not_out else 0))
        
        conn.commit()
        conn.close()
        with open("db_debug.log", "a") as f: f.write("Success!\n")
        return True
    except Exception as e:
        with open("db_debug.log", "a") as f: f.write(f"Error: {e}\n")
        print(f"Error adding player performance: {e}")
        return False

def get_match_performances(match_id):
    """Get all performances for a match"""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    performances = conn.execute(
        "SELECT * FROM match_player_performance WHERE match_id = ? ORDER BY created_at DESC",
        (match_id,)
    ).fetchall()
    conn.close()
    return performances

def calculate_batsman_fantasy_points(runs, fours, sixes, balls_faced, is_captain=False, is_vice_captain=False):
    """Calculate fantasy points for batsman performance"""
    points = (runs * 1) + (fours * 1) + (sixes * 2)
    
    # Strike rate bonus
    if balls_faced >= 10:
        sr = (runs / balls_faced) * 100
        if sr > 170: points += 12
        elif sr > 150: points += 8
        elif sr > 130: points += 4

    # Milestones
    if runs >= 100: points += 16
    elif runs >= 50: points += 8
    elif runs >= 30: points += 4
    
    # Multipliers
    if is_captain: points *= 2
    elif is_vice_captain: points *= 1.5
    return points

def calculate_bowler_fantasy_points(wickets, economy, overs_bowled, is_captain=False, is_vice_captain=False):
    """Calculate fantasy points for bowling performance"""
    points = wickets * 25
    
    # Wicket hauls
    if wickets >= 5: points += 16
    elif wickets >= 4: points += 8
    elif wickets >= 3: points += 4
    
    # Economy bonus (min 2 overs)
    if overs_bowled >= 2:
        if economy < 5: points += 12
        elif economy < 7: points += 8
        elif economy < 9: points += 4
            
    # Multipliers
    if is_captain: points *= 2
    elif is_vice_captain: points *= 1.5
    return points

def calculate_fielding_fantasy_points(catches, is_captain=False, is_vice_captain=False):
    """Calculate fantasy points for fielding performance"""
    points = catches * 8
    if is_captain: points *= 2
    elif is_vice_captain: points *= 1.5
    return points

def calculate_updated_fantasy_scores(tournament_id, match_id=None):
    """Recalculate fantasy scores based on player performance data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # If match_id is provided, only update for that match
        query = "SELECT * FROM fantasy_teams WHERE tournament_id = ?"
        params = [tournament_id]
        if match_id:
            query += " AND match_id = ?"
            params.append(match_id)
            
        fantasy_teams = conn.execute(query, params).fetchall()
        
        for team in fantasy_teams:
            total_points = 0
            team_data = json.loads(team['players_json'])
            selected_players = team_data.get('players', [])
            captain = team_data.get('captain', '')
            vice_captain = team_data.get('vice_captain', '')
            
            # Get performances for this match
            performances = conn.execute(
                "SELECT * FROM match_player_performance WHERE match_id = ?",
                (team['match_id'],)
            ).fetchall()
            
            for perf in performances:
                if perf['player_name'] in selected_players:
                    is_captain = (perf['player_name'] == captain)
                    is_vice_captain = (perf['player_name'] == vice_captain)
                    
                    points = 0
                    if perf['performance_type'] == 'batsman' or perf['runs'] > 0:
                        points += calculate_batsman_fantasy_points(
                            perf['runs'], perf['fours'], perf['sixes'], 
                            perf['balls_faced'], is_captain, is_vice_captain
                        )
                    
                    if perf['wickets'] > 0 or perf['overs_bowled'] > 0:
                        points += calculate_bowler_fantasy_points(
                            perf['wickets'], perf['economy'], perf['overs_bowled'],
                            is_captain, is_vice_captain
                        )

                    if perf['catches'] > 0:
                        points += calculate_fielding_fantasy_points(
                            perf['catches'], is_captain, is_vice_captain
                        )
                    
                    total_points += points
            
            # Update score
            cursor.execute(
                "INSERT OR REPLACE INTO fantasy_scores (fantasy_team_id, total_score, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (team['id'], total_points)
            )
            
            # Update user leaderboard total
            user_id = team['user_id']
            user_total = conn.execute(
                "SELECT SUM(total_score) as total FROM fantasy_scores fs "
                "JOIN fantasy_teams ft ON fs.fantasy_team_id = ft.id "
                "WHERE ft.user_id = ? AND ft.tournament_id = ?",
                (user_id, tournament_id)
            ).fetchone()['total'] or 0
            
            cursor.execute(
                "INSERT OR REPLACE INTO leaderboard (user_id, tournament_id, total_points, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (user_id, tournament_id, user_total)
            )
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating fantasy scores: {e}")
        if 'conn' in locals(): conn.close()
        return False

def get_tournament_stats(tournament_id, stat_type='runs'):
    """Get aggregated top performers for a specific tournament"""
    conn = get_db_connection()
    try:
        query = ""
        if stat_type == 'runs':
            query = """
                SELECT player_name, tt.team_name, SUM(runs) as total_runs, 
                       AVG(CAST(runs AS FLOAT) / NULLIF(balls_faced, 0) * 100) as avg_sr,
                       COUNT(mpp.id) as matches
                FROM match_player_performance mpp
                JOIN tournament_teams tt ON mpp.team_id = tt.id
                WHERE tt.tournament_id = ?
                GROUP BY player_name
                ORDER BY total_runs DESC LIMIT 10
            """
        elif stat_type == 'wickets':
            query = """
                SELECT player_name, tt.team_name, SUM(wickets) as total_wickets, 
                       AVG(economy) as avg_eco,
                       COUNT(mpp.id) as matches
                FROM match_player_performance mpp
                JOIN tournament_teams tt ON mpp.team_id = tt.id
                WHERE tt.tournament_id = ?
                GROUP BY player_name
                ORDER BY total_wickets DESC LIMIT 10
            """
        elif stat_type == 'sixes':
            query = """
                SELECT player_name, tt.team_name, SUM(sixes) as total_sixes
                FROM match_player_performance mpp
                JOIN tournament_teams tt ON mpp.team_id = tt.id
                WHERE tt.tournament_id = ?
                GROUP BY player_name
                ORDER BY total_sixes DESC LIMIT 10
            """
        elif stat_type == 'catches':
            query = """
                SELECT player_name, tt.team_name, SUM(catches) as total_catches
                FROM match_player_performance mpp
                JOIN tournament_teams tt ON mpp.team_id = tt.id
                WHERE tt.tournament_id = ?
                GROUP BY player_name
                ORDER BY total_catches DESC LIMIT 10
            """
            
        if not query: return pd.DataFrame()
            
        df = pd.read_sql_query(query, conn, params=(tournament_id,))
        return df
    except Exception as e:
        print(f"Error fetching tournament stats: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def promote_to_super8(tournament_id):
    """Automatically promote top 2 teams from each group to Super 8 groups"""
    try:
        # 1. Get current standings for A, B, C, D
        qualified = []
        for group in ['A', 'B', 'C', 'D']:
            standings = get_group_standings_with_nrr(tournament_id, group)
            if len(standings) < 2:
                return False, f"Group {group} does not have enough completed matches."
            qualified.append((group, standings[0], standings[1]))
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 2. Assign to Super 8 Groups (Group 1 and Group 2)
        # Seed logic: A1, B2, C1, D2 -> Group 1 | B1, A2, D1, C2 -> Group 2
        s8_group1 = [qualified[0][1], qualified[1][2], qualified[2][1], qualified[3][2]]
        s8_group2 = [qualified[1][1], qualified[0][2], qualified[3][1], qualified[2][2]]
        
        # 3. Create Super 8 matches (round robin within groups)
        def create_s8_RR(teams, s8_group_name):
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    # We create them as 'super8' stage
                    cursor.execute("""
                        INSERT INTO tournament_matches (tournament_id, team1_id, team2_id, stage, group_letter, status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (tournament_id, t1['team_id'], t2['team_id'], 'super8', s8_group_name, 'scheduled'))
        
        create_s8_RR(s8_group1, "1")
        create_s8_RR(s8_group2, "2")
        
        # 4. Update tournament status
        cursor.execute("UPDATE tournaments SET status = 'super8' WHERE id = ?", (tournament_id,))
        
        conn.commit()
        conn.close()
        return True, "Successfully promoted teams and scheduled Super 8 matches!"
    except Exception as e:
        print(f"Error promoting to Super 8: {e}")
        return False, str(e)

# ==================== AI TEAM STRENGTH ANALYSIS ====================

def get_player_stats(player_name):
    """Get player career stats from players table"""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    player = conn.execute(
        "SELECT * FROM players WHERE name = ? LIMIT 1",
        (player_name,)
    ).fetchone()
    conn.close()
    return player

def calculate_team_strength(team_players, tournament_id):
    """Calculate AI team strength (0-100) based on player selection"""
    try:
        if not team_players:
            return 0
        
        total_strength = 0
        player_count = len(team_players)
        
        # Player role weights
        role_weights = {
            'batsman': 1.0,
            'bowler': 1.0,
            'all-rounder': 1.3,  # All-rounders are more valuable
            'wicket-keeper': 1.1
        }
        
        batsmen_count = 0
        bowlers_count = 0
        all_rounders_count = 0
        
        for player_name in team_players:
            player = get_player_stats(player_name)
            
            if player:
                player_strength = 0
                
                # Get player role
                role = player.get('role', 'batsman').lower()
                weight = role_weights.get(role, 1.0)
                
                # Base strength from stats
                avg = float(player.get('average', 0)) or 0
                sr = float(player.get('strike_rate', 100)) or 100
                
                # Normalize stats
                avg_strength = min(avg / 60 * 100, 100)  # 60 is considered excellent average
                sr_strength = min(sr / 150 * 100, 100)   # 150 is considered excellent SR
                
                # Combine for batting strength
                if role in ['batsman', 'all-rounder', 'wicket-keeper']:
                    player_strength = (avg_strength * 0.6 + sr_strength * 0.4) * weight
                
                # Bowling stats
                if role in ['bowler', 'all-rounder']:
                    bowling_avg = float(player.get('bowling_average', 0)) or 0
                    economy = float(player.get('economy', 0)) or 0
                    
                    # Normalize bowling
                    bowling_strength = min((40 - bowling_avg) / 25 * 100, 100) if bowling_avg > 0 else 50
                    economy_strength = min((10 - economy) / 6 * 100, 100) if economy > 0 else 50
                    
                    bowling_component = (bowling_strength * 0.5 + economy_strength * 0.5) * weight
                    
                    if role == 'all-rounder':
                        player_strength = (player_strength * 0.5 + bowling_component * 0.5)
                    else:
                        player_strength = bowling_component
                
                total_strength += player_strength
                
                # Count player types for balance bonus
                if role == 'all-rounder':
                    all_rounders_count += 1
                elif role == 'batsman':
                    batsmen_count += 1
                else:
                    bowlers_count += 1
        
        # Calculate average strength
        avg_team_strength = total_strength / player_count
        
        # Balance bonus (well-rounded teams get bonus)
        if batsmen_count > 0 and bowlers_count > 0 and all_rounders_count > 0:
            balance_bonus = 10
        elif batsmen_count > 0 and bowlers_count > 0:
            balance_bonus = 5
        else:
            balance_bonus = 0
        
        # Final strength (0-100 scale)
        team_strength = min(avg_team_strength + balance_bonus, 100)
        
        return round(team_strength, 1)
    except Exception as e:
        print(f"Error calculating team strength: {e}")
        return 0

def get_team_strength_rating(tournament_id, team_id):
    """Get strength rating for a tournament team"""
    try:
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        
        # Get team players
        team = conn.execute(
            "SELECT * FROM tournament_teams WHERE id = ?",
            (team_id,)
        ).fetchone()
        conn.close()
        
        if team:
            players = team['players'].split(',') if team['players'] else []
            strength = calculate_team_strength(players, tournament_id)
            return strength
        
        return 0
    except Exception as e:
        print(f"Error getting team strength: {e}")
        return 0

def populate_csv_with_all_squad_players(tournament_id):
    """Ensure all players in tournament squads are present in wc_players.csv"""
    try:
        from .config import DATA_PATHS
        csv_path = DATA_PATHS.get("wc_players")
        if not csv_path or not os.path.exists(csv_path):
            return False, "CSV file not found"
            
        df = pd.read_csv(csv_path)
        
        # Ensure helper columns exist
        required_cols = ['player', 'Team', 'Format', 'matches', 'Innings', 'NO', 'runs', 'wickets', 
                         'average', 'bowling_average', 'strike_rate', 'HS', '100s', '50s', 
                         'batting_position', 'image_url', 'role', 'economy',
                         'total_balls_faced', 'total_overs_bowled', 'total_runs_conceded', 
                         'total_fours', 'total_sixes', 'total_catches']
        
        for col in required_cols:
            if col not in df.columns:
                if col in ['player', 'Team', 'Format', 'role', 'HS', 'image_url']:
                    df[col] = ""
                else:
                    df[col] = 0

        teams = get_tournament_teams(tournament_id)
        added_count = 0
        
        for team in teams:
            team_name = team['team_name']
            squad_data = team['squad'] 
            if not squad_data: continue
            
            # Handle both JSON list and legacy comma-separated string
            try:
                import json
                players = json.loads(squad_data)
                if not isinstance(players, list): players = [players]
            except:
                players = [p.strip() for p in squad_data.split(',') if p.strip()]
            
            for p_name in players:
                # Check if exists (case insensitive)
                if not (df['player'].str.lower() == p_name.lower()).any():
                    # Add new player row
                    new_row = {col: (0 if df[col].dtype in ['int64', 'float64'] else "") for col in df.columns}
                    new_row.update({
                        'player': p_name,
                        'Team': team_name,
                        'Format': 'T20',
                        'matches': 0,
                        'Innings': 0,
                        'runs': 0,
                        'wickets': 0,
                        'HS': '0'
                    })
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    added_count += 1
        
        if added_count > 0:
            df.to_csv(csv_path, index=False)
            
        return True, f"Successfully processed squads. Added {added_count} new players to CSV."
    except Exception as e:
        import traceback
        print(f"Bulk CSV Pop Error: {traceback.format_exc()}")
        return False, str(e)