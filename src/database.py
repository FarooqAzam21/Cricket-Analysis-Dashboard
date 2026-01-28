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
            stage TEXT,
            group_letter TEXT,
            status TEXT DEFAULT 'scheduled',
            winner_id INTEGER,
            team1_score INTEGER,
            team2_score INTEGER,
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY(team1_id) REFERENCES tournament_teams(id),
            FOREIGN KEY(team2_id) REFERENCES tournament_teams(id)
        )
    ''')
    
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
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            tournament_id INTEGER,
            total_points REAL DEFAULT 0,
            fantasy_teams_created INTEGER DEFAULT 0,
            rank INTEGER,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(tournament_id) REFERENCES tournaments(id)
        )
    ''')
    
    conn.commit()
    conn.close()

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
        cursor.execute("INSERT INTO tournaments (name, start_date, end_date) VALUES (?, ?, ?)",
                      (name, start_date, end_date))
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

def create_tournament_match(tournament_id, team1_id, team2_id, match_date, stage, group_letter=None):
    """Create a match in tournament"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tournament_matches (tournament_id, team1_id, team2_id, match_date, stage, group_letter, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (tournament_id, team1_id, team2_id, match_date, stage, group_letter, 'scheduled'))
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
    
    query += " ORDER BY match_date ASC"
    matches = conn.execute(query, params).fetchall()
    conn.close()
    return matches

def update_match_result(match_id, winner_id, team1_score, team2_score):
    """Update match result after completion"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Update match status
    cursor.execute("""
        UPDATE tournament_matches 
        SET status = ?, winner_id = ?, team1_score = ?, team2_score = ?
        WHERE id = ?
    """, ('completed', winner_id, team1_score, team2_score, match_id))
    
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