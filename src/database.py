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
            economy REAL DEFAULT 0,
            performance_type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(match_id) REFERENCES tournament_matches(id),
            FOREIGN KEY(team_id) REFERENCES tournament_teams(id)
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
    
    # Calculate fantasy points for all users who created teams for this match
    calculate_and_save_fantasy_points(match_id)

def calculate_and_save_fantasy_points(match_id):
    """Calculate fantasy points for all teams created for this match"""
    try:
        conn = get_db_connection()
        
        # Get all fantasy teams for this match
        fantasy_teams = conn.execute(
            "SELECT id, players_json FROM fantasy_teams WHERE match_id = ?",
            (match_id,)
        ).fetchall()
        
        if not fantasy_teams:
            conn.close()
            return
        
        # Get match details
        match = conn.execute(
            "SELECT * FROM tournament_matches WHERE id = ?",
            (match_id,)
        ).fetchone()
        
        # Simple scoring: Award points based on winning team players
        winning_team_id = match['winner_id']
        points_per_winning_player = 25  # 25 points for each player in winning team
        
        for fantasy_team in fantasy_teams:
            players_data = json.loads(fantasy_team['players_json'])
            players = players_data.get('players', [])
            captain = players_data.get('captain', '')
            vice_captain = players_data.get('vice_captain', '')
            
            # Get all players from winning team
            winning_team = conn.execute(
                "SELECT squad FROM tournament_teams WHERE id = ?",
                (winning_team_id,)
            ).fetchone()
            
            winning_squad = []
            if winning_team and winning_team['squad']:
                try:
                    winning_squad = json.loads(winning_team['squad'])
                except:
                    winning_squad = []
            
            # Calculate points
            total_points = 0
            
            for player in players:
                if player in winning_squad:
                    points = points_per_winning_player
                    
                    # Captain gets 2x, Vice-Captain gets 1.5x
                    if player == captain:
                        points = int(points * 2)
                    elif player == vice_captain:
                        points = int(points * 1.5)
                    
                    total_points += points
            
            # Save or update fantasy score
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM fantasy_scores WHERE fantasy_team_id = ?",
                (fantasy_team['id'],)
            )
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute(
                    "UPDATE fantasy_scores SET total_score = ?, updated_at = CURRENT_TIMESTAMP WHERE fantasy_team_id = ?",
                    (total_points, fantasy_team['id'])
                )
            else:
                cursor.execute(
                    "INSERT INTO fantasy_scores (fantasy_team_id, total_score) VALUES (?, ?)",
                    (fantasy_team['id'], total_points)
                )
            
            # Update user leaderboard
            user_id = conn.execute(
                "SELECT user_id FROM fantasy_teams WHERE id = ?",
                (fantasy_team['id'],)
            ).fetchone()['user_id']
            
            tournament_id = match['tournament_id']
            
            # Get total points for user in this tournament
            total_user_points = conn.execute(
                "SELECT COALESCE(SUM(fs.total_score), 0) as total FROM fantasy_scores fs "
                "JOIN fantasy_teams ft ON fs.fantasy_team_id = ft.id "
                "WHERE ft.user_id = ? AND ft.tournament_id = ?",
                (user_id, tournament_id)
            ).fetchone()['total']
            
            cursor.execute(
                "INSERT OR REPLACE INTO leaderboard (user_id, tournament_id, total_points, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (user_id, tournament_id, total_user_points)
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error calculating fantasy points: {e}")
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
    
    runs_for = 0
    runs_against = 0
    overs_played = 0
    overs_faced = 0
    
    for match in matches:
        if match['team1_id'] == team_id:
            runs_for += match['team1_score']
            runs_against += match['team2_score']
        else:
            runs_for += match['team2_score']
            runs_against += match['team1_score']
        
        # Assume 20 overs per T20 match
        overs_played += 20
        overs_faced += 20
    
    conn.close()
    
    # Calculate NRR
    if overs_played == 0:
        return 0.0
    
    run_rate_for = runs_for / (overs_played / 6)  # Convert overs to decimal
    run_rate_against = runs_against / (overs_faced / 6)
    nrr = run_rate_for - run_rate_against
    
    return round(nrr, 2)

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
    
    if tournament and tournament['tournament_format']:
        try:
            return json.loads(tournament['tournament_format'])
        except:
            return ['group', 'semi-final', 'final']
    
    return ['group', 'semi-final', 'final']

# ==================== PERFORMANCE TRACKING ====================

def add_player_performance(match_id, player_name, team_id, runs, balls_faced, fours, sixes, wickets=0, economy=0):
    """Add player performance for a match"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Determine performance type
        if balls_faced > 0:
            performance_type = 'batsman'
        else:
            performance_type = 'bowler'
        
        cursor.execute("""
            INSERT INTO match_player_performance 
            (match_id, player_name, team_id, runs, balls_faced, fours, sixes, wickets, economy, performance_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (match_id, player_name, team_id, runs, balls_faced, fours, sixes, wickets, economy, performance_type))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
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
    points = 0
    
    # Runs - 1 point per run
    points += runs * 1
    
    # Fours - bonus points
    points += fours * 1
    
    # Sixes - double bonus
    points += sixes * 2
    
    # Strike rate bonus (if SR > 120)
    if balls_faced > 0:
        sr = (runs / balls_faced) * 100
        if sr > 150:
            points += 10
        elif sr > 120:
            points += 5
    
    # Runs bonus
    if runs >= 50:
        points += 50 if runs >= 50 else 0
    if runs >= 100:
        points += 100
    
    # Captain/Vice Captain multiplier
    if is_captain:
        points *= 2
    elif is_vice_captain:
        points *= 1.5
    
    return points

def calculate_bowler_fantasy_points(wickets, economy, is_captain=False, is_vice_captain=False):
    """Calculate fantasy points for bowler performance"""
    points = 0
    
    # Wickets - points
    points += wickets * 25
    
    # Economy bonus (better economy = more points)
    if economy < 6:
        points += 10
    elif economy < 8:
        points += 5
    
    # Captain/Vice Captain multiplier
    if is_captain:
        points *= 2
    elif is_vice_captain:
        points *= 1.5
    
    return points

def calculate_updated_fantasy_scores(tournament_id):
    """Recalculate fantasy scores based on performance data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        conn.row_factory = sqlite3.Row
        
        # Get all fantasy teams for this tournament
        fantasy_teams = conn.execute("""
            SELECT * FROM fantasy_teams WHERE tournament_id = ?
        """, (tournament_id,)).fetchall()
        
        for team in fantasy_teams:
            total_points = 0
            team_id = team['team_id']
            players = team['team_composition'].split(',')
            
            # Get matches for this team in tournament
            matches = conn.execute("""
                SELECT * FROM tournament_matches 
                WHERE tournament_id = ? AND (team1_id = ? OR team2_id = ?)
                AND match_status = 'completed'
            """, (tournament_id, team_id, team_id)).fetchall()
            
            for match in matches:
                match_performances = conn.execute("""
                    SELECT * FROM match_player_performance WHERE match_id = ?
                """, (match['id'],)).fetchall()
                
                for performance in match_performances:
                    if performance['player_name'] in players:
                        if performance['performance_type'] == 'batsman':
                            points = calculate_batsman_fantasy_points(
                                performance['runs'],
                                performance['fours'],
                                performance['sixes'],
                                performance['balls_faced']
                            )
                        else:
                            points = calculate_bowler_fantasy_points(
                                performance['wickets'],
                                performance['economy']
                            )
                        
                        total_points += points
            
            # Update fantasy scores
            cursor.execute("""
                UPDATE fantasy_scores SET total_points = ? 
                WHERE fantasy_team_id = ?
            """, (total_points, team['id']))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating fantasy scores: {e}")
        return False

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