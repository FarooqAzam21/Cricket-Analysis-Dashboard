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
    
    conn.commit()
    conn.close()

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
