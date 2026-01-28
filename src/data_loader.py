import pandas as pd
import streamlit as st
import os
import hashlib
from .config import DATA_PATHS
from .database import fetch_all_players_from_db

def _get_csv_cache_key():
    """Get cache key based on CSV file content hash.
    Works on both local and Streamlit Cloud deployments.
    Cache invalidates when ANY CSV file changes."""
    csv_files = [
        'odi_batsman.csv',
        'odi_bowler.csv',
        'odi_all_rounders.csv',
        'yearwise_data.csv'
    ]
    
    hash_obj = hashlib.md5()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                with open(csv_file, 'rb') as f:
                    # Read file in chunks to handle large files
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_obj.update(chunk)
            except Exception as e:
                print(f"Warning: Could not hash {csv_file}: {e}")
    
    return hash_obj.hexdigest()

@st.cache_data
def load_all_data(_csv_cache_key=None):
    """Load, merge, and preprocess all cricket data files. 
    Cache invalidates when CSV files change (via _csv_cache_key parameter).
    Uses content-based hashing for reliable cache invalidation everywhere."""
    all_players = pd.DataFrame()
    
    # 1. Try DB first
    try:
        all_players = fetch_all_players_from_db()
        if not all_players.empty:
            # Map DB naming back to what UI expects
            db_to_ui = {
                'team': 'Team',
                'format': 'Format',
                'innings': 'Innings',
                'no': 'NO',
                'hundreds': '100s',
                'fifties': '50s'
            }
            all_players = all_players.rename(columns=db_to_ui)
    except Exception as e:
        print(f"DB Fetch failed, falling back to CSV: {e}")

    # 2. If DB empty or missing, load from CSVs
    if all_players.empty:
        try:
            df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
            df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
            df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()
            
            # Clean and combine (re-using the logic we had before)
            def clean(df):
                if df.empty: return df
                df.columns = df.columns.map(str).str.strip()
                for c in ['player', 'Team', 'Format', 'role']:
                    if c in df.columns: df[c] = df[c].astype(str).str.replace(r'[\t"\']', '', regex=True).str.strip()
                return df

            composite = pd.concat([clean(df_bat), clean(df_ar), clean(df_bowl)], ignore_index=True, sort=False)
            # Minimal aggregation for the migration phase
            all_players = composite.groupby(['player', 'Team', 'Format'], as_index=False).first()
        except Exception as csv_e:
            print(f"CSV fallback also failed: {csv_e}")

    # Load yearwise separately
    try:
        df_year = pd.read_csv(DATA_PATHS["yearwise"])
        if not df_year.empty:
            # Ensure 'player' column exists and clean it
            if 'player' in df_year.columns:
                df_year['player'] = df_year['player'].astype(str).str.strip()
            # Convert numeric columns
            numeric_cols = ['year', 'matches', 'innings', 'runs', 'average', 'SR', '100s', '50s']
            for col in numeric_cols:
                if col in df_year.columns:
                    df_year[col] = pd.to_numeric(df_year[col], errors='coerce')
    except Exception as year_error:
        print(f"Year-wise data loading error: {year_error}")
        df_year = pd.DataFrame()

    # Classification logic
    if not all_players.empty:
        # Ensure numeric for filtering and AI models
        numeric_cols = ['wickets', 'runs', 'average', 'strike_rate', 'Innings', 'bowling_average', 'economy', 'matches']
        for col in numeric_cols:
            if col in all_players.columns:
                all_players[col] = pd.to_numeric(all_players[col], errors='coerce').fillna(0)

        all_players['role_lower'] = all_players.get('role', '').astype(str).str.lower()
        batsmen = all_players[all_players['role_lower'].str.contains('batsman', na=False)]
        wicket_keepers = all_players[all_players['role_lower'].str.contains('wicket-keeper', na=False)]
        all_rounders = all_players[all_players['role_lower'].str.contains('all-rounder|fast-bowling|spinner|arm', na=False)]
        bowlers_data = all_players[all_players['role_lower'].str.contains('bowler|spinner|fast|arm', na=False) | (all_players.get('wickets', 0) > 0)]
        return all_players, batsmen, all_rounders, bowlers_data, df_year, batsmen, all_rounders, wicket_keepers
    
    return None, None, None, None, None, None, None, None
