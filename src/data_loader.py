import pandas as pd
import streamlit as st
import os
from .config import DATA_PATHS
from .database import fetch_all_players_from_db

# Store last known mtimes
_last_mtimes = {}

def _check_csv_changes():
    """Check if any CSV files have been modified and clear cache if so."""
    global _last_mtimes
    
    csv_files = [
        'odi_batsman.csv',
        'odi_bowler.csv',
        'odi_all_rounders.csv',
        'yearwise_data.csv'
    ]
    
    current_mtimes = {}
    cache_invalidated = False
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            mtime = os.path.getmtime(csv_file)
            current_mtimes[csv_file] = mtime
            
            # Check if this file was modified
            if csv_file in _last_mtimes and _last_mtimes[csv_file] != mtime:
                print(f"🔄 CSV changed detected: {csv_file}")
                cache_invalidated = True
            
            _last_mtimes[csv_file] = mtime
    
    if cache_invalidated:
        st.cache_data.clear()
    
    return cache_invalidated

@st.cache_data
def load_all_data():
    """Load, merge, and preprocess all cricket data files. Auto-invalidates when CSV files change."""
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
