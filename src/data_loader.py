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
            # Load CSVs
            df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
            df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
            df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()
            
            print(f"DEBUG CSV Load: batsman={len(df_bat)}, all_rounder={len(df_ar)}, bowler={len(df_bowl)}")
            
            # Simple cleaning function
            def clean(df, default_role=''):
                if df.empty: 
                    return df
                df = df.copy()
                df.columns = [col.strip() for col in df.columns]
                
                # Clean text columns
                text_cols = ['player', 'Team', 'Format', 'role']
                for c in text_cols:
                    if c in df.columns:
                        df[c] = (df[c]
                                .astype(str)
                                .str.strip()
                                .str.replace(r'[\t\r\n"\']', '', regex=True)
                                .str.replace(r'\s+', ' ', regex=True)
                                .str.strip())
                
                # Ensure role column exists
                if 'role' not in df.columns:
                    df['role'] = default_role
                    
                return df
            
            # Clean each with appropriate role
            df_bat = clean(df_bat, 'Batsman')
            df_ar = clean(df_ar, 'All-rounder')
            df_bowl = clean(df_bowl, 'Bowler')
            
            print(f"DEBUG After clean: batsman roles={df_bat['role'].unique()[:3] if len(df_bat)>0 else []}")
            print(f"DEBUG After clean: bowler roles={df_bowl['role'].unique()[:3] if len(df_bowl)>0 else []}")
            
            # Combine - use pandas concat with all columns
            dfs = [df for df in [df_bat, df_ar, df_bowl] if not df.empty]
            
            if dfs:
                # Use simple concat - all columns will be preserved
                all_players = pd.concat(dfs, ignore_index=True, sort=False)
                # Remove exact duplicates (same player, team, format)
                all_players = all_players.drop_duplicates(subset=['player', 'Team', 'Format'], keep='first')
                all_players = all_players.reset_index(drop=True)
                
                print(f"DEBUG After concat: {len(all_players)} players")
                print(f"DEBUG Sample data - runs: {all_players['runs'].head(3).tolist()}")
                print(f"DEBUG Sample data - wickets: {all_players['wickets'].head(3).tolist()}")
                print(f"DEBUG Sample data - roles: {all_players['role'].head(3).tolist()}")
            else:
                all_players = pd.DataFrame()
        except Exception as csv_e:
            print(f"CSV load error: {csv_e}")
            import traceback
            traceback.print_exc()

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
                # Replace '-' with 0 before converting
                all_players[col] = all_players[col].astype(str).str.replace('-', '0', regex=False).str.strip()
                all_players[col] = pd.to_numeric(all_players[col], errors='coerce').fillna(0)

        print(f"DEBUG: Sample runs after conversion: {all_players['runs'].head(5).tolist()}")
        print(f"DEBUG: Sample wickets after conversion: {all_players['wickets'].head(5).tolist()}")
        
        # Ensure 'role' column exists, if not create it as empty
        if 'role' not in all_players.columns:
            print("WARNING: 'role' column missing, creating empty role column")
            all_players['role'] = ''
        
        # Create role_lower for filtering with proper handling of NaN/None
        all_players['role_lower'] = all_players['role'].fillna('').astype(str).str.lower()
        
        print(f"DEBUG: Total players: {len(all_players)}")
        print(f"DEBUG: Unique roles (first 10): {list(all_players['role_lower'].unique()[:10])}")
        print(f"DEBUG: Non-empty roles: {(all_players['role_lower'].str.len() > 0).sum()}")
        print(f"DEBUG: Role value counts:\n{all_players['role_lower'].value_counts().head(10)}")
        
        # Classify players by role - be more lenient with classification
        batsmen = all_players[all_players['role_lower'].str.contains('batsman|batter', na=False, regex=False)]
        wicket_keepers = all_players[all_players['role_lower'].str.contains('wicket|keeper', na=False, regex=False)]
        all_rounders = all_players[all_players['role_lower'].str.contains('all-rounder|all rounder|allrounder|fast-bowling|spinning|spinner|arm', na=False)]
        bowlers_data = all_players[all_players['role_lower'].str.contains('bowler|spinner|fast|seam|pace', na=False) | (all_players['wickets'] > 0)]
        
        print(f"DEBUG: Primary classification - Batsmen: {len(batsmen)}, Bowlers: {len(bowlers_data)}")
        
        # If no batsmen found, classify by positive runs as fallback
        if len(batsmen) == 0 and 'runs' in all_players.columns:
            batsmen = all_players[all_players['runs'] > 0]
            print(f"DEBUG: Using runs-based fallback for batsmen, found {len(batsmen)}")
        
        # If no bowlers found, classify by positive wickets as fallback
        if len(bowlers_data) == 0 and 'wickets' in all_players.columns:
            bowlers_data = all_players[all_players['wickets'] > 0]
            print(f"DEBUG: Using wickets-based fallback for bowlers, found {len(bowlers_data)}")
        
        # Ultimate fallback: if STILL no data, use all players
        if len(batsmen) == 0:
            batsmen = all_players.copy()
            print(f"DEBUG: ULTIMATE FALLBACK - using all {len(batsmen)} players as batsmen")
        
        if len(bowlers_data) == 0:
            bowlers_data = all_players.copy()
            print(f"DEBUG: ULTIMATE FALLBACK - using all {len(bowlers_data)} players as bowlers")
        
        print(f"DEBUG: Final - Batsmen: {len(batsmen)}, Bowlers: {len(bowlers_data)}, All-rounders: {len(all_rounders)}, WK: {len(wicket_keepers)}")
        
        return all_players, batsmen, all_rounders, bowlers_data, df_year, batsmen, all_rounders, wicket_keepers
    
    return None, None, None, None, None, None, None, None
