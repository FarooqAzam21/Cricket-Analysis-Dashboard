from src.database import init_db, save_to_db
from src.config import DATA_PATHS
import pandas as pd
import os

def migrate():
    print("🔄 Syncing CSV changes to database...")
    
    # 1. Initialize database schema (creates tables if they don't exist, preserves existing data)
    init_db()
    
    # 2. Load DIRECTLY from CSV files (bypass database check)
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

    # 3. Save to database (INSERT OR REPLACE will update existing records)
    save_to_db(all_players)
    
    print(f"✅ Sync successful! {len(all_players)} player records updated in database.")
    print("ℹ️  User accounts preserved.")

if __name__ == "__main__":
    migrate()
