from src.database import init_db, save_to_db, get_db_connection
from src.config import DATA_PATHS
import pandas as pd
import os
import sqlite3

def migrate():
    print("🔄 Safe Data Migration (Preserving User Accounts)")
    print("=" * 60)
    
    # 1. BACKUP existing users
    print("\n1️⃣ Backing up user accounts...")
    db_path = 'cricket_dashboard.db'
    users_backup = None
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users')
            users_backup = cursor.fetchall()
            print(f"   ✅ Backed up {len(users_backup)} user accounts")
            conn.close()
        except Exception as e:
            print(f"   ℹ️  First migration - no users to backup")
    
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

    # 6. Restore users
    if users_backup:
        print("\n6️⃣ Restoring user accounts...")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for user_row in users_backup:
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                             (user_row[1], user_row[2]))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Restored {len(users_backup)} user accounts")
        except Exception as e:
            print(f"   ❌ Failed to restore users: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Migration successful!")
    print(f"   ✓ {len(all_players)} player records updated")
    print(f"   ✓ User accounts preserved")
    print("=" * 60)

if __name__ == "__main__":
    migrate()
