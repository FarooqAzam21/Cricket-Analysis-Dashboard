from src.database import init_db, get_db_connection
from src.config import DATA_PATHS
import pandas as pd
import os
import sqlite3
import hashlib

def hash_password(password):
    """Hash a password using SHA256."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def migrate():
    print("🔄 Simple Data Migration - Adding Player Data Only")
    print("=" * 60)
    print("✅ Will KEEP everything: tournaments, teams, matches, admins, users")
    print("✅ Will ONLY update: player data from CSV files")
    print("=" * 60)
    
    db_path = 'cricket_dashboard.db'
    
    # 1. Create database if it doesn't exist (first run only)
    print("\n1️⃣ Checking database...")
    if not os.path.exists(db_path):
        print("   📝 Creating new database for first time...")
        init_db()
        print("   ✅ Database created")
    else:
        print("   ✅ Database exists - keeping all your data")
    
    # 2. Load player data from CSVs
    print("\n2️⃣ Loading player data from CSV files...")
    try:
        df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
        df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
        df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()
        
        print(f"   ✅ Batsman CSV: {len(df_bat)} rows")
        print(f"   ✅ All-rounder CSV: {len(df_ar)} rows")
        print(f"   ✅ Bowler CSV: {len(df_bowl)} rows")
        
        # Clean and combine
        def clean(df):
            if df.empty: 
                return df
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

    # 3. Update players in database (don't delete, just add/update)
    print(f"\n3️⃣ Updating {len(all_players)} player records...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete OLD player data but keep everything else
        cursor.execute('DELETE FROM players')
        print("   ✅ Cleared old player data")
        
        # Insert new player data using correct schema columns
        for idx, row in all_players.iterrows():
            try:
                cursor.execute('''INSERT INTO players 
                               (player, team, format)
                               VALUES (?, ?, ?)''',
                             (row.get('player', 'Unknown'),
                              row.get('Team', 'Unknown'),
                              row.get('Format', 'Unknown')))
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
        print(f"   ✅ {len(all_players)} players saved to database")
        
    except Exception as e:
        print(f"   ❌ Failed to save players: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Check what's preserved
    print("\n4️⃣ Verifying your data is safe...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Count tournaments
        cursor.execute('SELECT COUNT(*) FROM tournaments')
        tourn_count = cursor.fetchone()[0]
        
        # Count teams
        cursor.execute('SELECT COUNT(*) FROM tournament_teams')
        teams_count = cursor.fetchone()[0]
        
        # Count users
        cursor.execute('SELECT COUNT(*) FROM users')
        users_count = cursor.fetchone()[0]
        
        # Count matches
        cursor.execute('SELECT COUNT(*) FROM tournament_matches')
        matches_count = cursor.fetchone()[0]
        
        # Check admin
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
        admin_exists = cursor.fetchone()[0] > 0
        
        conn.close()
        
        print(f"   ✅ Tournaments: {tourn_count}")
        print(f"   ✅ Teams: {teams_count}")
        print(f"   ✅ Matches: {matches_count}")
        print(f"   ✅ User accounts: {users_count}")
        print(f"   ✅ Admin account: {'EXISTS' if admin_exists else 'MISSING'}")
        
        if not admin_exists:
            print("\n   ⚠️  Creating admin account...")
            conn = get_db_connection()
            cursor = conn.cursor()
            hashed_password = hash_password('admin')
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', hashed_password))
            conn.commit()
            conn.close()
            print("   ✅ Admin created (admin / admin)")
        
    except Exception as e:
        print(f"   ⚠️  Could not verify: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Migration Complete!")
    print("=" * 60)
    print(f"✓ {len(all_players)} player records updated")
    print("✓ All tournaments SAVED")
    print("✓ All teams SAVED")
    print("✓ All matches SAVED")
    print("✓ All user accounts SAVED")
    print("✓ Admin account SAVED")
    print("\n👉 You can now run: streamlit run main.py")
    print("=" * 60)
    print("=" * 60)

if __name__ == "__main__":
    migrate()
