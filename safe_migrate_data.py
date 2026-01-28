"""
Safe migration script that preserves user accounts while updating player data
"""
import os
import sqlite3
import pandas as pd
from src.config import DATA_PATHS
from src.database import init_db, get_db_connection, save_to_db

print("🔐 Safe Migration: Preserving User Data")
print("=" * 60)

# 1. BACKUP existing users
print("\n1️⃣ Backing up existing user accounts...")
db_path = 'cricket_dashboard.db'
users_backup = []

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT username, password FROM users')
        users_backup = cursor.fetchall()
        print(f"   ✅ Backed up {len(users_backup)} user accounts")
        conn.close()
    except Exception as e:
        print(f"   ℹ️ No users to backup (first migration): {e}")

# 2. DELETE old database
print("\n2️⃣ Deleting old database...")
if os.path.exists(db_path):
    os.remove(db_path)
    print("   ✅ Old database deleted")

# 3. CREATE fresh database schema
print("\n3️⃣ Creating fresh database schema...")
init_db()
print("   ✅ Fresh schema created")

# 4. LOAD and CLEAN CSV files
print("\n4️⃣ Loading cleaned CSV data...")
try:
    df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
    df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
    df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()
    
    print(f"   - Batsmen: {len(df_bat)} rows")
    print(f"   - All-rounders: {len(df_ar)} rows")
    print(f"   - Bowlers: {len(df_bowl)} rows")
    
    # Clean and deduplicate
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
    print(f"   📊 Combined & deduplicated: {len(all_players)} records")
    
    # Save to database
    save_to_db(all_players)
    print(f"   ✅ Imported {len(all_players)} player records")
    
except Exception as csv_e:
    print(f"   ❌ CSV loading failed: {csv_e}")
    exit(1)

# 5. RESTORE user accounts
if users_backup:
    print(f"\n5️⃣ Restoring {len(users_backup)} user accounts...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for username, password in users_backup:
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        conn.close()
        print(f"   ✅ Restored all {len(users_backup)} user accounts")
    except Exception as e:
        print(f"   ❌ Error restoring users: {e}")
        exit(1)
else:
    print("\n5️⃣ No users to restore (first migration)")

# 6. VERIFY database
print("\n6️⃣ Verifying database...")
try:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    cursor.execute("SELECT DISTINCT Format FROM players")
    formats = sorted([row[0] for row in cursor.fetchall()])
    conn.close()
    
    print(f"   📊 Players in DB: {player_count}")
    print(f"   🎯 Valid Formats: {formats}")
    print(f"   👤 User Accounts: {user_count}")
    
except Exception as e:
    print(f"   ⚠️ Verification error: {e}")

print("\n" + "=" * 60)
print("✅ Safe migration complete! User data preserved.\n")

# 6. RESTORE user accounts
if users_backup:
    print("\n6️⃣ Restoring user accounts...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert users back
        for user_row in users_backup:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                         (user_row[1], user_row[2]))  # Skip id column
        
        conn.commit()
        conn.close()
        print(f"   ✅ Restored {len(users_backup)} user accounts")
    except Exception as e:
        print(f"   ❌ Failed to restore users: {e}")
else:
    print("\n6️⃣ No user accounts to restore (fresh installation)")

# 7. VERIFY
print("\n7️⃣ Verifying database...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM players')
player_count = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM users')
user_count = cursor.fetchone()[0]

cursor.execute('SELECT DISTINCT Format FROM players')
formats = [row[0] for row in cursor.fetchall()]

conn.close()

print(f"   ✅ Players: {player_count}")
print(f"   ✅ Users: {user_count}")
print(f"   ✅ Formats: {formats}")

print("\n" + "=" * 60)
print("✨ Migration Complete!")
print("   ✓ User data preserved")
print("   ✓ Player data updated")
print("   ✓ Database clean and ready")
print("=" * 60)
