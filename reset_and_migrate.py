"""
Reset database completely and rebuild from cleaned CSV files
"""
import os
import sqlite3
from src.config import DATA_PATHS
from src.database import init_db, save_to_db
import pandas as pd

# 1. Delete old database
db_path = 'cricket_dashboard.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✅ Deleted old database: {db_path}")

# 2. Create fresh database schema
init_db()
print("✅ Created fresh database schema")

# 3. Load CLEANED CSV files
print("\n📖 Loading CSV files...")
df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()

print(f"  - Batsmen: {len(df_bat)} rows")
print(f"  - All-rounders: {len(df_ar)} rows")
print(f"  - Bowlers: {len(df_bowl)} rows")

# 4. Clean data
def clean(df):
    if df.empty: 
        return df
    # Strip whitespace from column names
    df.columns = df.columns.map(str).str.strip()
    # Clean text columns
    for c in ['player', 'Team', 'Format', 'role']:
        if c in df.columns: 
            df[c] = df[c].astype(str).str.replace(r'[\t"\']', '', regex=True).str.strip()
    return df

df_bat_clean = clean(df_bat)
df_ar_clean = clean(df_ar)
df_bowl_clean = clean(df_bowl)

# 5. Combine
composite = pd.concat([df_bat_clean, df_ar_clean, df_bowl_clean], ignore_index=True, sort=False)
print(f"\n📊 Combined data: {len(composite)} rows")

# 6. Deduplicate by (player, Team, Format)
all_players = composite.groupby(['player', 'Team', 'Format'], as_index=False).first()
print(f"📊 After deduplication: {len(all_players)} unique player-team-format combinations")

# 7. Check for bad data
print("\n🔍 Data Quality Check:")
print(f"  - Unique Formats: {all_players['Format'].unique()}")
print(f"  - Unique Teams: {all_players['Team'].unique()}")
print(f"  - Any format='5'? {(all_players['Format'] == '5').sum()} rows")
print(f"  - Any format='5'? {(all_players['Format'] == 5).sum()} rows")

# 8. Save to fresh database
save_to_db(all_players)
print(f"\n✅ Imported {len(all_players)} player records to database")

# 9. Verify
conn = sqlite3.connect('cricket_dashboard.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM players')
total = cursor.fetchone()[0]
cursor.execute('SELECT DISTINCT Format FROM players')
formats = [row[0] for row in cursor.fetchall()]
conn.close()

print(f"\n✅ Database verification:")
print(f"  - Total records: {total}")
print(f"  - Formats: {formats}")
print(f"\n🎉 Database reset and migration complete!")
