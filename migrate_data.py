from src.data_loader import load_all_data
from src.database import init_db, save_to_db
import os

def migrate():
    print("🚀 Starting data migration from CSV to SQLite...")
    
    # 1. Load data using the existing robust loader
    # (Fixes duplicates and cleans strings during load)
    all_players, _, _, _, _, _, _, _ = load_all_data()
    
    if all_players is None or all_players.empty:
        print("❌ No data found to migrate.")
        return

    # 2. Initialize database schema
    init_db()
    
    # 3. Save to database
    save_to_db(all_players)
    
    print(f"✅ Migration successful! {len(all_players)} records moved to database.")

if __name__ == "__main__":
    migrate()
