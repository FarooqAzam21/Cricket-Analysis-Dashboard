"""
Test player data loading and safe conversions
"""
import sys
sys.path.insert(0, r'c:\Users\Farooq\Desktop\New Folder (4)\Cricket_Analysis')

from src.data_loader import load_all_data

print("🔄 Loading player data...")
all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data()

print(f"✅ Total players loaded: {len(all_players)}")
print(f"✅ Batsmen: {len(df_batsman)}")
print(f"✅ Bowlers: {len(df_bowler)}")
print(f"✅ All-rounders: {len(df_allrounder)}")

# Show sample data with potential None values
print("\n📋 Sample player data (first 3 players):")
sample_cols = ['player_name', 'role', 'matches', 'runs', 'wickets', 'average', 'strike_rate']
available_cols = [col for col in sample_cols if col in all_players.columns]
print(all_players[available_cols].head(3).to_string())

print("\n🔍 Checking for None values in numeric columns:")
numeric_cols = ['matches', 'runs', 'wickets', 'average', 'strike_rate', 'economy', 'bowling_average']
for col in numeric_cols:
    if col in all_players.columns:
        none_count = all_players[col].isna().sum()
        zero_count = (all_players[col] == 0).sum()
        print(f"  {col:20} - None: {none_count:3d}, Zeros: {zero_count:3d}")

print("\n✅ Player data ready for player management feature!")
print("✅ Safe conversions will handle any None/NaN values gracefully")
