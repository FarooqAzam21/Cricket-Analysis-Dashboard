"""
Emergency diagnostic: Test data flow from loader to display
"""
import sys
sys.path.insert(0, '/c/Users/Farooq/Desktop/New Folder (4)/Cricket_Analysis')

from src.data_loader import load_all_data

print("="*70)
print("EMERGENCY DATA FLOW TEST")
print("="*70)

# Load data
result = load_all_data()
print(f"\nReturn tuple length: {len(result)}")
print(f"Return types: {[type(r).__name__ for r in result]}")

all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = result

print(f"\n1. all_players:")
print(f"   Type: {type(all_players)}")
print(f"   Length: {len(all_players) if all_players is not None and hasattr(all_players, '__len__') else 'N/A'}")
if all_players is not None and not all_players.empty:
    print(f"   Columns: {list(all_players.columns)[:5]}...")
    print(f"   Sample player: {all_players.iloc[0]['player'] if 'player' in all_players.columns else 'N/A'}")
    print(f"   Sample runs: {all_players.iloc[0]['runs'] if 'runs' in all_players.columns else 'N/A'}")

print(f"\n2. df_batsman:")
print(f"   Type: {type(df_batsman)}")
print(f"   Length: {len(df_batsman) if df_batsman is not None and hasattr(df_batsman, '__len__') else 'N/A'}")

print(f"\n3. df_allrounder:")
print(f"   Type: {type(df_allrounder)}")
print(f"   Length: {len(df_allrounder) if df_allrounder is not None and hasattr(df_allrounder, '__len__') else 'N/A'}")

print(f"\n4. df_bowler:")
print(f"   Type: {type(df_bowler)}")
print(f"   Length: {len(df_bowler) if df_bowler is not None and hasattr(df_bowler, '__len__') else 'N/A'}")

print(f"\n5. year_wise:")
print(f"   Type: {type(year_wise)}")
print(f"   Length: {len(year_wise) if year_wise is not None and hasattr(year_wise, '__len__') else 'N/A'}")

print(f"\n6. batsmen:")
print(f"   Type: {type(batsmen)}")
print(f"   Length: {len(batsmen) if batsmen is not None and hasattr(batsmen, '__len__') else 'N/A'}")

print(f"\n7. all_rounders:")
print(f"   Type: {type(all_rounders)}")
print(f"   Length: {len(all_rounders) if all_rounders is not None and hasattr(all_rounders, '__len__') else 'N/A'}")

print(f"\n8. wicket_keepers:")
print(f"   Type: {type(wicket_keepers)}")
print(f"   Length: {len(wicket_keepers) if wicket_keepers is not None and hasattr(wicket_keepers, '__len__') else 'N/A'}")

print("\n" + "="*70)
print("✅ DATA FLOW TEST COMPLETE")
print("="*70)
