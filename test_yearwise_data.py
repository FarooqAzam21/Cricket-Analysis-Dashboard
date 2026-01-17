import pandas as pd
import sys
sys.path.insert(0, 'c:\\Users\\Farooq\\Desktop\\New Folder (4)\\Cricket_Analysis')

print("Testing year-wise data loading...")
print("=" * 50)

# Test 1: Load CSV directly
print("\n1. Loading yearwise_data.csv directly:")
df = pd.read_csv('c:\\Users\\Farooq\\Desktop\\New Folder (4)\\Cricket_Analysis\\yearwise_data.csv')
print(f"   - Rows: {len(df)}")
print(f"   - Columns: {list(df.columns)}")
print(f"   - Players: {len(df['player'].unique())} unique")
print(f"   - Sample: {df.iloc[0].to_dict()}")

# Test 2: Check data_loader
print("\n2. Testing data_loader.load_all_data():")
from src.data_loader import load_all_data
all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data()
print(f"   - Year-wise data rows: {len(year_wise)}")
print(f"   - Year-wise columns: {list(year_wise.columns)}")
if len(year_wise) > 0:
    print(f"   - Sample: {year_wise.iloc[0].to_dict()}")
else:
    print("   - WARNING: No data loaded!")

# Test 3: Check specific player
print("\n3. Testing specific player (Babar Azam):")
babar_data = year_wise[year_wise['player'] == 'Babar Azam']
print(f"   - Records: {len(babar_data)}")
if len(babar_data) > 0:
    print(f"   - Years: {sorted(babar_data['year'].unique())}")
    print(f"   - Runs: {babar_data['runs'].tolist()}")

print("\n" + "=" * 50)
print("✅ Year-wise data loading test completed!")
