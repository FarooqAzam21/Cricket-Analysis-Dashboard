"""
Test script to verify CSV column mapping and data loading
"""
import pandas as pd
import os

# Check CSV files exist
print("=" * 60)
print("STEP 1: Check CSV files exist")
print("=" * 60)

csv_files = {
    "odi_batsman.csv": "Batsman data",
    "odi_bowler.csv": "Bowler data",
    "odi_all_rounders.csv": "All-rounder data",
    "yearwise_data_cleaned.csv": "Year-wise stats"
}

for filename, description in csv_files.items():
    path = filename
    exists = os.path.exists(path)
    print(f"✓ {filename}: {description} - {'EXISTS' if exists else 'MISSING'}")
    if exists:
        df = pd.read_csv(path)
        print(f"  → Rows: {len(df)}, Columns: {len(df.columns)}")

print("\n" + "=" * 60)
print("STEP 2: Load and inspect headers")
print("=" * 60)

# Load each CSV
df_bat = pd.read_csv("odi_batsman.csv")
df_bowl = pd.read_csv("odi_bowler.csv")
df_ar = pd.read_csv("odi_all_rounders.csv")

print("\nODI Batsman Headers:")
print(f"  {list(df_bat.columns)}")
print(f"  Total: {len(df_bat.columns)} columns")

print("\nODI Bowler Headers:")
print(f"  {list(df_bowl.columns)}")
print(f"  Total: {len(df_bowl.columns)} columns")

print("\nODI All-Rounder Headers:")
print(f"  {list(df_ar.columns)}")
print(f"  Total: {len(df_ar.columns)} columns")

print("\n" + "=" * 60)
print("STEP 3: Check column differences")
print("=" * 60)

bat_cols = set(df_bat.columns)
bowl_cols = set(df_bowl.columns)
ar_cols = set(df_ar.columns)

print("\nColumns ONLY in Batsman:")
print(f"  {bat_cols - bowl_cols - ar_cols}")

print("\nColumns ONLY in Bowler:")
print(f"  {bowl_cols - bat_cols - ar_cols}")

print("\nColumns ONLY in All-Rounder:")
print(f"  {ar_cols - bat_cols - bowl_cols}")

print("\nCommon columns:")
print(f"  {bat_cols & bowl_cols & ar_cols}")

print("\n" + "=" * 60)
print("STEP 4: Check critical column: 'strike_rate' vs 'bowling_strike_rate'")
print("=" * 60)

print(f"\nBatsman has 'strike_rate': {'strike_rate' in df_bat.columns}")
print(f"Bowler has 'strike_rate': {'strike_rate' in df_bowl.columns}")
print(f"Bowler has 'bowling_strike_rate': {'bowling_strike_rate' in df_bowl.columns}")
print(f"All-Rounder has 'strike_rate': {'strike_rate' in df_ar.columns}")

print("\n" + "=" * 60)
print("STEP 5: Check 'role' column content")
print("=" * 60)

print(f"\nBatsman role unique values:")
print(f"  {df_bat['role'].unique()[:10]}")
print(f"  Empty values: {(df_bat['role'] == '').sum()}")

print(f"\nBowler role unique values:")
print(f"  {df_bowl['role'].unique()[:10]}")
print(f"  Empty values: {(df_bowl['role'] == '').sum()}")

print(f"\nAll-Rounder role unique values:")
print(f"  {df_ar['role'].unique()[:10]}")
print(f"  Empty values: {(df_ar['role'] == '').sum()}")

print("\n" + "=" * 60)
print("STEP 6: Simulate data loading with column mapping")
print("=" * 60)

# Standardize column names BEFORE concat
print("\nMapping 'bowling_strike_rate' → 'strike_rate' in Bowler CSV...")
if 'bowling_strike_rate' in df_bowl.columns and 'strike_rate' not in df_bowl.columns:
    df_bowl.rename(columns={'bowling_strike_rate': 'strike_rate'}, inplace=True)
    print("  ✓ Mapping completed")
else:
    print("  ✓ Already has 'strike_rate' or both present")

# Now concat
print("\nConcatenating CSVs...")
all_players = pd.concat([df_bat, df_ar, df_bowl], ignore_index=True, sort=False)
print(f"  Total rows after concat: {len(all_players)}")
print(f"  Total columns: {len(all_players.columns)}")

# Check for NaN columns
print("\nColumns with NaN values:")
nan_counts = all_players.isnull().sum()
nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
for col, count in nan_cols.head(10).items():
    print(f"  {col}: {count} NaN values ({100*count/len(all_players):.1f}%)")

print("\n" + "=" * 60)
print("STEP 7: Check numeric conversion")
print("=" * 60)

numeric_cols = ['wickets', 'runs', 'average', 'strike_rate', 'Innings', 'bowling_average', 'economy', 'matches']
print("\nConverting numeric columns...")
for col in numeric_cols:
    if col in all_players.columns:
        # Replace '-' with 0 before converting
        all_players[col] = all_players[col].astype(str).str.replace('-', '0', regex=False).str.strip()
        all_players[col] = pd.to_numeric(all_players[col], errors='coerce').fillna(0)
        
        # Check statistics
        non_zero = (all_players[col] > 0).sum()
        print(f"  {col}: {non_zero}/{len(all_players)} non-zero values")

print("\n" + "=" * 60)
print("STEP 8: Final data summary")
print("=" * 60)

print(f"\nTotal players: {len(all_players)}")
print(f"Columns in final dataset: {len(all_players.columns)}")
print(f"\nFinal columns:")
for col in sorted(all_players.columns):
    dtype = all_players[col].dtype
    non_null = (all_players[col].notna().sum())
    print(f"  {col}: {dtype} ({non_null} non-null)")

print(f"\nSample data (first 3 rows):")
print(all_players[['player', 'Team', 'Format', 'role', 'runs', 'wickets', 'strike_rate']].head(3))

print("\n" + "=" * 60)
print("✓ COLUMN MAPPING TEST COMPLETE")
print("=" * 60)
