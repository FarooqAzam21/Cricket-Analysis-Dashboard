#!/usr/bin/env python
"""Test script to verify data loading and classification."""

import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, '.')

def test_csv_loading():
    """Test if CSVs can be loaded."""
    print("=" * 60)
    print("TEST 1: CSV LOADING")
    print("=" * 60)
    
    csv_files = {
        'batsman': 'odi_batsman.csv',
        'bowler': 'odi_bowler.csv',
        'all_rounder': 'odi_all_rounders.csv'
    }
    
    for name, path in csv_files.items():
        exists = os.path.exists(path)
        print(f"\n{name} ({path}): {'✅ EXISTS' if exists else '❌ MISSING'}")
        if exists:
            df = pd.read_csv(path)
            print(f"  - Rows: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            print(f"  - Has 'role' column: {'role' in df.columns}")
            if 'role' in df.columns:
                print(f"  - Role values (first 5 unique): {list(df['role'].unique()[:5])}")
                print(f"  - Null roles: {df['role'].isna().sum()}")

def test_data_cleaning():
    """Test data cleaning function."""
    print("\n" + "=" * 60)
    print("TEST 2: DATA CLEANING")
    print("=" * 60)
    
    df_bat = pd.read_csv('odi_batsman.csv')
    
    # Apply cleaning
    def clean(df):
        if df.empty: 
            return df
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]
        for c in ['player', 'Team', 'Format', 'role']:
            if c in df.columns:
                df[c] = (df[c]
                        .astype(str)
                        .str.strip()
                        .str.replace(r'[\t\r\n"\']', '', regex=True)
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.strip())
        return df
    
    df_clean = clean(df_bat)
    
    print(f"\nBefore cleaning:")
    print(f"  - First role: '{df_bat['role'].iloc[0]}'")
    print(f"  - Unique roles: {df_bat['role'].nunique()}")
    
    print(f"\nAfter cleaning:")
    print(f"  - First role: '{df_clean['role'].iloc[0]}'")
    print(f"  - Unique roles: {df_clean['role'].nunique()}")
    print(f"  - Role values (first 10): {list(df_clean['role'].unique()[:10])}")

def test_classification():
    """Test role classification."""
    print("\n" + "=" * 60)
    print("TEST 3: ROLE CLASSIFICATION")
    print("=" * 60)
    
    df_bat = pd.read_csv('odi_batsman.csv')
    
    # Clean
    df = df_bat.copy()
    df.columns = [col.strip() for col in df.columns]
    for c in ['player', 'Team', 'Format', 'role']:
        if c in df.columns:
            df[c] = (df[c]
                    .astype(str)
                    .str.strip()
                    .str.replace(r'[\t\r\n"\']', '', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip())
    
    # Classify
    df['role_lower'] = df['role'].fillna('').astype(str).str.lower()
    
    print(f"\nTotal players: {len(df)}")
    print(f"Unique roles: {df['role_lower'].nunique()}")
    print(f"Role distribution:\n{df['role_lower'].value_counts()}")
    
    # Test filters
    batsmen = df[df['role_lower'].str.contains('batsman|batter', na=False, regex=False)]
    bowlers = df[df['role_lower'].str.contains('bowler|spinner|fast|seam|pace', na=False)]
    wk = df[df['role_lower'].str.contains('wicket|keeper', na=False, regex=False)]
    ar = df[df['role_lower'].str.contains('all-rounder|all rounder|allrounder|fast-bowling|spinner|arm', na=False)]
    
    print(f"\nFiltered results:")
    print(f"  - Batsmen: {len(batsmen)}")
    print(f"  - Bowlers: {len(bowlers)}")
    print(f"  - Wicket Keepers: {len(wk)}")
    print(f"  - All-rounders: {len(ar)}")
    
    # Fallback filters
    print(f"\nFallback filters (runs/wickets):")
    batsmen_fallback = df[df['runs'] > 0]
    bowlers_fallback = df[df['wickets'] > 0]
    print(f"  - Batsmen (runs > 0): {len(batsmen_fallback)}")
    print(f"  - Bowlers (wickets > 0): {len(bowlers_fallback)}")

def test_data_loader():
    """Test the actual data loader function."""
    print("\n" + "=" * 60)
    print("TEST 4: ACTUAL DATA LOADER")
    print("=" * 60)
    
    try:
        from src.data_loader import load_all_data, _get_csv_cache_key
        
        print("Loading data...")
        result = load_all_data(_csv_cache_key=_get_csv_cache_key())
        
        all_players, batsmen, all_rounders, bowlers, df_year, _, _, wk = result
        
        print(f"\nResults:")
        print(f"  - Total players: {len(all_players) if all_players is not None else 'None'}")
        print(f"  - Batsmen: {len(batsmen) if batsmen is not None else 'None'}")
        print(f"  - Bowlers: {len(bowlers) if bowlers is not None else 'None'}")
        print(f"  - All-rounders: {len(all_rounders) if all_rounders is not None else 'None'}")
        print(f"  - Wicket keepers: {len(wk) if wk is not None else 'None'}")
        
        if all_players is not None and len(all_players) > 0:
            print(f"\nAll players shape: {all_players.shape}")
            print(f"All players columns: {list(all_players.columns)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_csv_loading()
    test_data_cleaning()
    test_classification()
    test_data_loader()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
