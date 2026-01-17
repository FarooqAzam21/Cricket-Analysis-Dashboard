"""
Test script to verify cache invalidation works
"""
import os
import time
from src.data_loader import _get_csv_mtime

print("Testing CSV modification time detection...")
print()

# Initial check
print(f"Initial mtime sum: {_get_csv_mtime()}")
time.sleep(1)

# Simulate CSV change by checking again
print(f"After 1 second: {_get_csv_mtime()}")
time.sleep(1)

# Check individual file times
csv_files = [
    'odi_batsman.csv',
    'odi_bowler.csv',
    'odi_all_rounders.csv',
    'yearwise_data.csv'
]

print("\nIndividual file mtimes:")
for f in csv_files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        print(f"  {f}: {mtime}")
    else:
        print(f"  {f}: NOT FOUND")

print("\n✅ If you edit a CSV and run this again, the mtime sum should change!")
