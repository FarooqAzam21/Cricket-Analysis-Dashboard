import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import create_tournament, get_tournament

# Create a test tournament
tournament_id = create_tournament("T20 World Cup 2026", "2026-02-01", "2026-02-28")
print(f"Created tournament with ID: {tournament_id}")

# Verify it was created
tournament = get_tournament(tournament_id)
if tournament:
    print(f"✅ Found: {tournament['name']} (Status: {tournament['status']})")
else:
    print("❌ Tournament not found")
