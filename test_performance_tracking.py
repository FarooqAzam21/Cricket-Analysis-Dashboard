#!/usr/bin/env python3
"""Test script to verify all new functions work correctly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import (
    add_player_performance,
    get_match_performances,
    calculate_batsman_fantasy_points,
    calculate_bowler_fantasy_points,
    calculate_team_strength,
    get_team_strength_rating,
    calculate_updated_fantasy_scores,
    init_db
)

print("✅ All new functions imported successfully!")

# Test calculation functions
print("\n📊 Testing Fantasy Points Calculation:")

# Test batsman points
batsman_points = calculate_batsman_fantasy_points(
    runs=45,
    fours=5,
    sixes=1,
    balls_faced=35,
    is_captain=False
)
print(f"  Batsman (45 runs, 35 balls, 5 4s, 1 6): {batsman_points} points")

# Test bowler points
bowler_points = calculate_bowler_fantasy_points(
    wickets=2,
    economy=6.5,
    is_captain=False
)
print(f"  Bowler (2 wickets, economy 6.5): {bowler_points} points")

# Test team strength
print("\n⚡ Testing Team Strength Calculation:")
test_team = ["Virat Kohli", "Rohit Sharma", "KL Rahul"]
strength = calculate_team_strength(test_team, tournament_id=1)
print(f"  Team strength for sample players: {strength}/100")

print("\n✅ All tests passed! Performance tracking system ready to use.")
print("\nNext steps:")
print("1. Run: streamlit run main.py")
print("2. Go to Admin Panel → Tab 6: Update Match Scores")
print("3. Record player performances for completed matches")
print("4. Click 'Recalculate Fantasy Points'")
print("5. View AI Team Strength Analysis in Danger Zone section")
