# ⚡ Quick Start Guide - Performance Tracking

## 🎯 5-Minute Setup

### Step 1: Run Your App
```bash
cd "c:\Users\Farooq\Desktop\New Folder (4)\Cricket_Analysis"
streamlit run main.py
```

### Step 2: Create/Select Tournament
- Login as admin
- Go to Admin Panel
- Create tournament or select existing one
- Note the Tournament ID

### Step 3: Complete a Match
**Admin Panel → Tab 6 → Step 1**
1. Select incomplete match
2. Enter team scores
3. Select winner
4. Click "Update Score"
✅ Match is now completed

### Step 4: Record Player Performance
**Admin Panel → Tab 6 → Step 2**
1. Select completed match
2. Choose team
3. Select player
4. Enter stats:
   - Runs
   - Balls Faced
   - Fours
   - Sixes
5. Click "Add Performance"
6. Repeat for other players

**Example:**
```
Match: India vs Pakistan
Virat Kohli:
  Runs: 45
  Balls: 35
  Fours: 5
  Sixes: 1
  ✓ SR: 128.6% (auto-calculated)
```

### Step 5: Recalculate Points
**Admin Panel → Tab 6 → Bottom**
1. Click "🔄 Recalculate Fantasy Points"
2. Wait for confirmation
✅ Leaderboard updated!

### Step 6: View Team Strengths
**Admin Panel → "Team Strength Analysis" (Bottom)**
1. Enter Tournament ID
2. See all teams with strength scores
3. View bar chart
4. Check color ratings

---

## 📊 Fantasy Points Formula

### Batsman (45 runs, 35 balls, 5 fours, 1 six)
```
Runs:           45 × 1 = 45 points
Fours:          5 × 1  = 5 points
Sixes:          1 × 2  = 2 points
Strike Rate:    128.6% > 120% = 5 points
                ─────────────────────────
                Total = 57 points
```

### Bowler (2 wickets, economy 6.5)
```
Wickets:        2 × 25 = 50 points
Economy:        6.5 is between 6-8 = 5 points
                ─────────────────────────
                Total = 55 points
```

---

## 🟢 Team Strength Rating

### How It's Calculated
- Player batting average (normalized)
- Player strike rate (normalized)
- Player bowling stats (normalized)
- Player role multiplier (all-rounder = 1.3x)
- Team balance bonus

### Rating Scale
| Score | Rating | Emoji |
|-------|--------|-------|
| 70+   | Strong | 🟢    |
| 50-70 | Medium | 🟡    |
| <50   | Weak   | 🔴    |

### Example Ratings
- **India** (Strong squad): 75 🟢
- **Bangladesh** (Moderate): 60 🟡
- **New Zealand** (Recovering): 45 🔴

---

## 🔑 Key Tips

1. **Record Performance Immediately**
   - After match completes
   - While stats are fresh
   - Accurate data = accurate points

2. **Select Balanced Teams**
   - Mix batsmen and bowlers
   - Include all-rounders (1.3x value)
   - Aim for 70+ strength rating

3. **Check Team Strength**
   - Before finalizing squad
   - Compare different lineups
   - Use for fantasy picks

4. **Recalculate After All Data**
   - Don't recalculate for each player
   - Enter all performances first
   - Then click recalculate once

5. **Use Analytics for Decisions**
   - High strength = good team
   - View leaderboard to see impact
   - Analyze player performance patterns

---

## 🆘 Common Issues & Fixes

### Issue: Performance not showing?
**Solution:**
1. Check match status is "completed"
2. Verify player name matches exactly
3. Confirm team is correct

### Issue: Team strength = 0?
**Solution:**
1. Ensure all 15 players selected
2. Check players exist in database
3. Verify player stats are entered

### Issue: Fantasy points not updating?
**Solution:**
1. Click recalculation button
2. Wait for "✅" confirmation
3. Refresh page (F5)

### Issue: Can't find match?
**Solution:**
1. Go to Tab 5 (Manage Matches)
2. Verify match exists
3. Check match status (must be completed for performance)

---

## 📈 Workflow Diagram

```
┌─────────────────┐
│ Create Match    │
│ (Tab 4)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Complete Match  │
│ (Tab 6, Step 1) │
│ Enter score &   │
│ winner          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Record Perf     │
│ (Tab 6, Step 2) │
│ Player stats    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Recalculate     │
│ (Tab 6 bottom)  │
│ Update scores   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ View Results    │
│ Leaderboard &   │
│ Team Strength   │
└─────────────────┘
```

---

## 🎮 Example Session

### Tournament: T20 World Cup 2025
### Teams: India, Pakistan, Australia, England, etc.

#### Match 1: India vs Pakistan (Group A)
**Admin completes match:**
- India: 175/8
- Pakistan: 165/9
- Winner: India

**Admin records performance:**
```
Player: Virat Kohli
Runs: 45, Balls: 35, Fours: 5, Sixes: 1
→ Points: 57 (45 runs + 5 fours + 2 sixes + SR bonus)

Player: Jasprit Bumrah  
Wickets: 2, Economy: 6.5
→ Points: 55 (50 for wickets + 5 for economy)
```

**Leaderboard Updates:**
- User1 (selected Kohli as captain): +114 points (57 × 2)
- User2 (selected Bumrah): +55 points
- User3 (no India players): +0 points

#### Match 2: Australia vs England (Group B)
**Repeat process...**

---

## 📱 Navigation

### As Admin
1. **Home** → See platform info
2. **Cricket Analysis** → View existing data
3. **Tournament** → Create new tournament
4. **Admin Panel** → Management interface
   - Tab 1: Create tournament
   - Tab 2: Add teams
   - Tab 3: Add players (see strength here!)
   - Tab 4: Schedule matches
   - Tab 5: Manage matches
   - Tab 6: Update scores & record performance
   - Bottom: Team strength analysis

### As Regular User
1. **Home** → Platform overview
2. **Cricket Analysis** → Historical data
3. **Tournament** → Create fantasy team
4. **Leaderboard** → See rankings (updated after each match)

---

## 💾 Database Info

**File:** `cricket_dashboard.db`
**Location:** Same folder as main.py
**Size:** Grows with matches and performances

✅ **Backed up?** No automatic backups yet. Consider manual backups if important!

---

## 🚀 Next Steps

1. ✅ **Understand** the performance tracking system (you're reading this!)
2. ✅ **Test** with first match (enter a performance)
3. ✅ **Monitor** leaderboard updates
4. ✅ **Build** fantasy teams using strength ratings
5. ✅ **Analyze** player performance patterns
6. ✅ **Optimize** team selections based on data

---

## 📞 Support

**Need help?**
1. Check documentation: `PERFORMANCE_TRACKING.md`
2. Review setup guide: `SETUP_PERFORMANCE_TRACKING.md`
3. Check changelog: `CHANGELOG_PERFORMANCE_TRACKING.md`
4. Run test: `python test_performance_tracking.py`

---

## ✅ You're Ready!

1. Run streamlit
2. Go to Admin Panel
3. Complete a match
4. Record performance
5. Click recalculate
6. View results!

**Happy tracking! 🎯**
