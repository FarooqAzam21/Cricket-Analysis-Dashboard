# 🎯 Player Data Management - Admin Guide

**Created:** February 2, 2026  
**Feature:** Update player statistics directly from admin panel  
**Status:** ✅ Live

---

## How to Access Player Management

1. **Login as Admin**
   - Username: `admin`
   - Password: `admin`

2. **Navigate to Admin Panel**
   - Click "⚙️ Admin" button in sidebar
   - Or click the "Admin Panel" option

3. **Open Player Management Tab**
   - In the admin panel, find the "🏏 Player Data Management" tab (last tab)
   - Click to open

---

## How to Update a Single Player

### Step 1: Search Player
- Click "Select Player" dropdown
- Type player name (e.g., "Babar Azam", "Virat Kohli")
- App shows 888 players available

### Step 2: Select Format & Team
- Choose Format (ODI, T20, Test)
- Choose Team (automatically filtered for that player)

### Step 3: Review Current Stats
- See current statistics displayed:
  - Matches, Runs, Wickets, Average, Strike Rate
  - Bowling Average, Economy, Centuries, Half Centuries

### Step 4: Update Statistics
- **Batting Stats (Left Column)**
  - Matches
  - Innings
  - Not Out
  - Runs
  - Strike Rate
  - Batting Position

- **Bowling Stats (Middle Column)**
  - Wickets
  - Bowling Average
  - Economy

- **Achievement Stats (Right Column)**
  - Batting Average
  - Centuries
  - Half Centuries
  - Player Role

### Step 5: Submit Changes
- Click **✅ Submit & Update Database** button
- Wait for confirmation message
- Data updates immediately in database

---

## Example: Update Babar Azam's Stats

**Before:**
- Runs: 6501
- Strike Rate: 87.16

**Update Steps:**
1. Select "Babar Azam"
2. Select "ODI" format
3. Select "Pakistan" team
4. Change Runs to 6600
5. Change Strike Rate to 88.5
6. Click **Submit & Update Database**
7. See success message: "✅ Successfully updated Babar Azam's statistics!"

---

## Bulk Update with CSV

### When to Use
- Update multiple players at once
- Import updated stats from external source
- Batch updates across teams

### CSV Format Required

Create a CSV file with these columns:

```
player,team,format,matches,innings,no,runs,wickets,average,strike_rate,bowling_average,economy,hundreds,fifties,batting_position,role
Babar Azam,Pakistan,Odi,100,95,8,6600,0,75.0,88.5,0.0,0.0,15,35,1,Batsman
Virat Kohli,India,Odi,115,110,8,14800,2,144.0,95.0,0.0,0.0,50,65,2,Batsman
```

### Steps for Bulk Update

1. **Prepare CSV File**
   - Include columns: `player`, `team`, `format`, and any stats to update
   - Required columns: `player`, `team`, `format`

2. **Open Bulk Update Section**
   - Scroll to "📤 Bulk Update (CSV)" section
   - Click to expand

3. **Upload CSV**
   - Click "Upload CSV file"
   - Select your prepared CSV

4. **Preview Data**
   - Review the preview table
   - Verify data looks correct

5. **Submit Bulk Update**
   - Click **🚀 Bulk Update Database**
   - Wait for processing
   - See results: X players updated, Y failed

---

## Available Update Fields

### Numeric Fields (Integers)
- `matches` - Total matches played
- `innings` - Total innings batted
- `no` - Not out occurrences
- `runs` - Total runs scored
- `wickets` - Total wickets taken
- `hundreds` - Centuries scored
- `fifties` - Half centuries scored
- `batting_position` - Typical batting order position

### Decimal Fields (Floats)
- `average` - Batting average (runs per innings)
- `strike_rate` - Batting strike rate
- `bowling_average` - Bowling average (runs per wicket)
- `economy` - Bowling economy (runs per over)

### Text Fields
- `role` - Player role (Batsman, Bowler, All-rounder, Wicket-keeper)

---

## Tips & Best Practices

✅ **DO:**
- Update one player at a time for safety
- Review current stats before updating
- Use bulk update for many players
- Keep CSV files as backups

❌ **DON'T:**
- Leave required fields (player, team, format) blank
- Use negative numbers for stats
- Update wrong player accidentally

---

## Troubleshooting

**Problem:** "No players found in database"
- **Solution:** Make sure CSV data was loaded. Check data_loader.py

**Problem:** Update shows success but data unchanged
- **Solution:** Hard refresh browser to clear cache

**Problem:** Bulk update shows errors
- **Solution:** Check CSV format - ensure all required columns present

**Problem:** Can't find a specific player
- **Solution:** Check spelling - player names are case-sensitive

---

## Database Fields Updated

When you update a player, these fields in the database are modified:

| Field | Type | Example |
|-------|------|---------|
| player | TEXT | Babar Azam |
| team | TEXT | Pakistan |
| format | TEXT | Odi |
| matches | INTEGER | 100 |
| innings | INTEGER | 95 |
| no | INTEGER | 8 |
| runs | INTEGER | 6501 |
| wickets | INTEGER | 0 |
| average | REAL | 75.01 |
| strike_rate | REAL | 87.16 |
| bowling_average | REAL | 0.0 |
| economy | REAL | 0.0 |
| hundreds | INTEGER | 15 |
| fifties | INTEGER | 35 |
| batting_position | INTEGER | 1 |
| role | TEXT | Batsman |

---

## Security

- **Admin Only:** Only users logged in as 'admin' can access this feature
- **No Deletions:** Updates only, no data deletion possible
- **Audit Trail:** All updates are timestamped in database
- **Direct DB:** Changes go directly to database, bypassing cache

---

## Support

If you encounter issues:
1. Check the browser console (F12) for errors
2. Verify admin credentials
3. Ensure CSV format is correct
4. Clear browser cache and try again
5. Check that database file exists at `cricket_dashboard.db`

---

**Happy updating! 🎉**
