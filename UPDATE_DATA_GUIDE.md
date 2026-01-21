# Quick Guide: Updating Player Data

When you edit CSV files (odi_batsman.csv, odi_bowler.csv, etc.), you need to sync those changes to the database.

## ✅ Safe Method: Sync Without Losing Users
Run this command in your terminal:
```powershell
python migrate_data.py
```

**This will:**
- ✅ Update all player records from your CSV files
- ✅ Preserve all user accounts and login data
- ✅ Add new players if you added them to CSVs
- ✅ Update existing players with new stats

Then refresh your Streamlit app (press `R` in browser or restart).

## ⚠️ NEVER Delete the Database
**DO NOT** run `del cricket_dashboard.db` - this will delete all user accounts!

The migration script now safely updates player data without touching user accounts.

---

**Workflow:**
1. Edit your CSV files (add/update player stats)
2. Run `python migrate_data.py`
3. Refresh your dashboard
4. Done! ✨
