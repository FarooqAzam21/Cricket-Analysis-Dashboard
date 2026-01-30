# 🔐 Safe Data Migration Guide

## 📌 Problem Solved

✅ **Fixed:** Migration script now preserves:
- ✅ Admin user account (auto-created if missing)
- ✅ All tournaments, teams, and matches (if backed up)
- ✅ All fantasy teams and player performances
- ✅ Column name mismatch (`squad` vs `players`)

## ⚠️ Important Note

The database migration process **deletes the entire database** to refresh player data from CSVs. Therefore:
- Tournament data needs to be backed up manually
- User accounts are backed up automatically

## 🔄 Safe Migration Process (3 Steps)

### Step 1️⃣: Backup Your Tournaments
```bash
python backup_tournaments.py
```

**What this does:**
- ✅ Saves all tournaments to a JSON file
- ✅ Saves all tournament teams
- ✅ Saves all matches
- ✅ Saves all fantasy teams
- ✅ Creates file: `tournament_backup_YYYYMMDD_HHMMSS.json`

**Example output:**
```
✅ Tournament backup successful!
   📁 Backup file: tournament_backup_20260130_143022.json
   📊 Backed up:
      - 1 tournaments
      - 11 teams
      - 45 matches
      - 3 fantasy teams
```

### Step 2️⃣: Run Migration
```bash
python migrate_data.py
```

**What this does:**
- ✅ Deletes old database (necessary for fresh player data)
- ✅ Creates new database schema
- ✅ Loads 860+ player records from CSVs
- ✅ Recreates admin user if missing
- ✅ Restores any previously backed-up user accounts

**Example output:**
```
🔄 Safe Data Migration (Preserving User Accounts & Tournaments)
============================================================

1️⃣ Backing up user accounts and tournaments...
   ✅ Backed up 1 user accounts
   ✅ Backed up 0 tournaments (because we backed them up separately!)
2️⃣ Resetting database...
   ✅ Old database deleted
3️⃣ Creating fresh database schema...
4️⃣ Loading CSV data...
5️⃣ Importing 860 player records...
6️⃣ Restoring user accounts and tournament data...
   ✅ Created admin account
   ✅ Restored 1 user accounts
============================================================
✅ Migration successful!
   ✓ 860 player records updated
   ✓ User accounts preserved
```

### Step 3️⃣: Restore Your Tournaments
```bash
python backup_tournaments.py --restore tournament_backup_YYYYMMDD_HHMMSS.json
```

**Replace the file name with your actual backup file!**

**Example:**
```bash
python backup_tournaments.py --restore tournament_backup_20260130_143022.json
```

**What this does:**
- ✅ Restores all tournaments from backup
- ✅ Restores all tournament teams
- ✅ Restores all matches
- ✅ Restores all fantasy teams

## 🔑 Default Admin Credentials

After migration, if admin user doesn't exist:
- **Username:** `admin`
- **Password:** `admin123`

**⚠️ Change this password immediately after login!**

## 📋 Migration Checklist

Before running migration:
- [ ] Run `python backup_tournaments.py`
- [ ] Save the backup file in a safe location
- [ ] Note down the filename (e.g., `tournament_backup_20260130_143022.json`)

After migration:
- [ ] Run `python backup_tournaments.py --restore tournament_backup_YYYYMMDD_HHMMSS.json`
- [ ] Verify all tournaments appear in admin panel
- [ ] Verify all teams and matches are restored
- [ ] Test admin login with credentials
- [ ] Update admin password if needed

## 🆘 Troubleshooting

### Tournament not appearing after restore?
1. Check if backup file exists: `ls tournament_backup_*.json`
2. Try restore again: `python backup_tournaments.py --restore <filename>`
3. Check admin panel → Tournaments Overview

### Admin login failing?
1. Admin account should auto-create with credentials (admin/admin123)
2. If login still fails, delete database and run migration again

### Player data not updated?
1. Make sure CSVs are in correct location (odi_batsman.csv, odi_bowler.csv, odi_all_rounders.csv)
2. Check file paths in config.py

## ✅ What's Fixed in This Update

1. **Admin user auto-creation** - If admin doesn't exist after migration, it's created automatically
2. **Column name mismatch** - Fixed `players` → `squad` in database restore logic  
3. **IntegrityError handling** - Duplicate entries won't cause migration to fail
4. **Tournament backup script** - New tool to safely backup/restore tournaments
5. **Clear instructions** - Migration now displays helpful guidance

## 🎯 Summary

Your tournament and admin data will now be **completely safe** during migration if you follow these 3 simple steps:

1. `python backup_tournaments.py`
2. `python migrate_data.py`
3. `python backup_tournaments.py --restore tournament_backup_YYYYMMDD_HHMMSS.json`

All your data will be preserved! ✅
