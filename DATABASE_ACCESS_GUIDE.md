# 🗄️ Database Access Guide

## 📂 Database Location

Your database file is located at:
```
C:\Users\Farooq\Desktop\New Folder (4)\Cricket_Analysis\cricket_dashboard.db
```

## 🔍 3 Ways to Access the Database

### Method 1️⃣: Using Python (Easiest)

```bash
cd "c:\Users\Farooq\Desktop\New Folder (4)\Cricket_Analysis"
python -c "import sqlite3; conn = sqlite3.connect('cricket_dashboard.db'); c = conn.cursor(); c.execute('SELECT name FROM sqlite_master WHERE type=\"table\"'); print('\n'.join([t[0] for t in c.fetchall()]))"
```

This shows all tables in your database.

### Method 2️⃣: Using SQLite Command Line (Windows)

If you have SQLite installed, use:
```bash
sqlite3 cricket_dashboard.db
```

Then you can run SQL commands like:
```sql
SELECT * FROM users;
SELECT * FROM tournaments;
SELECT * FROM players;
```

To exit, type: `.exit`

### Method 3️⃣: Using a GUI Tool (Easiest Visual Way)

**Download DB Browser for SQLite:**
- Go to: https://sqlitebrowser.org/
- Download and install
- Open `cricket_dashboard.db` file in the browser
- View/edit data visually

## 📊 Important Tables

| Table | Purpose |
|-------|---------|
| `users` | Admin/user accounts |
| `tournaments` | Tournament info |
| `tournament_teams` | Teams in tournaments |
| `tournament_matches` | Matches |
| `players` | Player data |
| `fantasy_teams` | Fantasy teams created |
| `match_player_performance` | Player performance in matches |

## 🔧 Quick Commands

### View all tables:
```python
python -c "import sqlite3; c = sqlite3.connect('cricket_dashboard.db').cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\"); print('\n'.join([r[0] for r in c.fetchall()]))"
```

### Count records in a table:
```python
python -c "import sqlite3; c = sqlite3.connect('cricket_dashboard.db').cursor(); c.execute('SELECT COUNT(*) FROM players'); print(f'Players: {c.fetchone()[0]}')"
```

### View all users:
```python
python -c "import sqlite3; c = sqlite3.connect('cricket_dashboard.db').cursor(); c.execute('SELECT username FROM users'); print('\n'.join([r[0] for r in c.fetchall()]))"
```

### Delete a tournament:
```python
python -c "import sqlite3; conn = sqlite3.connect('cricket_dashboard.db'); c = conn.cursor(); c.execute('DELETE FROM tournaments WHERE id = 1'); conn.commit()"
```

## ✅ What Just Happened

- ✅ **Deleted tournament:** "T20 World Cup 2026"
- ✅ **Deleted all related:** Teams, Matches, Fantasy Teams
- ✅ **Kept safe:** Admin account, User accounts, Players data

## 🚀 Next Steps

Start fresh:
1. Run the app: `streamlit run main.py`
2. Login with: admin / admin
3. Create a new tournament from scratch

Good luck! 🎉
