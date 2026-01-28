# 🚀 T20 Fantasy Cricket - Quick Start Guide

## For Admin (You)

### First-Time Setup (5 minutes)

1. **Login as Admin**
   - Username: `admin`
   - Password: (use your admin password)

2. **Create Tournament**
   - Click **⚙️ Admin Panel** (only visible for admin)
   - Go to **Create Tournament** tab
   - Enter: `T20 World Cup 2024`
   - Set dates (e.g., Jan 1 - Jan 30)
   - ✅ Check **"Auto-setup with standard T20 teams & groups"**
   - Click **Create Tournament**
   - ✨ System creates 20 teams in 4 groups with all matches!

### Daily Operations

**After Each Match:**
1. Go to **Admin Panel** → **Update Scores** tab
2. Select tournament
3. Choose incomplete match
4. Enter both teams' scores
5. Select winner
6. Click **Update Score**
7. ✅ Match marked as completed
8. ✅ Users can now create fantasy teams for this match

### Viewing Tournament

**You can see:**
- **Tournament Home**: View all matches, group standings, results
- **Leaderboard**: User rankings and points
- **Admin Stats**: Tournament progress overview

---

## For Users

### Getting Started

1. **Login/Register**
   - Create account or login
   - You'll see **🏆 Tournament** in the menu

2. **Explore Tournament**
   - Go to **🏆 Tournament** → **Tournament Home**
   - See matches and group standings
   - Watch for match results

3. **Create Fantasy Team**
   - After a match is **COMPLETED**:
     - Go to **🏆 Tournament** → **Fantasy Cricket**
     - Select tournament and match
     - Pick 11 players from both teams
     - Assign positions (1-11)
     - Choose Captain & Vice-Captain
     - Submit team
   - 🎉 Your team is live! Points update when all results are finalized

4. **Check Leaderboard**
   - Go to **🏆 Tournament** → **Leaderboard**
   - See your rank and total points
   - Track progress towards 🥇 championship

---

## 📱 Navigation Summary

### For Admin
```
Main Menu
  ├─ 🏏 Cricket Analysis (use as normal)
  ├─ 🏆 Tournament
  │   ├─ Tournament Home (view)
  │   ├─ Fantasy Cricket (view)
  │   └─ Leaderboard (view)
  └─ ⚙️ Admin Panel ← YOUR CONTROL CENTER
      ├─ Create Tournament
      ├─ Manage Matches
      └─ Update Scores
```

### For Regular Users
```
Main Menu
  ├─ 🏏 Cricket Analysis (use as normal)
  └─ 🏆 Tournament
      ├─ Tournament Home (view matches & standings)
      ├─ Fantasy Cricket (create teams after matches end)
      └─ Leaderboard (check your ranking)
```

---

## 🎯 Important Rules

✅ **DO:**
- Create fantasy teams ONLY after match is completed
- Include exactly 11 players
- Select 1 captain and 1 vice-captain
- Update match scores after games end

❌ **DON'T:**
- Create teams before match is played
- Add more than 11 players
- Change scores multiple times
- Let non-admin users access admin panel

---

## 💾 Database Info

All tournament data is stored in `cricket_dashboard.db`:
- Tournaments & teams
- Matches & results
- User fantasy teams
- Leaderboard & scores

⚠️ **Backup your database regularly!**

---

## 🆘 Troubleshooting

### "Admin Panel not showing"
- Make sure you're logged in as `admin`
- Check username exactly matches `admin`

### "Can't create fantasy teams"
- Match must be marked as "completed"
- Admin needs to update the score first

### "Leaderboard is empty"
- Users haven't created fantasy teams yet
- Or no matches are completed

### Syntax errors in terminal
- All files compiled successfully ✅
- Try refreshing the page
- Restart Streamlit if needed

---

## 📊 Quick Stats

- **Teams**: 20 (4 groups × 5 teams)
- **Group Stage Matches**: 24 (6 per group)
- **Total Matches**: 27 (24 group + 2 semi-finals + 1 final)
- **Players per Team**: 11 in fantasy squad
- **Max Users**: Unlimited

---

## 🔐 Security

- Only `admin` user sees admin panel
- Admin password protects tournament management
- User data is encrypted in database
- Match results can only be updated by admin

---

## 📞 Need Help?

Check these files:
- [T20_FANTASY_CRICKET.md](T20_FANTASY_CRICKET.md) - Full documentation
- [main.py](main.py) - Application routing
- [src/database.py](src/database.py) - Database schema

---

**Ready to launch? Create your tournament now! 🚀**
