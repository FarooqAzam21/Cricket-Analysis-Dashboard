# 🏏 Cricket Pro Analytics - T20 Fantasy Cricket System

## 🎯 Overview

Welcome to the **Cricket Pro Analytics Dashboard** with integrated **T20 World Cup Fantasy Cricket System**!

This is a complete Streamlit-based application that combines:
- **Cricket Player Analysis** - Comprehensive player statistics, predictions, and insights
- **Smart Scouting** - AI-powered player comparison and recommendations  
- **Fantasy Cricket** - Create fantasy teams and compete on leaderboards
- **Admin Panel** - Complete tournament management system

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/FarooqAzam21/Cricket-Analysis-Dashboard.git
cd Cricket-Analysis-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### First Login
- Username: `admin`
- Password: (your admin password)

**First Time Admin?** Read [ADMIN_QUICKSTART.md](ADMIN_QUICKSTART.md)

---

## 📱 Navigation Guide

### Main Menu Structure
```
LOGIN/SIGNUP
    ↓
DASHBOARD
├─ 🏏 CRICKET ANALYSIS
│  ├─ Format Wise Analysis
│  ├─ Select Playing 11
│  ├─ Player Comparison
│  ├─ Player Analysis
│  ├─ Next Match Prediction
│  ├─ Yearly Performance Prediction
│  ├─ Smart Scout (AI)
│  └─ Ask Expert (AI)
│
├─ 🏆 TOURNAMENT
│  ├─ Tournament Home (Matches & Standings)
│  ├─ Fantasy Cricket (Create Teams)
│  └─ Leaderboard (Rankings)
│
└─ ⚙️ ADMIN PANEL (Admin Only)
   ├─ Create Tournament
   ├─ Manage Matches
   └─ Update Scores
```

---

## 📊 Features Breakdown

### 🏏 Cricket Analysis (Existing)
Comprehensive player analytics with:
- Player statistics by format (ODI/T20/Test)
- Performance analysis and trends
- Prediction models
- AI-powered insights and scouting
- Player comparison tools

### 🏆 Tournament System (NEW!)

#### Tournament Home
- **Match Schedule**: View all upcoming and completed matches
- **Group Standings**: Track points, wins, losses for each team
- **Knockout Bracket**: Semi-finals and finals progression
- **Results**: Live match results and scorecards

#### Fantasy Cricket
- **Create Teams**: After matches complete
- **11-Player Squads**: Select from both participating teams
- **Position Assignment**: Batting order 1-11
- **Captain/VC**: Choose captain (2x multiplier) and vice-captain (1.5x)
- **Points**: Automatic calculation based on player performance
- **Team History**: View all previously created teams

#### Leaderboard
- **User Rankings**: Sorted by total fantasy points
- **Awards**: 🥇 Champion, 🥈 Runner-up, 🥉 Third Place
- **Personal Stats**: Your rank, total points, average per team
- **Gap Tracking**: Distance from current leader

### ⚙️ Admin Panel (NEW!)

#### Tournament Creation
- Create new tournaments with custom dates
- Auto-generate T20 WC format (20 teams, 4 groups)
- Auto-schedule all group stage matches
- Support for knockout stages

#### Match Management
- View all tournament matches
- Filter by stage and group
- Update match results
- Auto-update team standings

#### Score Updates
- Enter team scores post-match
- Select match winner
- Auto-calculate group standings
- Enable fantasy team creation

---

## 🔐 Security & Access

### User Roles
- **Regular User**: Can view tournaments, create fantasy teams, see leaderboards
- **Admin User**: Full tournament management, score updates, tournament creation

### Authentication
- Username/password login system
- Account creation available
- Session-based authentication
- Admin access control via username check

### Admin Panel Security
```python
# Only username 'admin' can access admin panel
if st.session_state.username != 'admin':
    st.error("⛔ Unauthorized Access")
    st.stop()
```

---

## 📊 Database Structure

### Core Tables
- **users** - User accounts and passwords
- **tournaments** - Tournament metadata
- **tournament_teams** - Teams in tournament
- **tournament_matches** - Match schedule and results
- **fantasy_teams** - User fantasy team selections
- **fantasy_scores** - Calculated points per team
- **leaderboard** - User rankings and total points

### Additional Tables
- **players** - Cricket player data
- **scout_feedback** - User feedback on player comparisons

See [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql) for complete schema.

---

## 🎮 How to Use

### For Regular Users

#### 1. View Tournament
```
1. Login with your account
2. Navigate to 🏆 Tournament → Tournament Home
3. See all matches, group standings, knockout bracket
4. Track which teams are advancing
```

#### 2. Create Fantasy Team
```
1. After a match is COMPLETED:
   - Go to 🏆 Tournament → Fantasy Cricket
   - Select the tournament and match
   - Pick 11 players from both teams
   - Assign positions (1-11)
   - Select captain and vice-captain
   - Submit team
2. Your team is live! Points update automatically
```

#### 3. Check Leaderboard
```
1. Go to 🏆 Tournament → Leaderboard
2. See your rank and total points
3. View other users' scores
4. Track progress towards 🥇 championship
```

### For Admin Users

#### 1. Create Tournament
```
1. Login as admin
2. Go to ⚙️ Admin Panel → Create Tournament
3. Enter tournament name and dates
4. Check "Auto-setup with standard T20 teams & groups"
5. Click Create Tournament
   → System auto-generates:
      • 4 groups (A, B, C, D)
      • 20 teams (5 per group)
      • All group stage matches (24 total)
```

#### 2. Update Match Scores
```
1. After a match is played:
   - Go to ⚙️ Admin Panel → Update Scores
   - Select tournament and incomplete match
   - Enter both teams' scores
   - Select match winner
   - Click Update Score
   
   System automatically:
   - Marks match as completed
   - Updates team statistics
   - Updates group standings
   - Enables fantasy team creation for this match
```

#### 3. Monitor Tournament
```
1. View 🏆 Tournament → Tournament Home to see:
   - All matches and results
   - Current group standings
   - Knockout progression
   - Live leaderboard
```

---

## 📈 T20 World Cup Format

### Tournament Structure
- **Teams**: 20 (India, Pakistan, Afghanistan, etc.)
- **Groups**: 4 (A, B, C, D)
- **Teams per Group**: 5
- **Group Stage**: Round-robin (6 matches per group, 24 total)
- **Knockouts**: Top 2 teams from each group
  - Semi-finals: 4 teams
  - Final: 2 teams

### Scoring Points
- **Batting**: 1 point per run, +2 for 4s, +3 for 6s, +50 for 50, +100 for century
- **Bowling**: 20 per wicket, bonus points for economy
- **Fielding**: 10 per catch, 15 per runout, 25 per stumping
- **Captain**: 2x multiplier on all points
- **Vice-Captain**: 1.5x multiplier on all points

---

## 🎯 Key Rules

✅ **Fantasy Teams**
- Can ONLY create after match is completed
- Must select exactly 11 players
- Must assign batting positions (1-11)
- Must select captain and vice-captain (different players)
- Cannot change team after submission

✅ **Match Management** (Admin)
- Update scores ONLY after match is completed
- Select correct winner
- System auto-updates standings

✅ **Leaderboard**
- Updated automatically after scores are entered
- Based on total fantasy points
- Filtered by tournament
- Awards presented at tournament end

---

## 🔧 Configuration

### Environment Variables
No special configuration needed. All settings are database-driven.

### Database Location
```
cricket_dashboard.db
```

### Player Data
CSV files automatically loaded and cached:
- `odi_batsman.csv`
- `odi_bowler.csv`
- `odi_all_rounders.csv`
- `yearwise_data.csv`

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [T20_FANTASY_CRICKET.md](T20_FANTASY_CRICKET.md) | Complete feature documentation |
| [ADMIN_QUICKSTART.md](ADMIN_QUICKSTART.md) | Admin setup and operation guide |
| [DATABASE_SCHEMA.sql](DATABASE_SCHEMA.sql) | Database structure reference |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Technical implementation details |

---

## 🚨 Troubleshooting

### "Admin Panel not visible"
- Make sure you're logged in as `admin`
- Check username exactly matches `admin`
- Try refreshing the page

### "Can't create fantasy teams"
- Match must be marked as "completed"
- Admin needs to update the score first
- Check tournament is active

### "Leaderboard is empty"
- No fantasy teams created yet
- Or no matches are completed
- Users haven't participated yet

### "Database error"
- Check `cricket_dashboard.db` exists in root directory
- Verify write permissions to directory
- Try restarting Streamlit

---

## 📈 Performance Tips

- Close unused browser tabs
- Refresh leaderboard every few matches
- Clear browser cache if data seems stale
- Use desktop version for best experience

---

## 🤝 Contributing

To contribute improvements:
1. Make changes locally
2. Test thoroughly
3. Commit with clear messages
4. Push to GitHub
5. Create pull request

---

## 📝 Version History

**v1.0.0** - January 29, 2026
- ✅ Complete T20 Fantasy Cricket System
- ✅ Admin Panel with tournament management
- ✅ Fantasy team builder
- ✅ Dynamic leaderboard
- ✅ Group standings and knockout tracking

---

## 👥 Team & Credits

**Developer**: Farooq Azam  
**Repository**: https://github.com/FarooqAzam21/Cricket-Analysis-Dashboard  
**Last Updated**: January 29, 2026

---

## 📞 Support

For issues, questions, or feature requests:
- Check documentation files
- Review database schema
- Check admin quick-start guide
- Contact Farooq Azam

---

## 📜 License

Built with ❤️ for cricket enthusiasts everywhere

---

## 🎉 Ready to Play!

**Your T20 Fantasy Cricket League is ready to launch!**

```bash
streamlit run main.py
```

Create your first tournament and start building fantasy teams! 🏏🏆

---

**Happy Fantasy Cricket! 🎯**
