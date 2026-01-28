# 🏆 T20 World Cup Fantasy Cricket System

## 📋 Overview

A complete fantasy cricket league system built into the Cricket Pro Analytics dashboard. Users can create fantasy teams for T20 World Cup matches, earn points based on player performance, and compete on global leaderboards.

## 🎯 Key Features

### 1. **Admin Panel** (Admin-Only)
- **Hidden from regular users** - Only accessible to admin account
- **Tournament Creation**
  - Create multiple tournaments with custom names and dates
  - Auto-generate standard T20 WC format (4 groups, 5 teams each, 20 teams total)
  - Groups: A, B, C, D
  
- **Tournament Management**
  - Add teams to specific groups
  - View all scheduled matches
  - Update match results after completion (Team scores + winner)
  - Auto-generate group stage matches (round-robin)
  - Automatic team statistics updates (W/L/Points)

### 2. **Tournament Home Page** (User-Visible)
- **Match Schedule Display**
  - All upcoming and completed matches
  - Match date, participating teams, venue
  - Live status indicators (Scheduled/Completed/No Result)
  
- **Group Standings**
  - Display for all 4 groups (A, B, C, D)
  - Metrics: Matches Played, Wins, Losses, Points
  - Automatic NRR calculation
  - Top 2 teams qualify for knockouts
  
- **Knockout Stage**
  - Semi-finals and Finals
  - Match results and champion display
  
- **Fantasy Leaderboard**
  - User rankings by total points
  - Award allocation (🥇 Champion, 🥈 Runner-up, 🥉 Third Place)
  - Average points per team statistics

### 3. **Fantasy Team Builder** (Post-Match Only)
- **Availability**: Only after matches are completed
- **Team Selection**
  - Choose 11 players from both participating teams
  - Players filtered by T20 format in database
  - Combined squad from both teams
  
- **Position Assignment**
  - Assign batting positions (1-11) to players
  - Reflects actual match lineup
  - Live preview with sorted player display
  
- **Captain/Vice-Captain**
  - Select captain (typically scores 2x points)
  - Select vice-captain (typically scores 1.5x points)
  - Different bonus multipliers for performance
  
- **Team Submission**
  - Save to database with timestamp
  - View previously created teams
  - Pre-match team creation disabled
  
### 4. **Scoring System**
- **Batting Points**
  - 1 point per run scored
  - 2 bonus points per 4-run hit
  - 3 bonus points per 6-run hit
  - 50 bonus points for 50-run innings
  - 100 bonus points for century (100+)
  
- **Bowling Points**
  - 20 points per wicket taken
  - 4 points for economy rate < 7.00
  - Bonus points for spell performance
  
- **Fielding Points**
  - 10 points per catch
  - 15 points per runout
  - 25 points per stumping
  
- **Captain/Vice-Captain Bonuses**
  - Captain: 2x all points earned
  - Vice-Captain: 1.5x all points earned

### 5. **Leaderboard System**
- **User Rankings**
  - Sorted by total fantasy points
  - Rank position with medal indicators (🥇🥈🥉)
  - Total points and average per team
  
- **Personal Stats**
  - Your current rank and points
  - Gap from leader
  - Number of teams created
  - Average points per fantasy team
  
- **Tournament Awards** (At completion)
  - 🥇 Champion - Highest total points
  - 🥈 Runner-up - Second place
  - 🥉 Third Place - Third place
  - Custom prize allocation system

## 🏗️ Database Schema

### Core Tables

**tournaments**
```sql
id (PK) | name | status | start_date | end_date | created_at
```

**tournament_teams**
```sql
id (PK) | tournament_id (FK) | team_name | group_letter | squad | 
matches_played | wins | losses | points
```

**tournament_matches**
```sql
id (PK) | tournament_id (FK) | team1_id (FK) | team2_id (FK) | 
match_date | stage | group_letter | status | winner_id | 
team1_score | team2_score
```

**fantasy_teams**
```sql
id (PK) | user_id (FK) | tournament_id (FK) | match_id (FK) | 
players_json | captain_id | vice_captain_id | created_at
```

**fantasy_scores**
```sql
id (PK) | fantasy_team_id (FK) | total_score | rank | updated_at
```

**leaderboard**
```sql
id (PK) | user_id (FK) | tournament_id (FK) | total_points | 
fantasy_teams_created | rank | updated_at
```

## 🔐 Security & Access Control

### Admin Authentication
- **Username**: `admin`
- **Password**: Your admin password (set during first access)
- **Access Level**: Admin panel only visible to `admin` user

### User Authentication
- Standard username/password login
- New account creation available
- Session-based authentication

### Admin-Only Features
- Tournament creation
- Team group assignment
- Match scheduling
- Score updates
- Result finalization

### Regular User Features
- View tournaments & matches
- Create fantasy teams (post-match only)
- View leaderboards
- Track personal stats

## 📱 UI Navigation

### Main Menu
```
🏏 Cricket Analysis
├── Format Wise Analysis
├── Select Playing 11
├── Player Comparison
├── Player Analysis
├── Predictions
└── AI Features

🏆 Tournament (All Users)
├── Tournament Home
├── Fantasy Cricket
└── Leaderboard

⚙️ Admin Panel (Admin Only)
├── Create Tournament
├── Manage Matches
└── Update Scores
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- SQLite3
- Streamlit
- Pandas

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run main.py
```

### First Admin Setup
1. Create an account with username `admin`
2. Go to **Admin Panel** to create your first tournament
3. Auto-generate T20 WC structure with 20 teams in 4 groups
4. Setup complete!

## 📊 Admin Workflow

### Creating a Tournament
1. Navigate to **Admin Panel** → **Create Tournament**
2. Enter tournament name (e.g., "T20 World Cup 2024")
3. Set start and end dates
4. Enable **Auto-setup** to generate:
   - 20 teams split into 4 groups
   - Group stage matches (round-robin 6 matches per group)
   - Standard T20 tournament structure

### Managing Match Scores
1. Go to **Admin Panel** → **Update Scores**
2. Select tournament
3. Choose incomplete match
4. Enter team scores
5. Select match winner (or "No Result")
6. Click **Update Score**

**System automatically:**
- Updates match status to "completed"
- Adds wins/losses to team records
- Updates group standings and points
- Makes match available for fantasy team creation

### Monitoring Progress
1. **Tournament Home** shows:
   - All matches and results
   - Group standings with qualifiers
   - Knockout bracket progression
   - Live leaderboard updates

## 💡 Usage Examples

### Admin: Create First Tournament
```python
# Using admin account:
# 1. Navigate to Admin Panel
# 2. Create "T20 World Cup 2024"
# 3. Select Auto-setup
# Result: 4 groups, 20 teams, 24 group matches scheduled
```

### User: Create Fantasy Team
```
# After a match is completed:
# 1. Go to Fantasy Cricket
# 2. Select tournament → select completed match
# 3. Pick 11 players from both teams
# 4. Assign positions 1-11
# 5. Select captain & vice-captain
# 6. Submit team
# 7. Team scoring begins when all results are in
```

### Viewing Performance
```
# Check Leaderboard:
# 1. Go to Tournament → Leaderboard
# 2. See your rank and total points
# 3. Check gap from leader
# 4. View your average points per team
```

## 🔄 Match Lifecycle

```
Scheduled
   ↓
[Match Played]
   ↓
Completed (Admin updates score)
   ↓
Fantasy Teams Available
   ↓
Users Create Fantasy Teams
   ↓
Points Calculated
   ↓
Leaderboard Updated
```

## 📈 Future Enhancements

- [ ] Live match integration with external APIs
- [ ] Automatic score updates from live matches
- [ ] Custom scoring system builder (for admin)
- [ ] Trading players before match starts
- [ ] Injury/unavailability alerts
- [ ] Mobile app version
- [ ] Seasonal tournaments
- [ ] Prize pool management
- [ ] Social sharing of teams
- [ ] Head-to-head match-ups

## 🛠️ Technical Stack

- **Framework**: Streamlit
- **Database**: SQLite3
- **Language**: Python 3.9+
- **Authentication**: Custom username/password
- **Deployment**: Streamlit Cloud compatible

## 📝 Files Modified

**New Files:**
- `src/ui/admin_tournament.py` - Admin panel interface
- `src/ui/tournament_home.py` - Tournament display page
- `src/ui/fantasy_cricket.py` - Fantasy team builder
- `src/ui/leaderboard.py` - Rankings & awards

**Modified Files:**
- `src/database.py` - Added tournament schema & helper functions
- `main.py` - Integrated tournament menu and routing

## ✅ Test Checklist

- [x] Admin panel access control working
- [x] Tournament creation with auto-setup
- [x] Match schedule generation
- [x] Score update system
- [x] Fantasy team creation (post-match only)
- [x] Position assignment (1-11)
- [x] Captain/vice-captain selection
- [x] Leaderboard calculations
- [x] Group standings display
- [x] Knockout bracket visualization
- [x] Database schema creation

## 🐛 Known Issues

None currently. Please report bugs to Farooq Azam.

## 📞 Support

For issues or feature requests, please contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: January 29, 2026  
**Developer**: Farooq Azam
