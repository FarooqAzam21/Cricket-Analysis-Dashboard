-- T20 World Cup Fantasy Cricket System
-- Database Schema Reference
-- SQLite3

-- ============================================================
-- TOURNAMENT MANAGEMENT TABLES
-- ============================================================

-- Tournaments
-- Stores all tournament information
CREATE TABLE tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    status TEXT DEFAULT 'planning',
    start_date TEXT,
    end_date TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tournament Teams
-- Stores teams participating in each tournament
CREATE TABLE tournament_teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER,
    team_name TEXT,
    group_letter TEXT,
    squad TEXT,
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    points INTEGER DEFAULT 0,
    FOREIGN KEY(tournament_id) REFERENCES tournaments(id)
);

-- Tournament Matches
-- Stores match schedule and results
CREATE TABLE tournament_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER,
    team1_id INTEGER,
    team2_id INTEGER,
    match_date TEXT,
    stage TEXT,  -- 'group', 'semi-final', 'final'
    group_letter TEXT,
    status TEXT DEFAULT 'scheduled',  -- 'scheduled', 'completed', 'no_result'
    winner_id INTEGER,
    team1_score INTEGER,
    team2_score INTEGER,
    FOREIGN KEY(tournament_id) REFERENCES tournaments(id),
    FOREIGN KEY(team1_id) REFERENCES tournament_teams(id),
    FOREIGN KEY(team2_id) REFERENCES tournament_teams(id)
);

-- ============================================================
-- FANTASY CRICKET TABLES
-- ============================================================

-- Fantasy Teams
-- Stores user's fantasy team selections for each match
CREATE TABLE fantasy_teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    tournament_id INTEGER,
    match_id INTEGER,
    players_json TEXT,  -- JSON: {players: [], positions: {}, captain: '', vice_captain: ''}
    captain_id INTEGER,
    vice_captain_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(tournament_id) REFERENCES tournaments(id),
    FOREIGN KEY(match_id) REFERENCES tournament_matches(id)
);

-- Fantasy Scores
-- Stores calculated points for each fantasy team
CREATE TABLE fantasy_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fantasy_team_id INTEGER,
    total_score REAL DEFAULT 0,
    rank INTEGER,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(fantasy_team_id) REFERENCES fantasy_teams(id)
);

-- ============================================================
-- LEADERBOARD TABLE
-- ============================================================

-- Leaderboard
-- Stores user rankings and total points
CREATE TABLE leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    tournament_id INTEGER,
    total_points REAL DEFAULT 0,
    fantasy_teams_created INTEGER DEFAULT 0,
    rank INTEGER,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(tournament_id) REFERENCES tournaments(id)
);

-- ============================================================
-- EXISTING TABLES (REFERENCE)
-- ============================================================

-- Players (existing)
CREATE TABLE players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player TEXT,
    team TEXT,
    format TEXT,
    matches INTEGER,
    innings INTEGER,
    no INTEGER,
    runs INTEGER,
    wickets INTEGER,
    average REAL,
    strike_rate REAL,
    bowling_average REAL,
    economy REAL,
    hundreds INTEGER,
    fifties INTEGER,
    batting_position INTEGER,
    role TEXT,
    image_url TEXT,
    UNIQUE(player, team, format)
);

-- Users (existing)
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
);

-- Scout Feedback (existing)
CREATE TABLE scout_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    source_player TEXT,
    similar_player TEXT,
    format TEXT,
    rating TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Tournament queries
CREATE INDEX idx_tournaments_status ON tournaments(status);
CREATE INDEX idx_tournament_teams_tournament ON tournament_teams(tournament_id);
CREATE INDEX idx_tournament_teams_group ON tournament_teams(group_letter);
CREATE INDEX idx_tournament_matches_tournament ON tournament_matches(tournament_id);
CREATE INDEX idx_tournament_matches_status ON tournament_matches(status);
CREATE INDEX idx_tournament_matches_stage ON tournament_matches(stage);

-- Fantasy queries
CREATE INDEX idx_fantasy_teams_user ON fantasy_teams(user_id);
CREATE INDEX idx_fantasy_teams_tournament ON fantasy_teams(tournament_id);
CREATE INDEX idx_fantasy_teams_match ON fantasy_teams(match_id);
CREATE INDEX idx_fantasy_scores_team ON fantasy_scores(fantasy_team_id);

-- Leaderboard queries
CREATE INDEX idx_leaderboard_user ON leaderboard(user_id);
CREATE INDEX idx_leaderboard_tournament ON leaderboard(tournament_id);
CREATE INDEX idx_leaderboard_rank ON leaderboard(rank);

-- ============================================================
-- SAMPLE DATA INSERTION
-- ============================================================

-- Insert a sample tournament
-- INSERT INTO tournaments (name, start_date, end_date, status)
-- VALUES ('T20 World Cup 2024', '2024-06-01', '2024-07-14', 'planning');

-- Insert sample teams for Group A
-- INSERT INTO tournament_teams (tournament_id, team_name, group_letter)
-- VALUES 
-- (1, 'India', 'A'),
-- (1, 'Pakistan', 'A'),
-- (1, 'Afghanistan', 'A'),
-- (1, 'Australia', 'A'),
-- (1, 'Sri Lanka', 'A');

-- Insert sample match
-- INSERT INTO tournament_matches (tournament_id, team1_id, team2_id, match_date, stage, group_letter, status)
-- VALUES (1, 1, 2, '2024-06-01', 'group', 'A', 'scheduled');

-- ============================================================
-- USEFUL QUERIES
-- ============================================================

-- Get tournament standings
-- SELECT team_name, matches_played, wins, losses, points 
-- FROM tournament_teams 
-- WHERE group_letter = 'A' 
-- ORDER BY points DESC;

-- Get user's fantasy teams
-- SELECT ft.id, ft.created_at, tm.match_date, fs.total_score
-- FROM fantasy_teams ft
-- JOIN tournament_matches tm ON ft.match_id = tm.id
-- LEFT JOIN fantasy_scores fs ON ft.id = fs.fantasy_team_id
-- WHERE ft.user_id = 1
-- ORDER BY ft.created_at DESC;

-- Get leaderboard
-- SELECT u.username, l.total_points, l.rank
-- FROM leaderboard l
-- JOIN users u ON l.user_id = u.id
-- WHERE l.tournament_id = 1
-- ORDER BY l.total_points DESC;

-- ============================================================
-- MAINTENANCE
-- ============================================================

-- Reset tournament (admin only)
-- DELETE FROM fantasy_scores;
-- DELETE FROM fantasy_teams;
-- DELETE FROM leaderboard;
-- DELETE FROM tournament_matches;
-- DELETE FROM tournament_teams;
-- DELETE FROM tournaments;

-- Recalculate leaderboard
-- DELETE FROM leaderboard;
-- INSERT INTO leaderboard (user_id, tournament_id, total_points)
-- SELECT ft.user_id, ft.tournament_id, COALESCE(SUM(fs.total_score), 0)
-- FROM fantasy_teams ft
-- LEFT JOIN fantasy_scores fs ON ft.id = fs.fantasy_team_id
-- GROUP BY ft.user_id, ft.tournament_id;
