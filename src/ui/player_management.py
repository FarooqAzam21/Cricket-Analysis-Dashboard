"""
Admin Panel - Player Statistics Dashboard for World Cup
Shows top performers and player statistics from CSV data
"""
import streamlit as st
import pandas as pd
from ..config import DATA_PATHS
from ..database import get_db_connection, get_tournament_teams
import os

def safe_int(value, default=0):
    """Safely convert value to int, handling None and NaN"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    """Safely convert value to float, handling None and NaN"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def load_player_data():
    """Load player data from CSV files and filter by tournament selection.
    Prioritizes wc_players.csv if it exists.
    """
    try:
        # Check if WC-specific players CSV exists
        if os.path.exists(DATA_PATHS.get("wc_players", "")):
            all_data = pd.read_csv(DATA_PATHS["wc_players"])
            # Ensure Format/format column exists for T20 filtering later
            if 'Format' not in all_data.columns and 'format' not in all_data.columns:
                all_data['Format'] = 'T20' # Assume T20 if from WC CSV
        else:
            # Fallback to standard sources
            df_bat = pd.read_csv(DATA_PATHS["batsman"]) if os.path.exists(DATA_PATHS["batsman"]) else pd.DataFrame()
            df_ar = pd.read_csv(DATA_PATHS["all_rounder"]) if os.path.exists(DATA_PATHS["all_rounder"]) else pd.DataFrame()
            df_bowl = pd.read_csv(DATA_PATHS["bowler"]) if os.path.exists(DATA_PATHS["bowler"]) else pd.DataFrame()
            
            # Robust column renaming for bowlers
            if not df_bowl.empty:
                if 'bowling_strike_rate' in df_bowl.columns:
                    # If strike_rate already exists, drop it before renaming bowling_strike_rate to avoid duplicates
                    if 'strike_rate' in df_bowl.columns:
                        df_bowl.drop(columns=['strike_rate'], inplace=True)
                    df_bowl.rename(columns={'bowling_strike_rate': 'strike_rate'}, inplace=True)
            
            # Combine all global players
            all_data = pd.concat([df_bat, df_ar, df_bowl], ignore_index=True, sort=False)
        
        # --- Filter by World Cup Selected Players ---
        try:
            conn = get_db_connection()
            # Get latest tournament
            t = conn.execute("SELECT id FROM tournaments ORDER BY id DESC LIMIT 1").fetchone()
            if t:
                tournament_id = t['id']
                teams = get_tournament_teams(tournament_id)
                wc_player_names = set()
                for team in teams:
                    if team['players']:
                        # Handle both comma-separated strings and possible lists
                        plist = team['players'].split(',') if isinstance(team['players'], str) else []
                        wc_player_names.update([p.strip().lower() for p in plist])
                
                if wc_player_names:
                    # Filter all_data where player name is in wc_player_names
                    # and ensure we match case-insensitively
                    all_data = all_data[all_data['player'].str.lower().isin(wc_player_names)]
            conn.close()
        except Exception as db_e:
            print(f"Database filtering error: {db_e}")
            # Fallback to global data if DB fails
            
        # --- Filter by T20 Format Only ---
        if not all_data.empty:
            # Check for column case (Format vs format)
            format_col = 'Format' if 'Format' in all_data.columns else 'format' if 'format' in all_data.columns else None
            if format_col:
                all_data = all_data[all_data[format_col].str.contains('t20', case=False, na=False)]
        
        # Convert numeric columns
        numeric_cols = ['matches', 'Innings', 'NO', 'runs', 'wickets', 'average', 
                       'bowling_average', 'strike_rate', '100s', '50s', '6s', "6's", 'economy']
        
        # Drop duplicate columns if any managed to slip through
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
        
        for col in numeric_cols:
            if col in all_data.columns:
                all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
        
        return all_data
    except Exception as e:
        import traceback
        print(f"Player data error: {traceback.format_exc()}")
        st.error(f"Error loading player data: {e}")
        return pd.DataFrame()

def render_player_management():
    """Render player statistics dashboard"""
    
    st.markdown("""
    <style>
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: #e2e8f0;
        text-align: center;
        margin: 10px 0;
    }
    .stat-title {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    .stat-value {
        font-size: 32px;
        font-weight: bold;
    }
    .stat-player {
        font-size: 12px;
        opacity: 0.8;
        margin-top: 5px;
    }
    .tab-section {
        padding: 20px 0;
    }
    @media (max-width: 768px) {
        .stat-card {
            padding: 15px;
        }
        .stat-value {
            font-size: 24px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='stat-card'>
        <div class='stat-title'>🏏 World Cup Player Statistics</div>
        <p style='margin: 0; font-size: 14px;'>Top performers and key statistics for the upcoming tournament</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    all_data = load_player_data()
    
    if all_data.empty:
        st.error("❌ No player data available")
        return
    
    # Create tabs for different statistics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Top Run Scorers", 
        "🎲 Top Wicket Takers", 
        "⚡ Best Strike Rate",
        "🎳 Best Bowling Average",
        "🔥 Most 6's"
    ])
    
    # ===== TAB 1: TOP RUN SCORERS =====
    with tab1:
        st.markdown("### Top Run Scorers")
        
        # Get top 20 run scorers (with minimum 5 matches)
        scorers = all_data.dropna(subset=['runs', 'matches'])
        scorers = scorers[scorers['matches'] >= 5].sort_values('runs', ascending=False).head(20)
        
        if not scorers.empty:
            col1, col2, col3 = st.columns(3)
            
            # Top 3 highlighted
            for idx, (i, row) in enumerate(scorers.head(3).iterrows()):
                col = [col1, col2, col3][idx]
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='stat-title'>#{idx+1} Run Scorer</div>
                        <div class='stat-value'>{safe_int(row['runs']):,}</div>
                        <div class='stat-player'>{row['player']}</div>
                        <div class='stat-player'>{row.get('Team', 'N/A')} • {safe_int(row['matches'])} M</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("**Full Rankings:**")
            display_df = scorers[['player', 'Team', 'Format', 'matches', 'runs', 'average', 'strike_rate']].copy()
            display_df.columns = ['Player', 'Team', 'Format', 'Matches', 'Runs', 'Average', 'Strike Rate']
            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, width="stretch")
        else:
            st.info("No run scorer data available")
    
    # ===== TAB 2: TOP WICKET TAKERS =====
    with tab2:
        st.markdown("### Top Wicket Takers")
        
        # Get top 20 wicket takers (with minimum 3 matches)
        bowlers = all_data.dropna(subset=['wickets', 'matches'])
        bowlers = bowlers[bowlers['matches'] >= 3].sort_values('wickets', ascending=False).head(20)
        
        if not bowlers.empty:
            col1, col2, col3 = st.columns(3)
            
            # Top 3 highlighted
            for idx, (i, row) in enumerate(bowlers.head(3).iterrows()):
                col = [col1, col2, col3][idx]
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                        <div class='stat-title'>#{idx+1} Wicket Taker</div>
                        <div class='stat-value'>{safe_int(row['wickets'])}</div>
                        <div class='stat-player'>{row['player']}</div>
                        <div class='stat-player'>{row.get('Team', 'N/A')} • {safe_int(row['matches'])} M</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("**Full Rankings:**")
            display_df = bowlers[['player', 'Team', 'Format', 'matches', 'wickets', 'bowling_average', 'economy']].copy()
            display_df.columns = ['Player', 'Team', 'Format', 'Matches', 'Wickets', 'Bowling Avg', 'Economy']
            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, width="stretch")
        else:
            st.info("No wicket taker data available")
    
    # ===== TAB 3: BEST STRIKE RATE =====
    with tab3:
        st.markdown("### Best Strike Rate")
        
        # Get top strike rates (with minimum 20 runs)
        strikers = all_data.dropna(subset=['strike_rate', 'runs'])
        strikers = strikers[strikers['runs'] >= 20].sort_values('strike_rate', ascending=False).head(20)
        
        if not strikers.empty:
            col1, col2, col3 = st.columns(3)
            
            # Top 3 highlighted
            for idx, (i, row) in enumerate(strikers.head(3).iterrows()):
                col = [col1, col2, col3][idx]
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);'>
                        <div class='stat-title'>#{idx+1} Strike Rate</div>
                        <div class='stat-value'>{safe_float(row['strike_rate'], 0):.2f}</div>
                        <div class='stat-player'>{row['player']}</div>
                        <div class='stat-player'>{row.get('Team', 'N/A')} • {safe_int(row['runs'])} Runs</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("**Full Rankings:**")
            display_df = strikers[['player', 'Team', 'Format', 'runs', 'matches', 'strike_rate', 'average']].copy()
            display_df.columns = ['Player', 'Team', 'Format', 'Runs', 'Matches', 'Strike Rate', 'Average']
            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, width="stretch")
        else:
            st.info("No strike rate data available")
    
    # ===== TAB 4: BEST BOWLING AVERAGE =====
    with tab4:
        st.markdown("### Best Bowling Average")
        
        # Get best bowling averages (with minimum 5 wickets)
        best_bowlers = all_data.dropna(subset=['bowling_average', 'wickets'])
        best_bowlers = best_bowlers[best_bowlers['wickets'] >= 5].sort_values('bowling_average', ascending=True).head(20)
        
        if not best_bowlers.empty:
            col1, col2, col3 = st.columns(3)
            
            # Top 3 highlighted
            for idx, (i, row) in enumerate(best_bowlers.head(3).iterrows()):
                col = [col1, col2, col3][idx]
                with col:
                    st.markdown(f"""
                    <div class='stat-card' style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);'>
                        <div class='stat-title'>#{idx+1} Bowling Avg</div>
                        <div class='stat-value'>{safe_float(row['bowling_average'], 0):.2f}</div>
                        <div class='stat-player'>{row['player']}</div>
                        <div class='stat-player'>{row.get('Team', 'N/A')} • {safe_int(row['wickets'])} Wkts</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("**Full Rankings:**")
            display_df = best_bowlers[['player', 'Team', 'Format', 'wickets', 'matches', 'bowling_average', 'economy']].copy()
            display_df.columns = ['Player', 'Team', 'Format', 'Wickets', 'Matches', 'Bowling Avg', 'Economy']
            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df, width="stretch")
        else:
            st.info("No bowling average data available")
    
    # ===== TAB 5: MOST 6'S =====
    with tab5:
        st.markdown("### Most 6's")
        
        # Get most 6s (if column exists)
        if '6s' in all_data.columns:
            sixes = all_data.dropna(subset=['6s'])
            sixes = sixes.sort_values('6s', ascending=False).head(20)
            
            if not sixes.empty and (sixes['6s'] > 0).any():
                col1, col2, col3 = st.columns(3)
                
                # Top 3 highlighted
                for idx, (i, row) in enumerate(sixes.head(3).iterrows()):
                    col = [col1, col2, col3][idx]
                    with col:
                        st.markdown(f"""
                        <div class='stat-card' style='background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);'>
                            <div class='stat-title'>#{idx+1} Most 6's</div>
                            <div class='stat-value'>{safe_int(row['6s'])}</div>
                            <div class='stat-player'>{row['player']}</div>
                            <div class='stat-player'>{row.get('Team', 'N/A')} • {safe_int(row['runs'])} Runs</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("**Full Rankings:**")
                display_df = sixes[['player', 'Team', 'Format', '6s', 'runs', 'strike_rate', 'matches']].copy()
                display_df.columns = ['Player', 'Team', 'Format', '6s', 'Runs', 'Strike Rate', 'Matches']
                display_df = display_df.reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df, width="stretch")
            else:
                st.info("No 6's data available in current CSV")
        else:
            st.warning("⚠️ The '6s' column not found in player data. Add this column to CSV for this feature.")
    
    # ===== INFO SECTION =====
    st.markdown("---")
    with st.expander("ℹ️ How to Update Data"):
        st.info("""
        **To update player statistics:**
        1. Edit the CSV files manually (odi_batsman.csv, odi_bowler.csv, odi_all_rounders.csv)
        2. Add complete match details for World Cup players
        3. Save the CSV files
        4. The statistics will refresh automatically on this page
        
        **CSV Columns to Update:**
        - `player`: Player name
        - `Team`: Team name
        - `Format`: ODI/T20/Test
        - `matches`: Total matches
        - `runs`: Total runs (for batsmen)
        - `wickets`: Total wickets (for bowlers)
        - `strike_rate`: Strike rate percentage
        - `average`: Batting average
        - `bowling_average`: Bowling average
        - `economy`: Economy rate
        - `100s`: Centuries
        - `50s`: Half centuries
        - `6s`: Sixes hit (if tracking)
        """)
