"""
Admin Panel - Player Data Management
Allows admins to search, view, and update player statistics
"""
import streamlit as st
import pandas as pd
from ..database import get_db_connection, fetch_all_players_from_db

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

def get_value(series, col, default=None):
    """Safely get value from pandas Series"""
    try:
        val = series[col]
        if pd.isna(val):
            return default
        return val
    except (KeyError, TypeError):
        return default

def update_player_stats(player_name, team, format_type, stats_dict):
    """Update player statistics in the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        
        for column, value in stats_dict.items():
            if value is not None and value != '':
                # Convert to appropriate type
                if column in ['matches', 'innings', 'no', 'runs', 'wickets', 'hundreds', 'fifties', 'batting_position']:
                    value = int(value) if value else 0
                elif column in ['average', 'strike_rate', 'bowling_average', 'economy']:
                    value = float(value) if value else 0.0
                
                set_clauses.append(f"{column} = ?")
                values.append(value)
        
        if not set_clauses:
            st.error("❌ No valid data to update")
            return False
        
        # Add the WHERE clause values
        values.extend([player_name, team, format_type])
        
        query = f"""
            UPDATE players 
            SET {', '.join(set_clauses)}
            WHERE player = ? AND team = ? AND format = ?
        """
        
        cursor.execute(query, values)
        conn.commit()
        
        rows_affected = cursor.rowcount
        conn.close()
        
        return rows_affected > 0
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return False

def render_player_management():
    """Render player data management interface in admin panel"""
    
    st.markdown("""
    <style>
    .player-update-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .stats-input-container {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    @media (max-width: 768px) {
        .player-update-section {
            padding: 15px;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='player-update-section'>", unsafe_allow_html=True)
    st.subheader("🎯 Player Data Management")
    st.markdown("Update player statistics and performance metrics")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== SECTION 1: SEARCH PLAYER =====
    st.markdown("#### 🔍 Step 1: Search Player")
    
    col1, col2 = st.columns(2)
    with col1:
        # Load all players
        try:
            all_players_df = fetch_all_players_from_db()
            if all_players_df is not None and not all_players_df.empty:
                player_names = sorted(all_players_df['player'].unique().tolist())
                selected_player = st.selectbox("Select Player", player_names, key="player_select")
            else:
                st.warning("⚠️ No players found in database")
                return
        except Exception as e:
            st.error(f"Error loading players: {str(e)}")
            return
    
    with col2:
        st.info(f"✅ Found {len(player_names)} players in database")
    
    # ===== SECTION 2: SELECT FORMAT & TEAM =====
    st.markdown("#### 📋 Step 2: Select Format & Team")
    
    # Get player data for selected player
    player_data = all_players_df[all_players_df['player'] == selected_player]
    
    if player_data.empty:
        st.error(f"❌ No data found for {selected_player}")
        return
    
    # Show available formats for this player
    available_formats = player_data['format'].unique().tolist()
    available_teams = player_data['team'].unique().tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_format = st.selectbox("Select Format", available_formats, key="format_select")
    with col2:
        # Filter teams by selected format
        team_for_format = player_data[player_data['format'] == selected_format]['team'].unique().tolist()
        selected_team = st.selectbox("Select Team", team_for_format, key="team_select")
    
    # ===== SECTION 3: DISPLAY CURRENT STATS =====
    st.markdown("#### 📊 Step 3: Current Statistics")
    
    current_data = player_data[
        (player_data['format'] == selected_format) & 
        (player_data['team'] == selected_team)
    ].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", safe_int(get_value(current_data, 'matches', 0)))
    with col2:
        st.metric("Runs", safe_int(get_value(current_data, 'runs', 0)))
    with col3:
        st.metric("Wickets", safe_int(get_value(current_data, 'wickets', 0)))
    with col4:
        st.metric("Average", round(safe_float(get_value(current_data, 'average', 0)), 2))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strike Rate", round(safe_float(get_value(current_data, 'strike_rate', 0)), 2))
    with col2:
        st.metric("Bowling Avg", round(safe_float(get_value(current_data, 'bowling_average', 0)), 2))
    with col3:
        st.metric("Economy", round(safe_float(get_value(current_data, 'economy', 0)), 2))
    with col4:
        st.metric("Hundreds", safe_int(get_value(current_data, '100s', 0)))
    
    # ===== SECTION 4: UPDATE STATS =====
    st.markdown("#### ✏️ Step 4: Update Statistics")
    st.markdown("---")
    
    # Create input fields organized in columns
    st.markdown("<div class='stats-input-container'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Batting Stats**")
        new_matches = st.number_input("Matches", value=safe_int(get_value(current_data, 'matches', 0)), min_value=0, step=1, key="matches")
        new_innings = st.number_input("Innings", value=safe_int(get_value(current_data, 'Innings', 0)), min_value=0, step=1, key="innings")
        new_no = st.number_input("Not Out", value=safe_int(get_value(current_data, 'NO', 0)), min_value=0, step=1, key="no")
        new_runs = st.number_input("Runs", value=safe_int(get_value(current_data, 'runs', 0)), min_value=0, step=100, key="runs")
        new_sr = st.number_input("Strike Rate", value=round(safe_float(get_value(current_data, 'strike_rate', 0)), 2), min_value=0.0, step=1.0, format="%.2f", key="strike_rate")
        new_batting_pos = st.number_input("Batting Position", value=safe_int(get_value(current_data, 'batting_position', 0)), min_value=0, max_value=11, step=1, key="batting_pos")
    
    with col2:
        st.markdown("**Bowling Stats**")
        new_wickets = st.number_input("Wickets", value=safe_int(get_value(current_data, 'wickets', 0)), min_value=0, step=1, key="wickets")
        new_bowling_avg = st.number_input("Bowling Average", value=round(safe_float(get_value(current_data, 'bowling_average', 0)), 2), min_value=0.0, step=1.0, format="%.2f", key="bowling_average")
        new_economy = st.number_input("Economy", value=round(safe_float(get_value(current_data, 'economy', 0)), 2), min_value=0.0, step=0.1, format="%.2f", key="economy")
    
    with col3:
        st.markdown("**Achievement Stats**")
        new_average = st.number_input("Batting Average", value=round(safe_float(get_value(current_data, 'average', 0)), 2), min_value=0.0, step=1.0, format="%.2f", key="average")
        new_hundreds = st.number_input("Centuries", value=safe_int(get_value(current_data, '100s', 0)), min_value=0, step=1, key="hundreds")
        new_fifties = st.number_input("Half Centuries", value=safe_int(get_value(current_data, '50s', 0)), min_value=0, step=1, key="fifties")
        new_role = st.selectbox("Player Role", 
            ["Batsman", "Bowler", "All-rounder", "Wicket-keeper"],
            index=0,
            key="role"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== SECTION 5: SUBMIT =====
    st.markdown("#### 💾 Step 5: Submit Changes")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col2:
        submit_btn = st.button("✅ Submit & Update Database", use_container_width=True, type="primary")
    
    with col3:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    if reset_btn:
        st.rerun()
    
    # Handle submission
    if submit_btn:
        # Prepare stats dictionary
        stats_update = {
            'matches': new_matches,
            'innings': new_innings,
            'no': new_no,
            'runs': new_runs,
            'wickets': new_wickets,
            'average': new_average,
            'strike_rate': new_sr,
            'bowling_average': new_bowling_avg,
            'economy': new_economy,
            'hundreds': new_hundreds,
            'fifties': new_fifties,
            'batting_position': new_batting_pos,
            'role': new_role
        }
        
        # Show confirmation before update
        with st.spinner("🔄 Updating player data..."):
            success = update_player_stats(selected_player, selected_team, selected_format, stats_update)
        
        if success:
            st.success(f"✅ Successfully updated {selected_player}'s statistics!")
            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
            st.markdown(f"""
            **Updated Stats:**
            - Matches: {new_matches}
            - Runs: {new_runs}
            - Wickets: {new_wickets}
            - Strike Rate: {new_sr}
            - Bowling Average: {new_bowling_avg}
            - Economy: {new_economy}
            - Centuries: {new_hundreds}
            - Half Centuries: {new_fifties}
            - Role: {new_role}
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            st.info("Data will be reflected in the app within a few seconds.")
        else:
            st.error(f"❌ Failed to update {selected_player}'s data. Please try again.")
    
    # ===== SECTION 6: BULK UPDATE =====
    st.markdown("---")
    st.markdown("#### 📤 Bulk Update (CSV)")
    
    with st.expander("Upload CSV to update multiple players at once"):
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="bulk_upload")
        
        if uploaded_file is not None:
            try:
                bulk_df = pd.read_csv(uploaded_file)
                st.markdown("**Preview:**")
                st.dataframe(bulk_df.head(), use_container_width=True)
                
                if st.button("🚀 Bulk Update Database", type="primary"):
                    success_count = 0
                    error_count = 0
                    
                    with st.spinner("Processing bulk update..."):
                        for idx, row in bulk_df.iterrows():
                            try:
                                player = row.get('player')
                                team = row.get('team')
                                format_type = row.get('format')
                                
                                # Extract all numeric columns
                                stats_dict = {col: row[col] for col in bulk_df.columns 
                                            if col not in ['player', 'team', 'format']}
                                
                                if update_player_stats(player, team, format_type, stats_dict):
                                    success_count += 1
                                else:
                                    error_count += 1
                            except Exception as e:
                                error_count += 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✅ {success_count} players updated successfully")
                    with col2:
                        if error_count > 0:
                            st.error(f"❌ {error_count} players failed to update")
                    
                    st.info("💡 CSV should have columns: player, team, format, and any stats columns to update (matches, runs, wickets, etc.)")
            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

