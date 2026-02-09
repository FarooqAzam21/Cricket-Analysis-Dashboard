import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_team_builder(all_players):
    """
    Renders the Smart playing 11 selection page with custom criteria and role-based pools.
    """
    st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #10B981; margin-bottom: 2rem;'>
            <h2 style='margin: 0; color: #10B981;'>⚡ Team Builder (Manual Selection)</h2>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>Set your criteria, then hand-pick your dream XI from the filtered pool.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Sidebar/Top Filters ---
    with st.expander("🛠️ Step 1: Set Your Performance Criteria", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            formats = sorted(all_players['Format'].unique())
            default_format = 'ODI' if 'ODI' in formats else formats[0]
            selected_format = st.selectbox("Format", formats, index=formats.index(default_format))
            
            teams = ['All'] + sorted(all_players['Team'].dropna().unique().tolist())
            selected_team = st.selectbox("Team Filter", teams)

        with col2:
            min_matches = st.number_input("Min Matches Played", 0, 500, 10)
            min_bat_avg = st.slider("Min Batting Avg", 0.0, 60.0, 25.0, 1.0)
            
            # Batting position 1-11 selector
            selected_positions = st.multiselect(
                "Filter Batting Positions", 
                list(range(1, 12)), 
                default=list(range(1, 12)),
                help="1=Opener, 2-5=Middle order, 6-7=Lower middle, 8-11=Lower order"
            )

        with col3:
            min_sr = st.slider("Min Strike Rate", 50.0, 200.0, 80.0, 5.0)
            max_bowl_avg = st.slider("Max Bowling Avg", 15.0, 100.0, 40.0, 1.0)
            min_wickets = st.number_input("Min Wickets", 0, 1000, 5)

    # --- Filter Data ---
    df = all_players[all_players['Format'] == selected_format].copy()
    if selected_team != 'All':
        df = df[df['Team'] == selected_team]
    
    # Global Match Filter
    df = df[df['matches'] >= min_matches]
    
    # --- Role-wise Pools ---
    st.subheader("🏏 Step 2: Pick Your Players")
    st.info("Choose your players from each category. Aim for exactly **11 players** total.")

    col_wk, col_bt = st.columns(2)
    col_ar, col_bw = st.columns(2)

    with col_wk:
        wk_pool = df[df['role_lower'].str.contains('wicket-keeper', na=False)]
        # Filter WK by criteria
        wk_pool = wk_pool[(wk_pool['average'] >= min_bat_avg) & (wk_pool['strike_rate'] >= min_sr)]
        # Also filter by position if possible
        if pd.notna(wk_pool['batting_position']).any():
            wk_pool = wk_pool[(wk_pool['batting_position'].astype(float, errors='ignore').isin(selected_positions)) | (wk_pool['batting_position'].isna())]
        selected_wk_names = st.multiselect(f"🧤 Wicket Keepers ({len(wk_pool)})", wk_pool['player'].tolist())
    
    with col_bt:
        bt_pool = df[df['role_lower'].str.contains('batsman', na=False)]
        # Filter BT by criteria
        bt_pool = bt_pool[(bt_pool['average'] >= min_bat_avg) & (bt_pool['strike_rate'] >= min_sr)]
        # Also filter by position if possible
        if pd.notna(bt_pool['batting_position']).any():
            bt_pool = bt_pool[(bt_pool['batting_position'].astype(float, errors='ignore').isin(selected_positions)) | (bt_pool['batting_position'].isna())]
        selected_bt_names = st.multiselect(f"🏏 Batsmen ({len(bt_pool)})", bt_pool['player'].tolist())

    with col_ar:
        ar_pool = df[df['role_lower'].str.contains('all-rounder', na=False)]
        # Filter ARs by batting criteria (lenient on bowling)
        ar_pool = ar_pool[(ar_pool['average'] >= min_bat_avg) & (ar_pool['strike_rate'] >= min_sr)]
        # Also filter by position if possible
        if pd.notna(ar_pool['batting_position']).any():
            ar_pool = ar_pool[(ar_pool['batting_position'].astype(float, errors='ignore').isin(selected_positions)) | (ar_pool['batting_position'].isna())]
        selected_ar_names = st.multiselect(f"⚡ All-Rounders ({len(ar_pool)})", ar_pool['player'].tolist())

    with col_bw:
        bw_pool = df[df['role_lower'].str.contains('bowler|fast|spinner', na=False)]
        # Filter Bowlers by Bowl criteria
        bw_pool = bw_pool[(bw_pool['wickets'] >= min_wickets) & 
                          (bw_pool['bowling_average'] <= max_bowl_avg) & (bw_pool['bowling_average'] > 0)]
        selected_bw_names = st.multiselect(f"⚾ Bowlers ({len(bw_pool)})", bw_pool['player'].tolist())

    # Combine all selections
    all_selected_names = selected_wk_names + selected_bt_names + selected_ar_names + selected_bw_names
    total_selected = len(all_selected_names)

    # --- Verification & Display ---
    if total_selected == 11:
        st.success("✅ Perfect! You have selected your Playing 11.")
    elif total_selected > 11:
        st.error(f"⚠️ You have selected {total_selected} players. Please remove {total_selected - 11} to make it 11.")
    else:
        st.warning(f"ℹ️ Selected {total_selected}/11 players. Keep going!")

    if total_selected > 0:
        st.markdown("---")
        st.subheader("🏟️ Your Custom Playing XI")
        
        # Get full data for selected players
        playing_xi = all_players[all_players['player'].isin(all_selected_names) & (all_players['Format'] == selected_format)].drop_duplicates(subset=['player'])
        
        # Initialize session state for position overrides
        if 'position_overrides' not in st.session_state:
            st.session_state.position_overrides = {}
        
        # Step 1: Display selected players in a table with position editing
        with st.expander("✏️ Step 3: Customize Player Positions", expanded=True):
            st.info("💡 Assign batting position (1-11) for each player to optimize your team composition.")
            
            # Create position selector
            position_assignments = {}
            position_cols = st.columns(2)
            col_idx = 0
            
            for idx, (p_idx, player) in enumerate(playing_xi.reset_index().iterrows()):
                current_pos = st.session_state.position_overrides.get(player['player'], 
                                                                       int(player.get('batting_position', 5)))
                
                with position_cols[col_idx % 2]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{player['player']}** ({player['Team']})")
                    with col2:
                        role_icon = "🧤" if "wicket-keeper" in str(player['role_lower']).lower() else \
                                    "🏏" if "batsman" in str(player['role_lower']).lower() else \
                                    "⚡" if "all-rounder" in str(player['role_lower']).lower() else "⚾"
                        st.write(role_icon)
                    
                    new_pos = st.number_input(
                        "Position",
                        min_value=1,
                        max_value=11,
                        value=int(current_pos) if isinstance(current_pos, int) else 5,
                        key=f"pos_{idx}_{player['player']}",
                        help="1=Opener, 2-5=Middle order, 6-7=Lower middle, 8-11=Lower order"
                    )
                    st.session_state.position_overrides[player['player']] = new_pos
                    position_assignments[player['player']] = new_pos
                    col_idx += 1
        
        # Step 2: Display cards with updated positions
        st.subheader("📋 Your Playing XI with Assigned Positions")
        
        # Sort by assigned position
        sorted_players = sorted(playing_xi.reset_index().iterrows(), 
                               key=lambda x: position_assignments.get(x[1]['player'], 5))
        
        cols = st.columns(3)
        for idx, (p_idx, player) in enumerate(sorted_players):
            with cols[idx % 3]:
                role_icon = "🧤" if "wicket-keeper" in str(player['role_lower']).lower() else \
                            "🏏" if "batsman" in str(player['role_lower']).lower() else \
                            "⚡" if "all-rounder" in str(player['role_lower']).lower() else "⚾"
                
                # Get assigned position
                assigned_pos = position_assignments.get(player['player'], int(player.get('batting_position', 5)))
                
                img_url = player.get('image_url', "https://via.placeholder.com/150?text=No+Image")
                
                st.markdown(f"""
                    <div class="elite-card">
                        <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 15px;">
                            <img src="{img_url}" style="width: 60px; height: 60px; border-radius: 30px; object-fit: cover; border: 3px solid var(--primary);">
                            <div style="flex-grow: 1;">
                                <div style="font-size: 1.1rem; font-weight: 800; color: var(--primary-dark) !important;">{player['player']}</div>
                                <div style="font-size: 0.8rem; opacity: 0.7; font-weight: 600;">{player['Team']}</div>
                            </div>
                            <div style="background: var(--primary); color: #e2e8f0; padding: 4px 10px; border-radius: 8px; font-weight: 800; font-size: 0.8rem;">
                                #{assigned_pos}
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <span style="font-size: 0.9rem; font-weight: 600;">{role_icon} {player['role']}</span>
                            <span style="font-size: 0.8rem; background: rgba(0,0,0,0.05); padding: 2px 8px; border-radius: 4px;">Form: 📈 Match Ready</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.05);">
                            <div><b>Matches:</b> {int(player['matches'])}</div>
                            <div><b>{"Avg (B)": if role_icon in ["⚾", "⚡"] else "Avg (P)"}:</b> {player['bowling_average'] if role_icon in ["⚾", "⚡"] else player['average']:.1f}</div>
                            <div><b>{"Econ": if role_icon in ["⚾", "⚡"] else "SR"}:</b> {player['economy'] if role_icon in ["⚾", "⚡"] else player['strike_rate']:.1f}</div>
                            <div><b>Wickets:</b> {int(player['wickets'])}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Team Stats Summary
        with st.expander("📊 Squad Multi-Format Summary"):
            st.table(playing_xi[['player', 'Team', 'role', 'matches', 'average', 'strike_rate', 'wickets', 'economy']])

        # Visual Composition
        if total_selected == 11:
            st.subheader("📈 Squad Balance")
            role_counts = []
            if selected_wk_names: role_counts.append({'Role': 'Wicket-Keeper', 'Count': len(selected_wk_names)})
            if selected_bt_names: role_counts.append({'Role': 'Batsman', 'Count': len(selected_bt_names)})
            if selected_ar_names: role_counts.append({'Role': 'All-Rounder', 'Count': len(selected_ar_names)})
            if selected_bw_names: role_counts.append({'Role': 'Bowler', 'Count': len(selected_bw_names)})
            
            comp_df = pd.DataFrame(role_counts)
            fig = px.pie(comp_df, values='Count', names='Role', 
                         hole=0.4, 
                         color_discrete_sequence=['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6'],
                         template='plotly_dark')
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width="stretch")
