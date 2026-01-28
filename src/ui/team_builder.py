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
            
            # Position filter for Batters/ARs/WKs
            positions = sorted([str(p) for p in all_players['batting_position'].dropna().unique()])
            selected_positions = st.multiselect("Filter Batting Positions", positions, default=positions)

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
        # Filter WK by position and bat criteria
        wk_pool = wk_pool[(wk_pool['batting_position'].astype(str).isin(selected_positions)) & 
                          (wk_pool['average'] >= min_bat_avg) & (wk_pool['strike_rate'] >= min_sr)]
        selected_wk_names = st.multiselect(f"🧤 Wicket Keepers ({len(wk_pool)})", wk_pool['player'].tolist())
    
    with col_bt:
        bt_pool = df[df['role_lower'].str.contains('batsman', na=False)]
        # Filter BT by position and bat criteria
        bt_pool = bt_pool[(bt_pool['batting_position'].astype(str).isin(selected_positions)) & 
                          (bt_pool['average'] >= min_bat_avg) & (bt_pool['strike_rate'] >= min_sr)]
        selected_bt_names = st.multiselect(f"🏏 Batsmen ({len(bt_pool)})", bt_pool['player'].tolist())

    with col_ar:
        ar_pool = df[df['role_lower'].str.contains('all-rounder', na=False)]
        # Filter ARs by batting criteria (less strict on bowling to show more options)
        ar_pool = ar_pool[(ar_pool['batting_position'].astype(str).isin(selected_positions)) & 
                          (ar_pool['average'] >= min_bat_avg) & 
                          (ar_pool['strike_rate'] >= min_sr)]
        # Optional bowling filters (if they have data)
        ar_pool_with_bowling = ar_pool[(ar_pool['bowling_average'].fillna(999) <= max_bowl_avg) & 
                                       (ar_pool['wickets'].fillna(0) >= min_wickets)]
        # Use filtered pool if available, otherwise use batting-only filtered
        ar_pool = ar_pool_with_bowling if len(ar_pool_with_bowling) > 0 else ar_pool
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
            st.info("💡 You can change the batting position for each player. This helps optimize your team composition.")
            
            # Create editable dataframe
            edit_data = []
            all_positions = ['Opening', 'Top Order', 'Middle Order', 'Lower Middle', 'Tail']
            
            for idx, (p_idx, player) in enumerate(playing_xi.reset_index().iterrows()):
                current_pos = st.session_state.position_overrides.get(player['player'], 
                                                                       str(player.get('batting_position', 'Middle Order')))
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{player['player']}** ({player['Team']})")
                with col2:
                    new_pos = st.selectbox(
                        "Batting Position",
                        all_positions,
                        index=all_positions.index(current_pos) if current_pos in all_positions else 2,
                        key=f"pos_{idx}_{player['player']}",
                        label_visibility="collapsed"
                    )
                    st.session_state.position_overrides[player['player']] = new_pos
                with col3:
                    role_icon = "🧤" if "wicket-keeper" in str(player['role_lower']) else \
                                "🏏" if "batsman" in str(player['role_lower']) else \
                                "⚡" if "all-rounder" in str(player['role_lower']) else "⚾"
                    st.write(role_icon)
        
        # Step 2: Display cards with updated positions
        st.subheader("📋 Your Playing XI Details")
        cols = st.columns(3)
        for idx, (p_idx, player) in enumerate(playing_xi.reset_index().iterrows()):
            with cols[idx % 3]:
                role_icon = "🧤" if "wicket-keeper" in str(player['role_lower']) else \
                            "🏏" if "batsman" in str(player['role_lower']) else \
                            "⚡" if "all-rounder" in str(player['role_lower']) else "⚾"
                
                # Get custom position or default
                custom_pos = st.session_state.position_overrides.get(player['player'], 
                                                                     str(player.get('batting_position', 'Middle Order')))
                
                img_url = player.get('image_url', "https://via.placeholder.com/150?text=No+Image")
                
                st.markdown(f"""
                    <div style='background: rgba(255, 255, 255, 0.05); 
                                border: 1px solid rgba(255, 255, 255, 0.1); 
                                padding: 1rem; 
                                border-radius: 12px; 
                                margin-bottom: 20px;
                                backdrop-filter: blur(5px);
                                transition: transform 0.3s ease;'>
                        <div style='display: flex; gap: 10px; align-items: center; margin-bottom: 10px;'>
                            <img src='{img_url}' style='width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2px solid #10B981;'>
                            <div>
                                <div style='font-size: 1.1rem; font-weight: bold; color: #10B981;'>{player['player']}</div>
                                <div style='font-size: 0.75rem; opacity: 0.7;'>{player['Team']}</div>
                            </div>
                        </div>
                        <div style='display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin-bottom: 8px;'>
                            <span>{role_icon} {player['role']}</span>
                            <span style='background: #10B981; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;'>{custom_pos}</span>
                        </div>
                        <hr style='margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.1);'>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.75rem;'>
                            <div><b>Mat:</b> {int(player['matches'])}</div>
                            <div><b>{"Bowl Avg" if role_icon in ["⚾", "⚡"] else "Bat Avg"}:</b> {player['bowling_average'] if role_icon in ["⚾", "⚡"] else player['average']:.1f}</div>
                            <div><b>{"Econ" if role_icon in ["⚾", "⚡"] else "SR"}:</b> {player['economy'] if role_icon in ["⚾", "⚡"] else player['strike_rate']:.1f}</div>
                            <div><b>Wkts:</b> {int(player['wickets'])}</div>
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
            st.plotly_chart(fig, use_container_width=True)
