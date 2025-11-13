# stats_visualization_fixed.py
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# Config / helpers
# ---------------------------
st.set_page_config(page_title="Cricket Analysis Dashboard" ,  page_icon="🏏", layout='wide' , initial_sidebar_state="expanded")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob, .stDeployButton, .st-emotion-cache-1avcm0n {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
le = LabelEncoder()
scaler = StandardScaler()

def safe_col(df, col, default=0):
    """Return df[col] if exists else a Series of default values."""
    if col in df.columns:
        return df[col]
    return pd.Series([default]*len(df), index=df.index)

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = 0

# ---------------------------
# File paths (update if needed)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(BASE_DIR, "odi_batsman.csv")
file_path2 = os.path.join(BASE_DIR, "odi_all_rounders.csv")
file_path3 = os.path.join(BASE_DIR, "odi_bowler.csv")
file_path4 = os.path.join(BASE_DIR, "yearwise_data.csv")

st.sidebar.title("Cricket Analysis Menu")
menu = st.sidebar.radio(
    "Navigate to",[
        "Format Wise Analysis",
        "Select Playing 11",
        "Player Comparison",
        "Player Analysis",
        "Predict Runs",
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Developed by **Farooq Azam**")
# ---------------------------
# Load CSVs (with robust handling)
# ---------------------------
try:
    df = pd.read_csv(file_path1)
except Exception as e:
    st.error(f"Failed to read {file_path1}: {e}")
    raise SystemExit

try:
    df2 = pd.read_csv(file_path2)
except Exception:
    df2 = pd.DataFrame()  # allow empty

try:
    bowlers_data = pd.read_csv(file_path3)
except Exception:
    bowlers_data = pd.DataFrame()

try:
    year_wise_data = pd.read_csv(file_path4)
except Exception:
    year_wise_data = pd.DataFrame()

# strip column names
df.columns = df.columns.str.strip()
df2.columns = df2.columns.str.strip()
bowlers_data.columns = bowlers_data.columns.map(str).str.strip()
year_wise_data.columns = year_wise_data.columns.str.strip()

# ---------------------------
# Standardize text columns used for logic
# ---------------------------
for d in [df, df2, bowlers_data, year_wise_data]:
    if 'role' in d.columns:
        d['role'] = d['role'].astype(str).str.strip()
    if 'batting_position' in d.columns:
        # keep as string for consistent comparisons later
        d['batting_position'] = d['batting_position'].astype(str).str.strip()

# ---------------------------
# Build master frames
# ---------------------------
# concat batsmen + allrounders (original logic kept)
batsman_and_all_rounder = pd.concat([df, df2], ignore_index=True, sort=False)

# create subsets
# Ensure role lowercased for checks, but keep original values for display if present
batsman_and_all_rounder['role'] = batsman_and_all_rounder.get('role', '').astype(str).str.strip()
batsmen = batsman_and_all_rounder[batsman_and_all_rounder['role'].str.lower() == "batsman"]

# wicket keepers - check across df (original code read from df); be robust to variants
# use any source to find wicket-keepers (df primary, else look into combined data)
wicket_keepers = df[df['role'].astype(str).str.strip().str.lower() == 'wicket-keeper']
if wicket_keepers.empty:
    wicket_keepers = batsman_and_all_rounder[batsman_and_all_rounder['role'].astype(str).str.strip().str.lower().str.contains('wicket-keeper', na=False)]

all_rounders = batsman_and_all_rounder[(batsman_and_all_rounder['role'].astype(str).str.strip().str.lower() != "batsman") & (batsman_and_all_rounder['role'].astype(str).str.strip().str.lower() != "wicket-keeper")]

# combine everything including bowlers (keeps original logic)
all_players = pd.concat([batsmen, all_rounders, wicket_keepers, bowlers_data], ignore_index=True, sort=False)
num_cols = ['matches', 'runs', 'average', 'strike_rate', 'wickets', 'bowling_average']
for c in num_cols:
    all_players[c] = pd.to_numeric(all_players[c], errors='coerce').fillna(0)
# ---------------------------
# Basic cleaning & numeric conversion
# ---------------------------
# Ensure numeric columns exist and are numerifc
num_cols = ['matches', 'runs', 'average', 'strike_rate', 'wickets', 'bowling_average', 'economy', 'Innings', '100s', '50s']
ensure_numeric(all_players, num_cols)
ensure_numeric(df, num_cols)
ensure_numeric(bowlers_data, num_cols)
ensure_numeric(year_wise_data, ['year', 'matches', 'runs', 'average', 'SR', '50s', '100s'])

# Fill NA where reasonable (but avoid masking real missing data)
all_players.fillna({'runs':0, 'average':0, 'strike_rate':0, 'matches':0, 'wickets':0, 'bowling_average':999, 'economy':999}, inplace=True)

# Normalize role and batting_position strings in all_players for logical checks
all_players['role'] = all_players.get('role', '').astype(str).str.strip().str.lower()
all_players['batting_position'] = all_players.get('batting_position', '').astype(str).str.strip()

# copy used for ML feature encoding (like previous df3)
df3 = all_players.copy()
# encode team/format in df3 for model inputs (keeps original all_players unchanged for UI filters)
if 'Team' in df3.columns:
    try:
        df3['Team_encoded'] = le.fit_transform(df3['Team'].astype(str))
    except Exception:
        # fallback: numeric mapping
        df3['Team_encoded'] = df3['Team'].astype('category').cat.codes
if 'Format' in df3.columns:
    try:
        df3['Format_encoded'] = le.fit_transform(df3['Format'].astype(str))
    except Exception:
        df3['Format_encoded'] = df3['Format'].astype('category').cat.codes

required_for_rf = ['matches', 'Innings', 'runs', 'strike_rate', '100s', '50s', 'average']
for c in required_for_rf:
    if c not in df.columns:
        df[c] = 0

X_rf = df[['matches', 'Innings', 'runs', 'strike_rate', '100s', '50s']].fillna(0)
y_rf = df['average'].fillna(0)

# Scale features
try:
    x_scaled = scaler.fit_transform(X_rf)
except Exception:
    x_scaled = X_rf.values  # fallback if scaler fails

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Cricket Stats Dashboard", layout="wide")
st.title("🏏 Cricket Analytics Dashboard")

# Sidebar filters
st.sidebar.header("Filters & Options")
# For the team dropdown show actual team names from all_players (not encoded)
teams = ['All']
if 'Team' in all_players.columns:
    teams += sorted(all_players['Team'].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Select Team", teams)

if selected_team != "All":
    data = all_players[all_players['Team'] == selected_team]
else:
    data = all_players

# ---------------------------
# Top Visualizations (3 columns)
# ---------------------------

col1, col2, col3 = st.columns(3)

# ---------------------------
# Format-wise Charts
# ---------------------------

formats = ['Odi', 'T20', 'Test']  # Adjust according to your dataset values


if menu == 'Format Wise Analysis':
    for fmt in formats:
            st.markdown(f"## 🏏 {fmt} Format Analysis")
            st.markdown("---")

            # Filter data for this format
            fmt_batsmen = batsmen[batsmen['Format'] == fmt]
            fmt_all_rounders = all_rounders[all_rounders['Format'] == fmt]
            fmt_bowlers = bowlers_data[bowlers_data['Format'] == fmt]
            fmt_wicket_keepers = wicket_keepers[wicket_keepers['Format'] == fmt]
            fmt_all_players = pd.concat([fmt_batsmen , fmt_all_rounders , fmt_wicket_keepers , fmt_bowlers])
            # ---------------------------
            # Top Batsmen Charts
            # ---------------------------
            col1, col2, col3 = st.columns(3)

            filtered_batsmen = fmt_all_players[(fmt_all_players['matches'] > 10) & (fmt_all_players['role'].isin(['Batsman' , 'wicket-keeper']))] if not fmt_all_players.empty else pd.DataFrame()
            filtered_batsmen = filtered_batsmen.sort_values(by='runs', ascending=False)
            with col1:
                if not filtered_batsmen.empty:
                    fig1 = px.bar(
                        filtered_batsmen.sort_values(by='runs', ascending=False).head(10),
                        x='player',
                        y='runs', color='Team',
                        title=f"🏆 Top 10 Run Scorers - {fmt}"
                        )
                    st.plotly_chart(fig1, use_container_width=True, key=f'top_runs_{fmt}')
                else:
                    st.info(f"No batsman data for {fmt}.")
            with col2:
                if not filtered_batsmen.empty:
                    fig2 = px.scatter(
                        filtered_batsmen, x='average', y='strike_rate', color='Team',
                            size='matches', hover_name='player',
                            title=f"📈 Avg vs SR  - {fmt}"
                        )
                    st.plotly_chart(fig2, use_container_width=True, key=f'avg_sr_{fmt}')
                else:
                    st.info(f"No Average vs Strike Rate data for {fmt}.")

            with col3:
                if not fmt_wicket_keepers.empty:
                    fig3 = px.scatter(
                        fmt_wicket_keepers, x='average', y='strike_rate', color='Team',
                        size='matches', hover_name='player',
                        title=f"📊 Wicket Keepers: Avg vs SR - {fmt}"
                        )
                    st.plotly_chart(fig3, use_container_width=True, key=f'wk_scatter_{fmt}')
                else:
                    st.info(f"No wicket-keeper data for {fmt}.")

            # ---------------------------
            # More Charts
            # ---------------------------
            st.markdown("---")
            if not filtered_batsmen.empty:
                fig4 = px.bar(
                    filtered_batsmen.sort_values(by='average', ascending=False).head(10),
                    x='player',
                    y='average', color='Team',
                    title=f"Top 10 Batters by Average - {fmt}"
                    )
                st.plotly_chart(fig4, use_container_width=True, key=f'bat_avg_{fmt}')

                fig5 = px.bar(
                    filtered_batsmen.sort_values(by='strike_rate', ascending=False).head(10),
                    x='player',
                    y='strike_rate', color='Team',
                    title=f"Top 10 Batters by Strike Rate - {fmt}"
                    )
                st.plotly_chart(fig5, use_container_width=True, key=f'bat_sr_{fmt}')
            else:
                st.info(f"No batting data for {fmt} format.")

            # ---------------------------
            # All-rounders
            # ---------------------------
            col4, col5 = st.columns(2)
            with col4:
                if not fmt_all_rounders.empty:
                    fig6 = px.bar(
                        fmt_all_rounders.sort_values(by='wickets', ascending=False).head(10),
                        x='player', 
                        y='wickets', color='Team',
                        title=f"Top 10 All-Rounders by Wickets - {fmt}"
                        )
                    st.plotly_chart(fig6, use_container_width=True, key=f'wickets_{fmt}')
                else:
                    st.info(f"No all-rounder data for {fmt} format.")

            with col5:
                if not fmt_all_rounders.empty:
                    fig7 = px.bar(
                        fmt_all_rounders.sort_values(by='bowling_average', ascending=True).head(10),
                        x='player', 
                        y='bowling_average', color='Team',
                        title=f"Top 10 All-Rounders by Bowling Avg (Lower Better) - {fmt}"
                        )
                    st.plotly_chart(fig7, use_container_width=True, key=f'bowling_avg_{fmt}')
                else:
                    st.info(f"No bowling average data for {fmt} format.")

            # ---------------------------
            # Bowlers Section
            # ---------------------------
            st.markdown("---")
            st.subheader(f"⚡ Top 10 Bowlers - {fmt}")
            colB1, colB2 = st.columns(2)

            if not fmt_bowlers.empty:
                with colB1:
                    fig8 = px.bar(
                        fmt_bowlers.sort_values(by='wickets', ascending=False).head(10),
                        x='player', 
                        y='wickets', color='Team',
                        title=f"Top 10 Bowlers by Wickets - {fmt}"
                        )
                    st.plotly_chart(fig8, use_container_width=True, key=f'bowl_wickets_{fmt}')

                with colB2:
                    fig9 = px.bar(
                        fmt_bowlers.sort_values(by='bowling_average', ascending=True).head(10),
                        x='player', 
                        y='bowling_average', color='Team',
                        title=f"Top 10 Bowlers by Bowling Avg (Lower Better) - {fmt}"
                        )
                    st.plotly_chart(fig9, use_container_width=True, key=f'bowl_avg_{fmt}')
            else:
                st.info(f"No bowling data available for {fmt} format.")
            
elif menu == 'Player Comparison':
                st.markdown("---")
                st.subheader("⚔ Player Comparison")

                try:
                    # Select two players
                    col1, col2 = st.columns(2)
                    with col1:
                        player1 = st.selectbox("Select Player 1", all_players['player'].unique(), key="p1")
                    with col2:
                        player2 = st.selectbox("Select Player 2", all_players['player'].unique(), key="p2")

                    # Fetch their data
                    p1_data = all_players[all_players['player'] == player1]
                    p2_data = all_players[all_players['player'] == player2]

                    # Check if 'Format' column exists
                    if 'Format' in df.columns:
                        formats = all_players['Format'].unique()
                        selected_format = st.selectbox("Select Format", formats, key="fmt_cmp")

                        p1_format = p1_data[p1_data['Format'] == selected_format].iloc[0]
                        p2_format = p2_data[p2_data['Format'] == selected_format].iloc[0]
                    else:
                        st.warning("⚠ 'Format' column not found in your CSV. Please add it (ODI/T20/Test).")
                        st.stop()

                    # Show comparison side by side
                    st.write(f"### {selected_format} Format Comparison")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f"https://source.unsplash.com/400x400/?{player1},cricketer", caption=player1)
                        st.metric("Matches", p1_format['matches'])
                        st.metric("Innings", p1_format['Innings'])
                        st.metric("Runs", p1_format['runs'])
                        st.metric("Average", p1_format['average'])
                        st.metric("Strike Rate", p1_format['strike_rate'])
                        st.metric("100s", p1_format['100s'])
                        st.metric("50s", p1_format['50s'])

                    with col2:
                        st.image(f"https://source.unsplash.com/400x400/?{player2},cricketer", caption=player2)
                        st.metric("Matches", p2_format['matches'])
                        st.metric("Innings", p2_format['Innings'])
                        st.metric("Runs", p2_format['runs'])
                        st.metric("Average", p2_format['average'])
                        st.metric("Strike Rate", p2_format['strike_rate'])
                        st.metric("100s", p2_format['100s'])
                        st.metric("50s", p2_format['50s'])

                    # Optional: Add radar comparison chart
                    import plotly.graph_objects as go
                    categories = ['matches', 'Innings', 'runs', 'average', 'strike_rate', '100s', '50s']

                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=[p1_format[c] for c in categories],
                        theta=categories,
                        fill='toself',
                        name=player1
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=[p2_format[c] for c in categories],
                        theta=categories,
                        fill='toself',
                        name=player2
                    ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title=f"{selected_format} Comparison: {player1} vs {player2}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Comparison feature skipped due to error: {e}")
# ---------------------------
# Top 5 (or 7) Batters Recommendation logic (preserve original intention)
# ---------------------------
elif menu == 'Select Playing 11':

        st.markdown("---")
        st.header("⚡ Auto Recommendation: Top Batters (Position-aware)")
        format = all_players['Format'].unique()
        selected_format = st.selectbox("Select the format" , format , key='format_box_main')
        min_matches = 10
        min_avg = 45
        min_sr = 90

        # build positions list from all_players
        positions_raw = all_players['batting_position'].dropna().unique().tolist() if 'batting_position' in all_players.columns else []
        # try to normalize numeric-looking positions to ints then to strings
        positions = ['All']
        try:
            int_positions = sorted({int(float(p)) for p in positions_raw if str(p).strip() != ''})
            positions += [str(p) for p in int_positions]
        except Exception:
            # fallback keep as strings
            positions += sorted([str(p) for p in positions_raw if str(p).strip() != ''])

        selected_pos = st.selectbox("Select Batting Position", ['1', '2' , '3' , '4'  , '5' , '6' , '7' , '8' , '9' , '10' , '11' ], key='batting_order_select')

        # Prepare normalized fields in all_players
        all_players['matches'] = pd.to_numeric(all_players.get('matches', 0), errors='coerce').fillna(0)
        all_players['average'] = pd.to_numeric(all_players.get('average', 0), errors='coerce').fillna(0)
        all_players['strike_rate'] = pd.to_numeric(all_players.get('strike_rate', 0), errors='coerce').fillna(0)
        all_players['bowling_average'] = pd.to_numeric(all_players.get('bowling_average', 999), errors='coerce').fillna(999)
        all_players['economy'] = pd.to_numeric(all_players.get('economy', 999), errors='coerce').fillna(999)

        # Base filters
        base_filter_odi = (
            (all_players['matches'] >= min_matches) &
            (all_players['average'] >= min_avg) &
            (all_players['strike_rate'] >= min_sr)
        )
        base_filter_t20 = (
            (all_players['matches'] >= 10) &
            (all_players['average'] >= 30) &
            (all_players['strike_rate'] >= 130)
        )
        base_filter_test = (
            (all_players['matches'] >= 10) &
            (all_players['average'] >= 40)
        )

        base_filter_bowler_odi = (
            (all_players['role'] == 'fast-bowler') &
            (all_players['matches'] >= 10) &
            (all_players['bowling_average'] <= 35) &
            (all_players['economy'] < 6)
        )
        base_filter_bowler_t20 = (
            (all_players['role'] == 'fast-bowler') &
            (all_players['matches'] >= 10) &
            (all_players['bowling_average'] <= 35) &
            (all_players['economy'] < 9)
        )
        base_filter_bowler_test = (
            (all_players['role'] == 'fast-bowler') &
            (all_players['matches'] >= 10) &
            (all_players['bowling_average'] <= 35)
        )
        # Apply position-specific logic (preserve your original varied thresholds)

        if selected_format =='T20':
            if selected_pos == "All":
                filtered = all_players[base_filter_t20]

            elif selected_pos == '1':
                filtered = all_players[base_filter_t20 & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '2':
                filtered = all_players[base_filter_t20 & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '3':
                filtered = all_players[base_filter_t20 & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '4':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (30)) &
                    (all_players['strike_rate'] >= (130)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '5':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (30)) &
                    (all_players['strike_rate'] > (130)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['role'].str.contains('wicket', na=False)) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '6':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (25)) &
                    (all_players['strike_rate'] > (150)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '7':
                # combine conditions carefully with parentheses to maintain intended precedence
                filtered = all_players[
                    (
                        (all_players['matches'] >= min_matches) &
                        (all_players['average'] >= (100)) &
                        (all_players['strike_rate'] >= (120)) &
                        (all_players['batting_position'] == selected_pos) &
                        (all_players['Format'] == selected_format)
                    ) &
                    (
                        (all_players['bowling_average'] < 35.0) |
                        ((all_players['role'] != 'batsman')) &
                        (all_players['role'] != 'wicket-keeper')
                    )
                ]
            elif selected_pos == '8':
                # leg-spinner specific (fix missing comma and ensure boolean expression valid)
                filtered = all_players[
                    (all_players['role'] == 'leg-spinner') &
                    (all_players['matches'] >= min_matches) &
                    (all_players['bowling_average'] < 30) &
                    (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)
                ]

            elif selected_pos in ['9', '10', '11']:
                # bowlers for positions 9-11 (use base_filter_bowler and batting_position match)
                filtered = all_players[base_filter_bowler_t20 & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            else:
                filtered = all_players[base_filter_t20]
        elif selected_format =='Odi':
            if selected_pos == "All":
                filtered = all_players[base_filter_odi]

            elif selected_pos == '1':
                filtered = all_players[base_filter_odi & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '2':
                filtered = all_players[base_filter_odi & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '3':
                filtered = all_players[base_filter_odi & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '4':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (min_avg - 10)) &
                    (all_players['strike_rate'] >= (min_sr - 5)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '5':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (min_avg - 10)) &
                    (all_players['strike_rate'] > (min_sr + 10)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['role'].str.contains('wicket', na=False)) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '6':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (min_avg - 10)) &
                    (all_players['strike_rate'] > (min_sr + 10)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '7':
                # combine conditions carefully with parentheses to maintain intended precedence
                filtered = all_players[
                    (
                        (all_players['matches'] >= min_matches) &
                        (all_players['average'] >= (min_avg - 20)) &
                        (all_players['strike_rate'] >= (min_sr + 5)) &
                        (all_players['batting_position'] == selected_pos) &
                        (all_players['Format'] == selected_format)
                    ) &
                    (
                        (all_players['bowling_average'] < 35.0) |
                        ((all_players['role'] != 'batsman')) &
                        (all_players['role'] != 'wicket-keeper')
                    )
                ]

            elif selected_pos == '8':
                # leg-spinner specific (fix missing comma and ensure boolean expression valid)
                filtered = all_players[
                    (all_players['role'] == 'leg-spinner') &
                    (all_players['matches'] >= min_matches) &
                    (all_players['bowling_average'] < 35) &
                    (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)
                ]

            elif selected_pos in ['9', '10', '11']:
                # bowlers for positions 9-11 (use base_filter_bowler and batting_position match)
                filtered = all_players[base_filter_bowler_odi & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            else:
                filtered = all_players[base_filter_odi]
        elif selected_format == 'Test':
            if selected_pos == "All":
                filtered = all_players[base_filter_test]

            elif selected_pos == '1':
                filtered = all_players[base_filter_test & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '2':
                filtered = all_players[base_filter_test & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '3':
                filtered = all_players[base_filter_test & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            elif selected_pos == '4':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (min_avg - 10)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '5':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (min_avg - 10)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['role'].str.contains('wicket', na=False)) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '6':
                filtered = all_players[
                    (all_players['matches'] >= min_matches) &
                    (all_players['average'] >= (min_avg - 10)) &
                    (all_players['batting_position'] == selected_pos) &
                    (all_players['Format'] == selected_format)
                ]

            elif selected_pos == '7':
                # combine conditions carefully with parentheses to maintain intended precedence
                filtered = all_players[
                    (
                        (all_players['matches'] >= min_matches) &
                        (all_players['average'] >= (min_avg - 20)) &
                        (all_players['batting_position'] == selected_pos) &
                        (all_players['Format'] == selected_format)
                    ) &
                    (
                        (all_players['bowling_average'] < 35.0) |
                        ((all_players['role'] != 'batsman')) &
                        (all_players['role'] != 'wicket-keeper')
                    )
                ]

            elif selected_pos == '8':
                # leg-spinner specific (fix missing comma and ensure boolean expression valid)
                filtered = all_players[
                    (all_players['role'].contain('spinner')) &
                    (all_players['matches'] >= min_matches) &
                    (all_players['bowling_average'] < 35) &
                    (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)
                ]

            elif selected_pos in ['9', '10', '11']:
                # bowlers for positions 9-11 (use base_filter_bowler and batting_position match)
                filtered = all_players[base_filter_bowler_test & (all_players['batting_position'] == selected_pos) & (all_players['Format'] == selected_format)]

            else:
                filtered = all_players[base_filter_test]        
        # Sorting and show top7 (preserve original ordering)
        top7 = filtered.sort_values(by=['average', 'strike_rate', 'runs', 'bowling_average'], ascending=False).head(7)
        display_cols = ['player', 'Team' , 'runs' , 'matches', 'average', 'strike_rate', 'batting_position', 'bowling_average', 'wickets', 'economy']
        display_cols = [c for c in display_cols if c in top7.columns]
        st.dataframe(top7[display_cols])

        # ---------------------------
        # Team Builder (unique position constraint)
        # ---------------------------
        st.markdown("---")
        st.header("🧩 Team Builder (1 Player per Position)")

        if "selected_team" not in st.session_state:
            st.session_state.selected_team = []
        if "used_positions" not in st.session_state:
            st.session_state.used_positions = set()

        for i, row in top7.iterrows():
            pos = str(row.get('batting_position', 'NA'))
            btn_label = f"Add {row.get('player', 'Unknown')} ({pos})"
            if st.button(btn_label, key=f"add_{i}_{pos}"):
                if pos in st.session_state.used_positions:
                    st.warning(f"⚠️ Position '{pos}' already taken!")
                else:
                    st.session_state.selected_team.append(row.to_dict())
                    st.session_state.used_positions.add(pos)
                    st.success(f"✅ {row.get('player', 'Unknown')} added to your team!")

        if st.session_state.selected_team:
            team_df = pd.DataFrame(st.session_state.selected_team)
            cols_to_show = [c for c in ['player', 'Team', 'average', 'strike_rate', 'batting_position'] if c in team_df.columns]
            st.dataframe(team_df[cols_to_show])

        # ---------------------------
        # Summary Comparison Chart (top7)
        # ---------------------------
        if not top7.empty:
            fig_summary = px.bar(
                top7,
                x='player',
                y=[c for c in ['average', 'strike_rate'] if c in top7.columns],
                barmode='group',
                title="🏅 Top Players: Average vs Strike Rate"
            )
            st.plotly_chart(fig_summary, use_container_width=True, key='summary_chart')

# ---------------------------
# Player Search & Analysis
# ---------------------------
elif menu == 'Player Analysis':

        st.markdown("---")
        st.header("🔍 Player Search & Analysis")
        player_list = sorted(all_players['player'].dropna().unique().tolist()) if 'player' in all_players.columns else []
        selected_player = st.selectbox("Search Player", player_list, key='player_search_box')

        if selected_player:
            player_data = all_players[all_players['player'] == selected_player]
            if not player_data.empty:
                player_row = player_data.iloc[0]
                col_img, col_info = st.columns([1, 2])

                # image
                if 'image_url' in player_row and pd.notna(player_row['image_url']):
                    col_img.image(player_row['image_url'], width=180)
                else:
                    col_img.image("https://via.placeholder.com/150?text=No+Image", width=180)

                # summary
                col_info.markdown(f"### {player_row.get('player','Unknown')}")
                col_info.markdown(f"**Team:** {player_row.get('Team','-')}")
                if 'role' in player_row:
                    col_info.markdown(f"**Role:** {player_row.get('role','-')}")
                if 'batting_position' in player_row:
                    col_info.markdown(f"**Batting Position:** {player_row.get('batting_position','-')}")

                if 'Format' in player_data.columns:
                    formats = player_data['Format'].unique()

                    for fmt in formats:
                        fmt_data = player_data[player_data['Format'] == fmt]

                        st.markdown(f"#### 🏏 {fmt} Format")

                        cols = st.columns(6)
                        cols[0].metric("Matches", int(fmt_data['matches'].sum() if 'matches' in fmt_data.columns else 0))
                        
                        if 'runs' in fmt_data.columns:
                            cols[1].metric("Runs", int(fmt_data['runs'].sum()))

                        if 'average' in fmt_data.columns:
                            cols[2].metric("Average", round(fmt_data['average'].mean(), 2))

                        if 'strike_rate' in fmt_data.columns:
                            cols[3].metric("Strike Rate", round(fmt_data['strike_rate'].mean(), 2))

                        if 'wickets' in fmt_data.columns:
                            cols[4].metric("Wickets", int(fmt_data['wickets'].sum()))

                        if 'bowling_average' in fmt_data.columns:
                            cols[5].metric("Bowling Average", round(fmt_data['bowling_average'].mean(), 2))

                        st.markdown("---")

                    # --- Combined Chart for all formats ---
                    st.subheader("📈 Runs Comparison Across Formats")
                    format_stats = (
                        player_data.groupby('Format')[['runs', 'matches', 'average', 'strike_rate']]
                        .mean()
                        .reset_index()
                    )

                    fig_format = px.bar(
                        format_stats,
                        x='Format',
                        y='runs',
                        color='Format',
                        title=f"Runs by Format for {selected_player}"
                    )
                    st.plotly_chart(fig_format, use_container_width=True, key=f'format_{selected_player}')

# ---------------------------
# Predicted Batting Average (RF) - preserve original approach
# ---------------------------
elif menu == 'Predict Runs':
        st.markdown("---")
        st.subheader("🎯 Predict Batsman Runs (Format-wise)")

        try:
            # Load dataset
            df = pd.read_csv("odi_batsman.csv")

            # Check required columns
            required_cols = ['player', 'matches', 'Innings', 'average', 'strike_rate', '100s', '50s', 'Format', 'runs']
            for col in required_cols:
                if col not in df.columns:
                    st.error(f"Missing column: {col}")
                    st.stop()

            # Select format (like ODI, T20, Test)
            selected_format = st.selectbox("Select Format", sorted(df['Format'].unique()), key='format_select')

            # Filter data based on format
            df_format = df[df['Format'] == selected_format]

            # Avoid empty data issues
            if df_format.empty:
                st.warning(f"No data found for format: {selected_format}")
                st.stop()

            # Define features and target
            features = ['matches', 'Innings', 'average', 'strike_rate', '100s', '50s']
            target = 'runs'

            x = df_format[features]
            y = df_format[target] / df_format['matches']  # runs per match (realistic next match target)

            # Scale features
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            # Train model
            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_model.fit(x_scaled, y)

            # Player selection for prediction
            selected_player = st.selectbox(
                f"Select Player ({selected_format})",
                df_format['player'].unique(),
                key='avg_pred_player'
            )

            # Fetch player data
            single_row = df_format[df_format['player'] == selected_player].iloc[0]
            input_data = single_row[features].values.reshape(1, -1)

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            predicted_run = rf_model.predict(input_scaled)[0]

            # Display result
            st.metric(
                label=f"Predicted Next Match Runs for {selected_player} ({selected_format})",
                value=f"{round(predicted_run, 0)} runs"
            )

        except Exception as e:
            st.warning(f"⚠ Prediction skipped due to error: {e}")

        # ---------------------------
        # Year-wise prediction (next-year) & plot
        # ---------------------------
        st.markdown("---")
        st.header("Predict Player Next Year Performance (Year-wise)")

        if not year_wise_data.empty and 'player' in year_wise_data.columns:
            players_year = sorted(year_wise_data['player'].unique().tolist())
            sel_player_year = st.selectbox("Select player for year-wise prediction", players_year, key='yearwise_player')

            if sel_player_year:
                player_year_df = year_wise_data[year_wise_data['player'] == sel_player_year].copy()
                # ensure required columns exist
                required_cols = ['year', 'matches', 'runs', 'average', 'SR', '50s', '100s']
                for c in required_cols:
                    if c not in player_year_df.columns:
                        player_year_df[c] = 0
                # sort by year and drop NA runs
                player_year_df['year'] = pd.to_numeric(player_year_df['year'], errors='coerce')
                player_year_df = player_year_df.sort_values('year').dropna(subset=['year'])

                if len(player_year_df) >= 3:
                    # train on historical year rows
                    Xy = player_year_df[['matches', 'average', 'SR', '50s', '100s']].fillna(0)
                    yy = player_year_df['runs'].fillna(0)
                    x_train, x_test, y_train, y_test = train_test_split(Xy, yy, test_size=0.2, random_state=42)
                    model_year = RandomForestRegressor(n_estimators=200, random_state=42)
                    model_year.fit(x_train, y_train)

                    # predict next year using latest row stats
                    latest = player_year_df.iloc[-1][['matches', 'average', 'SR', '50s', '100s']].values.reshape(1, -1)
                    predicted_next = model_year.predict(latest)[0]
                    next_year = int(player_year_df['year'].max()) + 1

                    # plot actual + predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=player_year_df['year'],
                        y=player_year_df['runs'],
                        mode='lines+markers',
                        name='Actual Runs',
                        line=dict(color='blue', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[next_year],
                        y=[predicted_next],
                        mode='markers+text',
                        name='Predicted Runs',
                        marker=dict(color='red', size=12, symbol='star'),
                        text=[f"{int(predicted_next)}"],
                        textposition="top center"
                    ))
                    fig.update_layout(
                        title=f"📊 {sel_player_year} — Yearly Runs Trend",
                        xaxis_title="Year",
                        yaxis_title="Total Runs",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"🏏 {sel_player_year} is predicted to score approximately {predicted_next:.0f} runs in {next_year}.")
                    est_matches = int(player_year_df.iloc[-1]['matches']) if 'matches' in player_year_df.columns else 1
                    st.info(f"📈 Estimated average per match: {predicted_next/est_matches:.1f} runs/match")
                else:
                    st.warning("Not enough year-wise data (need at least 3 years) to train a model for this player.")
        else:
            st.info("No year-wise data file found or yearwise CSV missing 'player' column.")    


st.markdown("---")
st.caption("Dashboard logic preserved from original; code cleaned for stability, fixed boolean precedence, and safe handling of missing columns.")


