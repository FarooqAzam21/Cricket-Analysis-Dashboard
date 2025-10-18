import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
import plotly.graph_objects as go
le = LabelEncoder()
scaler = StandardScaler()
# =====================================
# CSV file ka path (update apna path)
# =====================================
csv_path = r"c:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_batsman.csv"
csv_path_2 = r"c:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\odi_all_rounders.csv"
csv_pat_3 = r"C:\Users\Farooq\Desktop\New folder (4)\Cricket_Analysis\yearwise_data.csv"
# =====================================
# Load CSV
# =====================================
df = pd.read_csv(csv_path)
df2 = pd.read_csv(csv_path_2)
year_wise_data = pd.read_csv(csv_pat_3)
df.columns = df.columns.str.strip()
df2.columns = df2.columns.str.strip()
year_wise_data.columns = year_wise_data.columns.str.strip()
batsman_and_all_rounder = pd.concat([df , df2] , ignore_index=True , sort=False)
batsmen = batsman_and_all_rounder[batsman_and_all_rounder['role'] == "Batsman"]
all_rounders = batsman_and_all_rounder[batsman_and_all_rounder['role'] != "Batsman"]
wicket_keepers = df[df['role'] == 'wicket-keeper']
all_players = pd.concat([batsmen , all_rounders , wicket_keepers])  
# Basic Cleaning
batsman_and_all_rounder = batsman_and_all_rounder.dropna(subset=['player', 'Team', 'matches', 'average', 'strike_rate' , 'wickets' , 'bowling_average'])
num_cols = ['matches', 'runs', 'average', 'strike_rate', 'wickets']
for col in num_cols:
    if col in batsman_and_all_rounder.columns:
        batsman_and_all_rounder[col] = pd.to_numeric(batsman_and_all_rounder[col], errors='coerce')
df3 = all_players.copy()      
df3['role'] = df3['role'].str.strip().str.lower()
df3['Team'] = le.fit_transform(df3['Team'])
df3['Format'] = le.fit_transform(df3['Format'])
df3.fillna(0 , inplace=True)
x = df[['matches', 'Innings' , 'runs', 'strike_rate', '100s', '50s']]
y = df['average']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# =====================================
# Streamlit Setup
# =====================================
st.set_page_config(page_title="Cricket Stats Dashboard", layout="wide")
st.title("🏏 Cricket Analytics Dashboard")

# Sidebar
st.sidebar.header("Filters & Options")
teams = ['All'] + sorted(batsman_and_all_rounder['Team'].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Select Team", teams)

if selected_team != "All":
    data = all_players[all_players['Team'] == selected_team]
else:
    data = all_players

# =====================================
# 🧭 Top Visualizations
# =====================================
col1, col2  = st.columns(2)
filtered_batsmen = batsmen[batsmen['runs'] > 1000] 
with col1:
    fig1 = px.bar(
        filtered_batsmen.sort_values(by='runs', ascending=False).head(10),
        x='player', y='runs', color='Team',
        title="🏆 Top 10 Run Scorers"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(
        filtered_batsmen, x='average', y='strike_rate', color='Team',
        size='matches', hover_name='player',
        title="📈 Average vs Strike Rate Comparison"
    )
    st.plotly_chart(fig2, use_container_width=True)
filtered_batsmen = batsmen[batsmen['runs'] > 1000]        
fig4 = px.bar(
    filtered_batsmen.sort_values(by='average' , ascending=False).head(10),
    x='player' , y = 'average', color='Team',
    title="Top 10 batters by Average"
    )
st.plotly_chart(fig4 , use_container_width=True , key='batting_average_chart')
fig5 = px.bar(
    filtered_batsmen.sort_values(by='strike_rate' , ascending=False).head(10),
    x='player' , y = 'strike_rate', color='Team',
    title="Top 10 batters by Strike Rate"
    )
st.plotly_chart(fig5 , use_container_width=True , key='strike_rate_chart')
col3 , col4 = st.columns(2)
with col3:
    fig6 = px.bar(
        all_rounders.sort_values(by='wickets' , ascending=False).head(10),
        x='player' , y='wickets' , color='Team',
        title="Top 10 all rounders by wickets "
    )
st.plotly_chart(fig6 , use_container_width=True , key='wicket_chart')  
with col4:
    fig7 = px.bar(
        all_rounders.sort_values(by='bowling_average' , ascending=True).head(10),
        x='player' , y='wickets' , color='Team',
        title="Top 10 all rounders by wickets "
    )
st.plotly_chart(fig7 , use_container_width=True , key='bowling_average_chart')

# =====================================
# ⚡ Top 5 Batters Recommendation
# =====================================
st.markdown("---")
st.header("⚡ Auto Recommendation: Top 5 Batters")

min_matches = 10
min_avg = 45
min_sr = 90

positions = ['All'] + [str(int(pos)) for pos in sorted(all_players['batting_position'].dropna().unique())]
selected_pos = st.selectbox("Select Batting Position", positions , key='batting order')

# Normalize data
all_players['role'] = all_players['role'].str.strip().str.lower()
all_players['batting_position'] = all_players['batting_position'].astype(str).str.strip()

# Base filter
base_filter = (
    (all_players['matches'] >= min_matches) &
    (all_players['average'] >= min_avg) &
    (all_players['strike_rate'] >= min_sr)
)
# Apply based on position
if selected_pos == "All":
    filtered = all_players[base_filter]

elif selected_pos == '1':
    filtered = all_players[base_filter & (all_players['batting_position'] == selected_pos)]

elif selected_pos == '2':
    filtered = all_players[base_filter & (all_players['batting_position'] == selected_pos)]

elif selected_pos == '3':
    filtered = all_players[base_filter & (all_players['batting_position'] == selected_pos)]

elif selected_pos == '4':
    filtered = all_players[
        (all_players['matches'] >= min_matches) &
        (all_players['average'] >= min_avg - 10) &
        (all_players['strike_rate'] >= min_sr + 5) &
        (all_players['batting_position'] == selected_pos)
    ]
elif selected_pos == '5':
    filtered = all_players[
        (all_players['matches'] >= min_matches) &
        (all_players['average'] >= min_avg - 10) &
        (all_players['strike_rate'] > min_sr + 10) &
        (all_players['batting_position'] == selected_pos) &
        (all_players['role'].str.contains('wicket-keeper'))
    ]

elif selected_pos == '6':
    filtered = all_players[
        (all_players['matches'] >= min_matches) &
        (all_players['average'] >= min_avg - 10) &
        (all_players['strike_rate'] > min_sr + 10) &
        (all_players['batting_position'] == selected_pos)
    ]

elif selected_pos == '7':
    filtered = all_players[
        (
            (all_players['matches'] >= min_matches) &
            (all_players['average'] >= min_avg - 20) &
            (all_players['strike_rate'] >= min_sr + 5) &
            (all_players['batting_position'] == selected_pos)
        ) &
        (all_players['bowling_average'] < 35.0) |
        (all_players['role'] != 'batsman' ) &
        (all_players['role'] != 'wicket-keeper') 

    ]
else:
    filtered = all_players[base_filter]

top7 = filtered.sort_values(by=['average', 'strike_rate', 'runs' , 'bowling_average'], ascending=False).head(7)
st.dataframe(top7[['player', 'Team', 'matches', 'average', 'strike_rate', 'batting_position' , 'bowling_average']])

# =====================================
# 🧩 Team Builder (Unique Position Logic)
# =====================================
st.markdown("---")
st.header("🧩 Team Builder (1 Player per Position)")

if "selected_team" not in st.session_state:
    st.session_state.selected_team = []
if "used_positions" not in st.session_state:
    st.session_state.used_positions = set()

for i, row in top7.iterrows():
    btn_label = f"Add {row['player']} ({row['batting_position']})"
    if st.button(btn_label):
        if row['batting_position'] in st.session_state.used_positions:
            st.warning(f"⚠️ Position '{row['batting_position']}' already taken!")
        else:
            st.session_state.selected_team.append(row.to_dict())
            st.session_state.used_positions.add(row['batting_position'])
            st.success(f"✅ {row['player']} added to your team!")

if st.session_state.selected_team:
    st.dataframe(pd.DataFrame(st.session_state.selected_team)[['player', 'Team', 'average', 'strike_rate', 'batting_position']])

# =====================================
# 📊 Summary Comparison
# =====================================
fig3 = px.bar(
    top7,
    x='player',
    y=['average', 'strike_rate'],
    barmode='group',
    title="🏅 Top 5 Players: Average vs Strike Rate"
)
st.plotly_chart(fig3, use_container_width=True)

# =====================================
# 🔍 Player Search & Full Analysis
# =====================================
st.markdown("---")
st.header("🔍 Player Search & Analysis")

player_list = sorted(all_players['player'].dropna().unique().tolist())
selected_player = st.selectbox("Search Player", player_list)

if selected_player:
    player_data = all_players[all_players['player'] == selected_player]

    # Player Info
    player_row = player_data.iloc[0]
    col_img, col_info = st.columns([1, 2])

    # 🖼️ Image from CSV
    if 'image_url' in player_row and pd.notna(player_row['image_url']):
        col_img.image(player_row['image_url'], width=180)
    else:
        col_img.image("https://via.placeholder.com/150?text=No+Image", width=180)

    # 🎯 Player Summary Info
    col_info.markdown(f"### {player_row['player']}")
    col_info.markdown(f"**Team:** {player_row['Team']}")
    if 'role' in player_row:
        col_info.markdown(f"**Role:** {player_row['role']}")
    if 'batting_position' in player_row:
        col_info.markdown(f"**Batting Position:** {player_row['batting_position']}")

    # 🧮 Metrics
    cols = st.columns(4)
    cols[0].metric("Matches", int(player_data['matches'].mean()))
    if 'runs' in player_data.columns:
        cols[1].metric("Runs", int(player_data['runs'].mean()))
    if 'average' in player_data.columns:
        cols[2].metric("Average", round(player_data['average'].mean(), 2))
    if 'strike_rate' in player_data.columns:
        cols[3].metric("Strike Rate", round(player_data['strike_rate'].mean(), 2))

    # =====================================
    # Format-wise Stats
    # =====================================
    if 'Format' in player_data.columns:
        st.subheader("📊 Format-wise Performance")
        format_stats = player_data.groupby('Format')[['runs', 'matches', 'average', 'strike_rate']].mean().reset_index()
        fig_format = px.bar(format_stats, x='Format', y='runs', color='Format', title="Runs by Format")
        st.plotly_chart(fig_format, use_container_width=True)
st.subheader("🎯 Predict Player Runs") 
rf_model = RandomForestRegressor(n_estimators=200 ,random_state=42)
rf_model.fit(x_scaled , y)
selected_player = st.selectbox("Select the player and get the prediction of average" , df3['player'].unique())
player_data = df3[df3['player']== selected_player].iloc[0]
input_data = [[
    player_data['matches'],
    player_data['Innings'],
    player_data['runs'],
    player_data['strike_rate'],
    player_data['100s'],
    player_data['50s']
]]
predicted_average = rf_model.predict(input_data)[0]
st.metric(label=f"Predicted Batting Average for {selected_player}", value=round(predicted_average, 2))
st.header("Predict Player Next Year Performance")
players = sorted(year_wise_data['player'].unique().tolist())
selected_player = st.selectbox("Select the player and get the prediction of average" , players )
player_data = year_wise_data[year_wise_data['player']== selected_player].copy()
X = player_data[['matches', 'average', 'SR', '50s', '100s']] 
y = player_data['runs']
x_train , x_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200 , random_state=42)
model.fit(x_train , y_train)
latest = player_data.iloc[-1][['matches', 'average', 'SR', '50s', '100s']].values.reshape(1, -1)
predicted_next = model.predict(latest)[0]
next_year = player_data['year'].max()+1
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=player_data['year'],
    y=player_data['runs'],
    mode='lines+markers',
    name='Actual Runs',
    line=dict(color='blue' , width=3)
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
    title=f"📊 {selected_player} — Yearly Runs Trend", 
    xaxis_title="Year", 
    yaxis_title="Total Runs", 
    template="plotly_white" )
st.plotly_chart(fig, use_container_width=True)
st.success(f"🏏 **{selected_player}** is predicted to score approximately **{predicted_next:.0f} runs** in **{next_year}**.") 
st.info(f"📈 Estimated average per match: **{predicted_next / player_data.iloc[-1]['matches']:.1f} runs/match**")