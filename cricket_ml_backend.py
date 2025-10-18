from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import Ridge
app = FastAPI(title="Cricket ML Backend")
try:
    df = pd.read_csv("players.csv")
except FileNotFoundError:
    raise Exception("⚠️ Please create a 'players.csv' file in the same folder.")
def train_model(player_df):
    if len(player_df) < 3:  # need enough matches
        return None

    X = np.arange(len(player_df)).reshape(-1, 1)  # match index
    y = player_df["runs"].values
    model = Ridge()
    model.fit(X, y)
    return model
class Best11Request(BaseModel):
    venue: str
    pitch: str = "balanced"
    available_players: list[str] = []
@app.get("/")
def root():
    return {"message": "🏏 Cricket ML Backend is running!"}


@app.get("/player/{player_name}")
def get_player_analysis(player_name: str):
    player_data = df[df["player"].str.lower() == player_name.lower()]

    if player_data.empty:
        raise HTTPException(status_code=404, detail="Player not found")

    avg = player_data["average"].mean()
    sr = player_data["strike_rate"].mean()
    model = train_model(player_data)
    prediction = None
    if model:
        next_match_index = np.array([[len(player_data) + 1]])
        prediction = model.predict(next_match_index)[0]

    return {
        "name": player_name,
        "matches": int(player_data["matches"].sum()),
        "batting_avg": round(avg, 2),
        "strike_rate": round(sr, 2),
        "prediction_next_match_runs": round(float(prediction), 2) if prediction else "Not enough data"
    }


@app.get("/best11")
def best11(venue: str, pitch: str = "balanced"):
    venue_players = df[df["venue"].str.lower() == venue.lower()]

    if venue_players.empty:
        raise HTTPException(status_code=404, detail="No data for this venue")

    # Sort by batting average (for simplicity, improve later)
    best_players = venue_players.sort_values("average", ascending=False).head(11)

    team = []
    for _, row in best_players.iterrows():
        team.append({
            "name": row["player"],
            "avg": row["average"],
            "strike_rate": row["strike_rate"],
            "wickets": row.get("wickets", None)
        })

    return {
        "venue": venue,
        "team": team,
        "reasoning": f"Selected top 11 players by batting average for {venue}"
    }
