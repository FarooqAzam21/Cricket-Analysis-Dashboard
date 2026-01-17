import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def train_predict_runs(df_format, features, target):
    """Train RF and predict runs for a player in a specific format. Optimized model."""
    x = df_format[features].fillna(0)
    y = df_format[target] / df_format['matches'].replace(0, 1) # runs per match

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Optimized: reduced n_estimators from 200 to 100, added max_depth and parallel processing
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(x_scaled, y)
    
    return rf_model, scaler

def predict_yearwise(player_year_df):
    """Predict next year performance based on historical data. Optimized model."""
    required_cols = ['matches', 'average', 'SR', '50s', '100s']
    Xy = player_year_df[required_cols].fillna(0)
    yy = player_year_df['runs'].fillna(0)
    
    x_train, x_test, y_train, y_test = train_test_split(Xy, yy, test_size=0.2, random_state=42)
    # Optimized: reduced n_estimators from 200 to 80, added max_depth and parallel processing
    model_year = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
    model_year.fit(x_train, y_train)
    
    # Latest performance for prediction
    latest = player_year_df.iloc[-1][required_cols].values.reshape(1, -1)
    predicted_next = model_year.predict(latest)[0]
    
    return predicted_next, model_year
