import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..models import train_predict_runs, predict_yearwise

def render_predictions(df_batsman, year_wise_data):
    st.markdown("---")
    st.subheader("🎯 Predict Batsman Runs (Format-wise)")

    try:
        formats = sorted(df_batsman['Format'].unique())
        selected_format = st.selectbox("Select Format", formats, key='format_select')
        df_format = df_batsman[df_batsman['Format'] == selected_format]

        if not df_format.empty:
            features = ['matches', 'Innings', 'average', 'strike_rate', '100s', '50s']
            target = 'runs'
            
            rf_model, scaler = train_predict_runs(df_format, features, target)
            
            selected_player = st.selectbox(f"Select Player ({selected_format})", df_format['player'].unique())
            player_row = df_format[df_format['player'] == selected_player].iloc[0]
            
            input_data = player_row[features].values.reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            predicted_run = rf_model.predict(input_scaled)[0]

            st.metric(label=f"Predicted Next Match Runs for {selected_player}", value=f"{round(predicted_run, 0)} runs")
        else:
            st.warning("No data for format.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.header("Predict Player Next Year Performance")

    if not year_wise_data.empty and 'player' in year_wise_data.columns:
        players = sorted(year_wise_data['player'].unique().tolist())
        sel_player = st.selectbox("Select player for year-wise prediction", players)

        if sel_player:
            player_df = year_wise_data[year_wise_data['player'] == sel_player].sort_values('year')
            if len(player_df) >= 3:
                predicted_next, _ = predict_yearwise(player_df)
                next_year = int(player_df['year'].max()) + 1
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=player_df['year'], y=player_df['runs'], mode='lines+markers', name='Actual Runs'))
                fig.add_trace(go.Scatter(x=[next_year], y=[predicted_next], mode='markers+text', name='Predicted', 
                                         marker=dict(color='red', size=12, symbol='star'), text=[f"{int(predicted_next)}"], textposition="top center"))
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"Predicted to score ~{int(predicted_next)} runs in {next_year}.")
            else:
                st.warning("Not enough data (need 3+ years).")
