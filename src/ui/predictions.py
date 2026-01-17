import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..models import train_predict_runs, predict_yearwise

def render_next_match_prediction(df_batsman, df_allrounder=None, df_bowler=None, wicket_keepers=None):
    """Predict runs for next match based on format-wise performance for all player types."""
    st.markdown("---")
    st.header("🎯 Next Match Runs Predictor")
    st.info("Predict how many runs a player will score in their next match based on their format-wise statistics and recent performance.")

    try:
        # Combine all player data
        all_data = []
        if df_batsman is not None and not df_batsman.empty:
            all_data.append(("Batsman", df_batsman))
        if df_allrounder is not None and not df_allrounder.empty:
            all_data.append(("All-Rounder", df_allrounder))
        if df_bowler is not None and not df_bowler.empty:
            all_data.append(("Bowler", df_bowler))
        if wicket_keepers is not None and not wicket_keepers.empty:
            all_data.append(("Wicket-Keeper", wicket_keepers))
        
        if not all_data:
            st.warning("No player data available.")
            return
        
        # Select player type
        player_types = [name for name, _ in all_data]
        selected_type = st.selectbox("Select Player Type", player_types, key='player_type_select')
        
        # Get data for selected type
        df_players = dict(all_data)[selected_type]
        
        if df_players.empty:
            st.warning(f"No {selected_type} data available.")
            return
            
        formats = sorted(df_players['Format'].unique())
        selected_format = st.selectbox("Select Format", formats, key='format_select_match')
        df_format = df_players[df_players['Format'] == selected_format]

        if not df_format.empty:
            features = ['matches', 'Innings', 'average', 'strike_rate', '100s', '50s']
            target = 'runs'
            
            # Show progress spinner while training model
            with st.spinner("Training prediction model..."):
                rf_model, scaler = train_predict_runs(df_format, features, target)
            
            selected_player = st.selectbox(f"Select {selected_type} ({selected_format})", df_format['player'].unique(), key='player_select_match')
            player_row = df_format[df_format['player'] == selected_player].iloc[0]
            
            # Display player info and stats
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**Type:** {selected_type}")
                st.markdown(f"**Team:** {player_row.get('Team', 'N/A')}")
            
            with col2:
                # Display player stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Career Runs", int(player_row.get('runs', 0)))
                col2.metric("Average", round(player_row.get('average', 0), 2))
                col3.metric("Strike Rate", round(player_row.get('strike_rate', 0), 2))
                col4.metric("Matches", int(player_row.get('matches', 0)))
            
            # Make prediction
            input_data = player_row[features].values.reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            predicted_run = rf_model.predict(input_scaled)[0]

            st.success(f"**Predicted Runs for {selected_player}'s Next Match:** {round(predicted_run, 0)} runs")
            
            # Add confidence range
            confidence_range = round(predicted_run * 0.15, 0)  # ±15% confidence range
            st.info(f"Confidence Range: {round(predicted_run - confidence_range, 0)} - {round(predicted_run + confidence_range, 0)} runs")
            
        else:
            st.warning("No data available for selected format.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

def render_yearly_prediction(yearwise_data):
    """Predict player performance for next year based on historical trends."""
    st.markdown("---")
    st.header("📈 Yearly Performance Predictor")
    st.info("Analyze historical performance trends and predict how a player will perform in the next year.")

    if yearwise_data.empty or 'player' not in yearwise_data.columns:
        st.warning("No year-wise data available.")
        return
        
    players = sorted(yearwise_data['player'].unique().tolist())
    sel_player = st.selectbox("Select Player for Yearly Prediction", players, key='player_select_yearly')

    if sel_player:
        player_df = yearwise_data[yearwise_data['player'] == sel_player].sort_values('year').copy()
        
        # Ensure numeric conversion for key columns
        player_df['year'] = pd.to_numeric(player_df['year'], errors='coerce')
        player_df['runs'] = pd.to_numeric(player_df['runs'], errors='coerce')
        
        # Remove rows with NaN values
        player_df = player_df.dropna(subset=['year', 'runs'])
        
        if len(player_df) >= 3:
            # Display historical stats
            st.subheader(f"Historical Performance - {sel_player}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Years in Dataset", len(player_df))
            col2.metric("Career Total Runs", int(player_df['runs'].sum()))
            col3.metric("Average per Year", round(player_df['runs'].mean(), 0))
            
            # Predict next year
            predicted_next, _ = predict_yearwise(player_df)
            next_year = int(player_df['year'].max()) + 1
            
            # Create visualization with trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=player_df['year'], 
                y=player_df['runs'], 
                mode='lines+markers', 
                name='Actual Runs',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=[next_year], 
                y=[predicted_next], 
                mode='markers+text', 
                name='Predicted', 
                marker=dict(color='red', size=15, symbol='star'),
                text=[f"{int(predicted_next)}"],
                textposition="top center"
            ))
            
            fig.update_layout(
                title=f"Performance Trend & Prediction - {sel_player}",
                xaxis_title="Year",
                yaxis_title="Runs Scored",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction result
            st.success(f"**Predicted Performance for {next_year}:** {int(predicted_next)} runs")
            
            # Calculate trend
            if len(player_df) >= 2:
                recent_avg = player_df.tail(2)['runs'].mean()
                overall_avg = player_df['runs'].mean()
                trend = "📈 Improving" if recent_avg > overall_avg else "📉 Declining"
                st.info(f"**Trend:** {trend} | Recent Avg: {round(recent_avg, 0)} | Overall Avg: {round(overall_avg, 0)}")
        else:
            st.warning(f"Not enough data for {sel_player} (need 3+ years of history). Found {len(player_df)} years.")

def render_predictions(df_batsman, yearwise_data):
    """Combined prediction view for backward compatibility."""
    render_next_match_prediction(df_batsman)
    st.divider()
    render_yearly_prediction(yearwise_data)
