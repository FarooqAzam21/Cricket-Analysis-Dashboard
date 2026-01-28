import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..models import train_predict_metric, predict_yearwise

def render_next_match_prediction(df_batsman, df_allrounder=None, df_bowler=None, wicket_keepers=None):
    """Predict runs and strike rate for next match based on format-wise performance."""
    st.markdown("---")
    st.header("🎯 Next Match Performance Predictor")
    st.info("Predict how many runs and at what strike rate a player will perform in their next match based on historical trends.")

    try:
        # Combine all player data
        all_data = []
        
        # Combine Batsman and Wicket-Keeper as requested
        df_bat_wk = pd.DataFrame()
        if df_batsman is not None and not df_batsman.empty:
            df_bat_wk = pd.concat([df_bat_wk, df_batsman])
        if wicket_keepers is not None and not wicket_keepers.empty:
            df_bat_wk = pd.concat([df_bat_wk, wicket_keepers])
            
        if not df_bat_wk.empty:
            all_data.append(("Batsman & WK", df_bat_wk))
            
        if df_allrounder is not None and not df_allrounder.empty:
            all_data.append(("All-Rounder", df_allrounder))
        if df_bowler is not None and not df_bowler.empty:
            all_data.append(("Bowler", df_bowler))
        
        if not all_data:
            st.warning("No player data available.")
            return
        
        # Select player type
        player_types = [name for name, _ in all_data]
        selected_type = st.selectbox("Select Player Pool", player_types, key='player_type_select')
        
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
            
            # Show progress spinner while training models
            with st.spinner("Analyzing player patterns and training AI models..."):
                # Goal 1: Predict Runs
                model_runs, scaler_runs = train_predict_metric(df_format, features, 'runs')
                # Goal 2: Predict Strike Rate
                model_sr, scaler_sr = train_predict_metric(df_format, features, 'strike_rate')
            
            selected_player = st.selectbox(f"Select Player ({selected_format})", sorted(df_format['player'].unique()), key='player_select_match')
            player_row = df_format[df_format['player'] == selected_player].iloc[0]
            
            # Display player info and stats
            st.markdown(f"#### 👤 {selected_player} ({player_row.get('Team', 'N/A')})")
            
            col_stats = st.columns(4)
            col_stats[0].metric("Career Runs", int(player_row.get('runs', 0)))
            col_stats[1].metric("Avg", round(player_row.get('average', 0), 2))
            col_stats[2].metric("SR", round(player_row.get('strike_rate', 0), 2))
            col_stats[3].metric("Matches", int(player_row.get('matches', 0)))
            
            # Make predictions
            input_data = player_row[features].fillna(0).values.reshape(1, -1)
            
            # Predict Runs
            input_scaled_runs = scaler_runs.transform(input_data)
            predicted_runs = model_runs.predict(input_scaled_runs)[0]
            
            # Predict SR
            input_scaled_sr = scaler_sr.transform(input_data)
            predicted_sr = model_sr.predict(input_scaled_sr)[0]

            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"""
                <div style='background: rgba(16, 185, 129, 0.1); border: 2px solid #10B981; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                    <h5 style='margin: 0; color: #10B981;'>Predicted Runs</h5>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{round(predicted_runs, 0)}</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>Next Match Target</div>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f"""
                <div style='background: rgba(59, 130, 246, 0.1); border: 2px solid #3B82F6; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                    <h5 style='margin: 0; color: #3B82F6;'>Predicted Strike Rate</h5>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{round(predicted_sr, 1)}</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>Intent Forecast</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add insight text
            st.markdown(f"""
            > [!NOTE]
            > Based on {selected_player}'s career format stats, the AI expects a performance around **{round(predicted_runs, 0)} runs** at **{round(predicted_sr, 1)} SR**. 
            > This suggests a **{'high-impact' if predicted_sr > 120 else 'steady'}** innings for the upcoming match.
            """)
            
        else:
            st.warning("No data available for selected format.")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

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
