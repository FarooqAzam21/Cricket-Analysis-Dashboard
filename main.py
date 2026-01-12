import streamlit as st
from src.config import apply_custom_styles, MENU_OPTIONS
from src.data_loader import load_all_data
from src.ui.format_wise import render_format_analysis
from src.ui.team_builder import render_team_builder
from src.ui.comparison import render_comparison
from src.ui.analysis import render_player_analysis
from src.ui.predictions import render_predictions
from src.ui.smart_scout import render_smart_scout
from src.ui.ai_chat import render_ai_chat

def main():
    # 1. Setup config
    apply_custom_styles()
    
    # 2. Sidebar elements
    st.sidebar.title("Cricket Analysis Menu")
    menu = st.sidebar.radio("Navigate to", MENU_OPTIONS)
    st.sidebar.markdown("---")
    st.sidebar.info("Developed by **Farooq Azam**")
    
    # 3. Load Data
    all_players, df_batsman, df_allrounder, df_bowler, year_wise, batsmen, all_rounders, wicket_keepers = load_all_data()
    
    if all_players is None:
        st.stop()

    st.title("🏏 Cricket Analytics Dashboard")

    # 4. Global Filters
    teams = ['All']
    if 'Team' in all_players.columns:
        teams += sorted(all_players['Team'].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Select Team", teams)

    # Note: Filtering 'data' here if needed, but many sub-pages use their own filters
    # For now, we pass the dataframes as needed.

    # 5. Routing
    if menu == "Format Wise Analysis":
        render_format_analysis(batsmen, all_rounders, df_bowler, wicket_keepers)
    elif menu == "Select Playing 11":
        render_team_builder(all_players)
    elif menu == "Player Comparison":
        render_comparison(all_players)
    elif menu == "Player Analysis":
        render_player_analysis(all_players)
    elif menu == "Predict Runs":
        render_predictions(df_batsman, year_wise)
    elif menu == "Smart Scout (AI)":
        render_smart_scout(all_players)
    elif menu == "Ask Expert (AI)":
        render_ai_chat(all_players)

if __name__ == "__main__":
    main()
