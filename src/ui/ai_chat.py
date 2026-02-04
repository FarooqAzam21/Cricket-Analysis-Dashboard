import streamlit as st
from ..ai_features import stream_ollama_response
import pandas as pd

def get_relevant_context(all_players, query):
    """Dynamically select relevant data based on the user's query."""
    query = str(query).lower()
    
    # Base context: Top overall performers
    relevant_df = all_players.sort_values(by='runs', ascending=False).head(5)
    
    # 1. Check for specific country/team
    teams = all_players['Team'].unique()
    for team in teams:
        if str(team).lower() in query:
            relevant_df = pd.concat([relevant_df, all_players[all_players['Team'] == team].head(10)])
            
    # 2. Check for player names
    players = all_players['player'].unique()
    for p in players:
        if str(p).lower() in query:
            relevant_df = pd.concat([relevant_df, all_players[all_players['player'] == p]])
            
    # 3. Bowling context if mentioned
    if any(word in query for word in ['bowler', 'wicket', 'economy', 'spinner', 'fast']):
        relevant_df = pd.concat([relevant_df, all_players.sort_values(by='wickets', ascending=False).head(10)])
        
    # Deduplicate and limit to 15 rows to avoid context window overflow
    relevant_df = relevant_df.drop_duplicates().head(15)
    
    # Format as a clean markdown table
    context_str = relevant_df[['player', 'Team', 'Format', 'runs', 'average', 'wickets', 'economy']].to_markdown(index=False)
    return context_str

def render_ai_chat(all_players):
    st.markdown("---")
    st.header("🤖 AI Expert Hub")
    
    # ELITE UI: Persona Selector
    with st.expander("👤 Choose Your AI Expert Persona", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        persona = st.radio(
            "Select Expert Viewpoint:",
            ["The Traditionalist", "The Aggressive Strategist", "The Data Scientist"],
            horizontal=True,
            help="Different personas provide advice with different priorities (e.g., consistency vs. impact)."
        )
        
        persona_instructions = {
            "The Traditionalist": "Focus on averages, consistency, and technique. High priority on stability.",
            "The Aggressive Strategist": "Value strike rate, boundary hitting, and match-winning impact above all else.",
            "The Data Scientist": "Use advanced metrics like Z-scores, standard deviations, and trend lines to justify advice."
        }
        st.caption(f"**Current Persona:** {persona} - *{persona_instructions[persona]}*")

    st.markdown(f"""
    <div class="elite-card" style="padding: 15px; border-left: 5px solid var(--primary);">
        <h4 style="margin:0; color:var(--primary-dark) !important;">Expert Hub Online</h4>
        <p style="margin:0; font-size: 0.9rem; opacity: 0.8;">Currently operating under <b>{persona}</b> logic with full context retrieval.</p>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about players, teams, or strategy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Progressively build the response
            full_response = ""
            message_placeholder = st.empty()
            
            with st.spinner("🔍 Consulting the expert..."):
                context = get_relevant_context(all_players, prompt)
                # Inject persona into context
                context += f"\n\n### EXPERT PERSONA INSTRUCTIONS:\nYou are {persona}. {persona_instructions[persona]}"
                
            for chunk in stream_ollama_response(prompt, context_data=context):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Add helpful examples
    if len(st.session_state.messages) == 0:
        st.divider()
        st.subheader("💡 Suggested Queries:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Analyze the current top-performing Nepali batsman.")
        with col2:
            st.info("Compare Babar Azam and Virat Kohli's T20 impact.")
        with col3:
            st.info("Who is the most consistent bowler in the dataset?")
