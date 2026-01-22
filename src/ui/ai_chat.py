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
    st.header("🤖 Ask the Expert (Ollama AI)")
    st.info("🔗 **Ollama**: Now featuring **Streaming Responses** and **Dynamic Context Retrieval**.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something (e.g., 'Tell me about Virat Kohli' or 'Top Australian bowlers')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Progressively build the response
            full_response = ""
            message_placeholder = st.empty()
            
            with st.spinner("🔍 Fetching relevant data..."):
                context = get_relevant_context(all_players, prompt)
                
            for chunk in stream_ollama_response(prompt, context_data=context):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Add helpful examples
    if len(st.session_state.messages) == 0:
        st.divider()
        st.subheader("💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("- Who is the best batsman?")
        with col2:
            st.markdown("- Top Australian bowlers?")
        with col3:
            st.markdown("- Compare India vs England stats")
