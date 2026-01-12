import streamlit as st
from ..ai_features import get_ollama_response

def render_ai_chat(all_players):
    st.markdown("---")
    st.header("🤖 Ask the Expert (Ollama AI)")
    st.info("Ask any question about player stats or cricket trends. (Requires Ollama running)")
    
    # Simple context preparation (top players)
    top_context = all_players.sort_values(by='runs', ascending=False).head(10)[['player', 'Team', 'runs', 'average']].to_string()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something (e.g., Who is the best batsman in the dataset?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Expert is thinking..."):
                response = get_ollama_response(prompt, context_data=top_context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
