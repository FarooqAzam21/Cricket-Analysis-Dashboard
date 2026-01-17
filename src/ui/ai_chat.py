import streamlit as st
from ..ai_features import get_ollama_response

def render_ai_chat(all_players):
    st.markdown("---")
    st.header("🤖 Ask the Expert (Ollama AI)")
    st.info("🔗 **Ollama**: Ask questions about player stats, trends & predictions. If Ollama is unavailable, we'll provide rule-based analysis.")
    
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
            with st.spinner("🤔 Thinking..."):
                response = get_ollama_response(prompt, context_data=top_context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add helpful examples
    if len(st.session_state.messages) == 0:
        st.divider()
        st.subheader("💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("📊 Who is the best batsman?")
        with col2:
            st.caption("🎯 Top bowlers this season?")
        with col3:
            st.caption("📈 Compare player performance")
