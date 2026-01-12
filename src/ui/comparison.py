import streamlit as st
import plotly.graph_objects as go

def render_comparison(all_players):
    st.markdown("---")
    st.subheader("⚔ Player Comparison")

    try:
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Select Player 1", all_players['player'].unique(), key="p1")
        with col2:
            player2 = st.selectbox("Select Player 2", all_players['player'].unique(), key="p2")

        p1_data = all_players[all_players['player'] == player1]
        p2_data = all_players[all_players['player'] == player2]

        if 'Format' in all_players.columns:
            formats = all_players['Format'].unique()
            selected_format = st.selectbox("Select Format", formats, key="fmt_cmp")

            p1_fmt = p1_data[p1_data['Format'] == selected_format]
            p2_fmt = p2_data[p2_data['Format'] == selected_format]

            if not p1_fmt.empty and not p2_fmt.empty:
                p1_row = p1_fmt.iloc[0]
                p2_row = p2_fmt.iloc[0]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(f"https://source.unsplash.com/400x400/?{player1},cricketer", caption=player1)
                    for stat in ['matches', 'Innings', 'runs', 'average', 'strike_rate', '100s', '50s']:
                        st.metric(stat.capitalize(), p1_row[stat])

                with col_b:
                    st.image(f"https://source.unsplash.com/400x400/?{player2},cricketer", caption=player2)
                    for stat in ['matches', 'Innings', 'runs', 'average', 'strike_rate', '100s', '50s']:
                        st.metric(stat.capitalize(), p2_row[stat])

                # Radar chart
                categories = ['matches', 'Innings', 'runs', 'average', 'strike_rate', '100s', '50s']
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=[p1_row[c] for c in categories], theta=categories, fill='toself', name=player1))
                fig.add_trace(go.Scatterpolar(r=[p2_row[c] for c in categories], theta=categories, fill='toself', name=player2))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title=f"{selected_format} Comparison")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data not available for selected format.")
    except Exception as e:
        st.error(f"Comparison error: {e}")
