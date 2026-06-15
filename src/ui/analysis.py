import pandas as pd
import plotly.express as px
import streamlit as st


BATTER_METRICS = {
    "Runs": ("runs", False),
    "Batting Average": ("average", False),
    "Strike Rate": ("strike_rate", False),
}

BOWLER_METRICS = {
    "Wickets": ("wickets", False),
    "Bowling Average": ("bowling_average", True),
    "Bowling Strike Rate": ("bowling_strike_rate", True),
}


def _to_numeric(df, columns):
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = (
                result[col]
                .astype(str)
                .str.replace("-", "0", regex=False)
                .str.strip()
            )
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    return result


def _filtered_career_data(df, selected_format, selected_teams, min_matches):
    data = df.copy()
    if selected_format != "All" and "Format" in data.columns:
        data = data[data["Format"] == selected_format]
    if selected_teams and "Team" in data.columns:
        data = data[data["Team"].isin(selected_teams)]
    if "matches" in data.columns:
        data = data[data["matches"] >= min_matches]
    return data


def _player_options(df):
    if df is None or df.empty or "player" not in df.columns:
        return []
    return sorted(df["player"].dropna().unique().tolist())


def _render_ranking_table(data, metric_label, metric_col, ascending, top_n, title):
    if metric_col not in data.columns or data.empty:
        st.info(f"No data available for {metric_label}.")
        return

    ranked = data[data[metric_col] > 0].sort_values(metric_col, ascending=ascending).head(top_n)
    if ranked.empty:
        st.info(f"No players match the current filters for {metric_label}.")
        return

    display_cols = [
        col
        for col in ["player", "Team", "Format", "role", "matches", "runs", "wickets", "average", "strike_rate", "bowling_average", "bowling_strike_rate", "economy"]
        if col in ranked.columns
    ]

    fig = px.bar(
        ranked,
        x="player",
        y=metric_col,
        color="Team" if "Team" in ranked.columns else None,
        title=title,
        template="plotly_white",
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(ranked[display_cols], width="stretch", hide_index=True)


def _render_yearly_graphs(year_wise, selected_players, year_range, selected_format):
    if year_wise is None or year_wise.empty:
        st.info("No year-wise data is available.")
        return

    required = {"player", "year", "runs", "average", "SR", "matches"}
    if not required.issubset(set(year_wise.columns)):
        missing = sorted(required - set(year_wise.columns))
        st.info(f"Year-wise data is missing required columns: {missing}")
        return

    trend = year_wise.copy()
    trend = _to_numeric(trend, ["year", "matches", "runs", "average", "SR", "100s", "50s"])

    if selected_players:
        trend = trend[trend["player"].isin(selected_players)]
    if selected_format != "All" and "Format" in trend.columns:
        trend = trend[trend["Format"] == selected_format]

    trend = trend[(trend["year"] >= year_range[0]) & (trend["year"] <= year_range[1])]

    if trend.empty:
        st.info("No yearly records match the current player, year, and format filters.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            trend,
            x="year",
            y="runs",
            color="player",
            markers=True,
            title="Player Runs by Year",
            template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.line(
            trend,
            x="year",
            y="average",
            color="player",
            markers=True,
            title="Batting Average by Year",
            template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(
            trend,
            x="year",
            y="SR",
            color="player",
            markers=True,
            title="Strike Rate by Year",
            template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")

    with c4:
        yearly_total = trend.groupby("year", as_index=False)[["matches", "runs"]].sum()
        fig = px.bar(
            yearly_total,
            x="year",
            y="runs",
            title="Filtered Total Runs by Year",
            template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        trend.sort_values(["player", "year"])[["player", "year", "Format", "matches", "runs", "average", "SR", "100s", "50s"]],
        width="stretch",
        hide_index=True,
    )


def render_player_analysis(all_players, year_wise=None):
    st.markdown("---")
    st.header("Player Analytics & Year-wise Performance")

    if all_players is None or all_players.empty:
        st.error("No player data is available.")
        return

    all_players = _to_numeric(
        all_players,
        ["matches", "Innings", "runs", "wickets", "average", "strike_rate", "bowling_average", "bowling_strike_rate", "economy"],
    )

    formats = ["All"] + sorted(all_players["Format"].dropna().unique().tolist()) if "Format" in all_players.columns else ["All"]
    teams = sorted(all_players["Team"].dropna().unique().tolist()) if "Team" in all_players.columns else []

    with st.expander("Ranking Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selected_format = st.selectbox("Format", formats, key="player_rank_format")
        with c2:
            min_matches = st.number_input("Minimum Matches", min_value=0, value=10, step=5, key="player_rank_min_matches")
        with c3:
            top_n = st.slider("Top N Players", min_value=5, max_value=50, value=15, step=5, key="player_rank_top_n")
        with c4:
            selected_teams = st.multiselect("Teams", teams, key="player_rank_teams")

    filtered = _filtered_career_data(all_players, selected_format, selected_teams, min_matches)

    rank_tabs = st.tabs(["Batters", "Bowlers", "Player Profile", "Year-wise Graphs"])

    with rank_tabs[0]:
        st.subheader("Batter Rankings")
        metric_label = st.selectbox("Rank batters according to", list(BATTER_METRICS.keys()), key="batter_metric")
        metric_col, ascending = BATTER_METRICS[metric_label]
        batter_data = filtered[filtered["runs"] > 0] if "runs" in filtered.columns else filtered
        _render_ranking_table(
            batter_data,
            metric_label,
            metric_col,
            ascending,
            top_n,
            f"Top {top_n} Batters by {metric_label}",
        )

    with rank_tabs[1]:
        st.subheader("Bowler Rankings")
        metric_label = st.selectbox("Rank bowlers according to", list(BOWLER_METRICS.keys()), key="bowler_metric")
        metric_col, ascending = BOWLER_METRICS[metric_label]
        bowler_data = filtered[filtered["wickets"] > 0] if "wickets" in filtered.columns else filtered
        _render_ranking_table(
            bowler_data,
            metric_label,
            metric_col,
            ascending,
            top_n,
            f"Top {top_n} Bowlers by {metric_label}",
        )

    with rank_tabs[2]:
        st.subheader("Individual Player Profile")
        player_list = _player_options(all_players)
        if not player_list:
            st.info("No player names are available.")
            return

        default_idx = 0
        if "preselected_player" in st.session_state and st.session_state.preselected_player in player_list:
            default_idx = player_list.index(st.session_state.preselected_player)
            del st.session_state.preselected_player

        selected_player = st.selectbox("Search Player", player_list, index=default_idx, key="player_search_box")
        player_data = all_players[all_players["player"] == selected_player]

        if player_data.empty:
            st.info("No records found for the selected player.")
            return

        player_row = player_data.iloc[0]
        st.markdown(
            f"""
            <div class="elite-card">
                <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
                    <img src="{player_row.get('image_url', 'https://via.placeholder.com/150?text=No+Img')}"
                         style="width: 150px; height: 150px; border-radius: 75px; object-fit: cover; border: 4px solid var(--primary);">
                    <div style="flex-grow: 1;">
                        <h1 style="margin: 0; border: none; padding: 0;">{player_row.get('player','Unknown')}</h1>
                        <p style="font-size: 1.1rem; margin: 8px 0; color: var(--primary-dark) !important; font-weight: 600;">
                            {player_row.get('Team','-')} | {player_row.get('role','-')}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "Format" in player_data.columns:
            profile_tabs = st.tabs([str(fmt) for fmt in player_data["Format"].dropna().unique()])
            for idx, fmt in enumerate(player_data["Format"].dropna().unique()):
                with profile_tabs[idx]:
                    fmt_data = player_data[player_data["Format"] == fmt]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Matches", int(fmt_data["matches"].sum()))
                    c1.metric("Runs", int(fmt_data["runs"].sum()))
                    c2.metric("Average", round(fmt_data["average"].mean(), 2))
                    c2.metric("Strike Rate", round(fmt_data["strike_rate"].mean(), 2))
                    c3.metric("Wickets", int(fmt_data["wickets"].sum()))
                    c3.metric("Bowling Avg", round(fmt_data["bowling_average"].mean(), 2))

        format_stats = player_data.groupby("Format", as_index=False)[["runs", "matches", "average", "strike_rate", "wickets", "bowling_average"]].mean()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(format_stats, x="Format", y="runs", color="Format", title=f"Runs by Format for {selected_player}", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.scatter(
                format_stats,
                x="average",
                y="strike_rate",
                size="runs",
                color="Format",
                title="Average vs Strike Rate",
                template="plotly_white",
                text="Format",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")

    with rank_tabs[3]:
        st.subheader("Year-wise Player Performance")
        if year_wise is None or year_wise.empty:
            st.info("No year-wise data is available.")
            return

        year_wise = _to_numeric(year_wise, ["year", "matches", "runs", "average", "SR", "100s", "50s"])
        min_year = int(year_wise["year"].min())
        max_year = int(year_wise["year"].max())
        year_players = _player_options(year_wise)
        default_players = year_players[:3]

        yc1, yc2, yc3 = st.columns([2, 1, 1])
        with yc1:
            selected_year_players = st.multiselect(
                "Players for Year-wise Graph",
                year_players,
                default=default_players,
                key="yearwise_players",
            )
        with yc2:
            year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year), key="yearwise_range")
        with yc3:
            year_format_options = ["All"] + sorted(year_wise["Format"].dropna().unique().tolist()) if "Format" in year_wise.columns else ["All"]
            year_format = st.selectbox("Year-wise Format", year_format_options, key="yearwise_format")

        _render_yearly_graphs(year_wise, selected_year_players, year_range, year_format)
