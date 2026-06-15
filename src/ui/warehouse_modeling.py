import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


NUMERIC_COLUMNS = [
    "matches",
    "Innings",
    "runs",
    "wickets",
    "average",
    "strike_rate",
    "bowling_average",
    "economy",
    "100s",
    "50s",
]


def _safe_numeric(df, columns):
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


def _quality_score(df):
    if df is None or df.empty:
        return 0

    completeness = 1 - (df.isna().sum().sum() / max(df.size, 1))
    required_cols = ["player", "Team", "Format", "matches", "runs", "wickets", "role"]
    present_required = sum(1 for col in required_cols if col in df.columns) / len(required_cols)

    if {"player", "Team", "Format"}.issubset(df.columns):
        duplicate_rate = df.duplicated(subset=["player", "Team", "Format"]).mean()
    else:
        duplicate_rate = df.duplicated().mean()

    return round(((completeness * 0.45) + (present_required * 0.35) + ((1 - duplicate_rate) * 0.20)) * 100, 1)


def _schema_profile(df):
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_pct": round(df[col].notna().mean() * 100, 1),
                "unique_values": int(df[col].nunique(dropna=True)),
                "sample_value": "" if df[col].dropna().empty else str(df[col].dropna().iloc[0])[:50],
            }
        )
    return pd.DataFrame(rows)


def _source_inventory(all_players, year_wise):
    return pd.DataFrame(
        [
            {
                "layer": "Raw/Staging",
                "asset": "odi_batsman.csv",
                "grain": "one player-format batting record",
                "business_use": "batting analytics, player dimension enrichment",
            },
            {
                "layer": "Raw/Staging",
                "asset": "odi_bowler.csv",
                "grain": "one player-format bowling record",
                "business_use": "bowling analytics, player comparison",
            },
            {
                "layer": "Raw/Staging",
                "asset": "odi_all_rounders.csv",
                "grain": "one player-format all-rounder record",
                "business_use": "role modeling, balanced player scoring",
            },
            {
                "layer": "Raw/Staging",
                "asset": "yearwise_data.csv",
                "grain": "one player-year record",
                "business_use": "trend analysis and forecasting",
            },
            {
                "layer": "Analytics Mart",
                "asset": "all_players",
                "grain": "one player-team-format record",
                "business_use": "BI dashboard, AI scouting, modeling features",
            },
        ]
    )


def _build_model_mart(all_players):
    df = _safe_numeric(all_players, NUMERIC_COLUMNS)
    group_cols = [col for col in ["Team", "Format", "role"] if col in df.columns]

    if not group_cols:
        return pd.DataFrame()

    mart = (
        df.groupby(group_cols, dropna=False)
        .agg(
            players=("player", "nunique"),
            total_matches=("matches", "sum"),
            total_runs=("runs", "sum"),
            total_wickets=("wickets", "sum"),
            avg_strike_rate=("strike_rate", "mean"),
            avg_batting_average=("average", "mean"),
            avg_economy=("economy", "mean"),
        )
        .reset_index()
    )
    return mart.round(2)


def render_warehouse_modeling(all_players, year_wise):
    """Render a data warehousing, data modeling, and visualization readiness dashboard."""

    if all_players is None or all_players.empty:
        st.error("No analytics data is loaded. Please check the CSV sources.")
        return

    all_players = _safe_numeric(all_players, NUMERIC_COLUMNS)
    year_wise = pd.DataFrame() if year_wise is None else year_wise.copy()

    st.title("Data Warehousing & Modeling")
    st.caption("Warehouse design, data quality, dimensional modeling, analytics marts, and visualization readiness.")

    total_rows = len(all_players)
    total_columns = len(all_players.columns)
    quality = _quality_score(all_players)
    model_mart = _build_model_mart(all_players)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Warehouse Rows", f"{total_rows:,}")
    k2.metric("Modeled Columns", total_columns)
    k3.metric("Quality Score", f"{quality}%")
    k4.metric("Analytics Mart Rows", f"{len(model_mart):,}")

    tab_overview, tab_model, tab_quality, tab_visuals, tab_mart = st.tabs(
        [
            "Warehouse Overview",
            "Dimensional Model",
            "Data Quality",
            "Visualization Analytics",
            "Modeling Mart",
        ]
    )

    with tab_overview:
        st.subheader("Source-to-Mart Inventory")
        st.dataframe(_source_inventory(all_players, year_wise), width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Format" in all_players.columns:
                fmt_counts = all_players["Format"].value_counts().reset_index()
                fmt_counts.columns = ["Format", "Rows"]
                fig = px.bar(fmt_counts, x="Format", y="Rows", title="Warehouse Rows by Format", color="Format")
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "role" in all_players.columns:
                role_counts = all_players["role"].fillna("Unknown").value_counts().head(12).reset_index()
                role_counts.columns = ["Role", "Rows"]
                fig = px.pie(role_counts, names="Role", values="Rows", title="Role Distribution")
                st.plotly_chart(fig, width="stretch")

        st.subheader("Schema Profile")
        st.dataframe(_schema_profile(all_players), width="stretch", hide_index=True)

    with tab_model:
        st.subheader("Recommended Star Schema")
        st.markdown(
            """
            **Fact Player Performance** is the central analytical table at the grain of one player, team, and format.

            Dimensions:
            - **Dim Player**: player name, role, batting position, profile image
            - **Dim Team**: team/country
            - **Dim Format**: ODI, T20, Test
            - **Dim Time**: year-level trend records from `yearwise_data.csv`
            - **Dim Role**: batter, bowler, all-rounder, wicket-keeper and bowling style groups
            """
        )

        fig = go.Figure()
        nodes = {
            "Fact Player Performance": (0.5, 0.5),
            "Dim Player": (0.15, 0.78),
            "Dim Team": (0.85, 0.78),
            "Dim Format": (0.15, 0.22),
            "Dim Time": (0.85, 0.22),
            "Dim Role": (0.5, 0.9),
        }
        for name, (x, y) in nodes.items():
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=42, color="#10b981" if name.startswith("Fact") else "#2563eb"),
                    text=[name],
                    textposition="bottom center",
                    showlegend=False,
                )
            )
        for name, (x, y) in nodes.items():
            if name != "Fact Player Performance":
                fig.add_shape(type="line", x0=0.5, y0=0.5, x1=x, y1=y, line=dict(color="#94a3b8", width=2))
        fig.update_layout(
            height=520,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            title="Logical Dimensional Model",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")

    with tab_quality:
        st.subheader("Data Quality Checks")
        missing = all_players.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["Missing %"] = (missing["Missing Values"] / max(len(all_players), 1) * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(missing.sort_values("Missing %", ascending=False).head(15), x="Column", y="Missing %", title="Top Missing Columns")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, width="stretch")
        with c2:
            if {"player", "Team", "Format"}.issubset(all_players.columns):
                duplicate_count = int(all_players.duplicated(subset=["player", "Team", "Format"]).sum())
            else:
                duplicate_count = int(all_players.duplicated().sum())
            checks = pd.DataFrame(
                [
                    {"check": "Required player column", "status": "Pass" if "player" in all_players.columns else "Fail"},
                    {"check": "Required team column", "status": "Pass" if "Team" in all_players.columns else "Fail"},
                    {"check": "Required format column", "status": "Pass" if "Format" in all_players.columns else "Fail"},
                    {"check": "Duplicate business keys", "status": "Pass" if duplicate_count == 0 else f"Review {duplicate_count}"},
                    {"check": "Numeric measures converted", "status": "Pass"},
                ]
            )
            st.dataframe(checks, width="stretch", hide_index=True)

        st.subheader("Null and Uniqueness Matrix")
        st.dataframe(missing, width="stretch", hide_index=True)

    with tab_visuals:
        st.subheader("Visualization Analytics")
        c1, c2 = st.columns(2)
        with c1:
            if {"Team", "runs", "wickets"}.issubset(all_players.columns):
                team_perf = (
                    all_players.groupby("Team", dropna=False)[["runs", "wickets"]]
                    .sum()
                    .reset_index()
                    .sort_values("runs", ascending=False)
                    .head(15)
                )
                fig = px.scatter(
                    team_perf,
                    x="runs",
                    y="wickets",
                    size="runs",
                    color="Team",
                    hover_name="Team",
                    title="Team Run/Wicket Production",
                )
                st.plotly_chart(fig, width="stretch")
        with c2:
            if {"Format", "runs", "wickets"}.issubset(all_players.columns):
                format_perf = all_players.groupby("Format")[["runs", "wickets"]].sum().reset_index()
                fig = px.bar(format_perf, x="Format", y=["runs", "wickets"], barmode="group", title="Measures by Format")
                st.plotly_chart(fig, width="stretch")

        if not year_wise.empty and {"year", "runs"}.issubset(year_wise.columns):
            yw = year_wise.copy()
            yw["year"] = pd.to_numeric(yw["year"], errors="coerce")
            yw["runs"] = pd.to_numeric(yw["runs"], errors="coerce").fillna(0)
            trend = yw.groupby("year", dropna=True)["runs"].sum().reset_index()
            fig = px.line(trend, x="year", y="runs", markers=True, title="Year-wise Runs Trend")
            st.plotly_chart(fig, width="stretch")

    with tab_mart:
        st.subheader("Analytics Mart Preview")
        st.dataframe(model_mart, width="stretch", hide_index=True)

        st.subheader("Model-Ready Feature Columns")
        feature_cols = [
            "matches",
            "runs",
            "wickets",
            "average",
            "strike_rate",
            "bowling_average",
            "economy",
            "Format",
            "Team",
            "role",
        ]
        available = [col for col in feature_cols if col in all_players.columns]
        st.write(", ".join(available))

        if available:
            st.download_button(
                "Download Modeling Mart CSV",
                model_mart.to_csv(index=False).encode("utf-8"),
                file_name="analytics_modeling_mart.csv",
                mime="text/csv",
                width="stretch",
            )
