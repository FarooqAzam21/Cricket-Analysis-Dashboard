import pandas as pd

def safe_col(df, col, default=0):
    """Return df[col] if exists else a Series of default values."""
    if col in df.columns:
        return df[col]
    return pd.Series([default]*len(df), index=df.index)

def ensure_numeric(df, cols):
    """Ensure specified columns are numeric."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = 0
    return df

def sort_players(df, top_n=10, by=None, ascending=False):
    """
    Sort DataFrame based on a specific metric or default role-based logic.
    Returns top N players.
    """
    if df.empty:
        return df
    
    if by and by in df.columns:
        return df.sort_values(by=by, ascending=ascending).head(top_n)

    role_col = df.get('role', pd.Series([""]*len(df)))
    role_col = role_col.astype(str).str.lower()

    # Default logic (mostly used for general selection/team building)
    if role_col.str.contains('batsman|wicket', na=False).any():
        sort_df = df.sort_values(
            by=['average', 'strike_rate', 'runs'], 
            ascending=[False, False, False], 
            kind="stable", 
            ignore_index=True
        )
    elif role_col.str.contains('all-rounder|bowler|spinner', na=False).any():
        sort_df = df.sort_values(
            by=['wickets', 'bowling_average'], 
            ascending=[False, True], # More wickets is good, lower bowling avg is good
            kind="stable", 
            ignore_index=True
        )
    else:
        sort_df = df.sort_values(by='average', ascending=False, kind="stable", ignore_index=True)

    return sort_df.head(top_n)
