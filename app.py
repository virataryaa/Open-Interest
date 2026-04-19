import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import date

st.set_page_config(page_title="OI Progression Monitor", layout="wide")

st.title("OI Progression Monitor — Softs")

DATA_DIR = Path(__file__).parent / "data"

COMMODITIES = {
    "KC":  "Coffee Arabica (ICE)",
    "CC":  "Cocoa (ICE)",
    "CT":  "Cotton (ICE)",
    "SB":  "Sugar #11 (ICE)",
    "OJ":  "OJ (ICE)",
    "RC":  "Robusta Coffee (LIFFE)",
    "LCC": "Cocoa (LIFFE)",
    "LSU": "Sugar #5 (LIFFE)",
}

MONTH_CODES = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr",
    "K": "May", "M": "Jun", "N": "Jul", "Q": "Aug",
    "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec",
}


@st.cache_data(ttl=3600)
def load(commodity: str) -> pd.DataFrame:
    path = DATA_DIR / f"{commodity}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
    df["LTD"] = pd.to_datetime(df["LTD"], errors="coerce")
    df["FND"] = pd.to_datetime(df["FND"], errors="coerce")
    return df


def contract_label(ric: str) -> str:
    if len(ric) >= 4:
        mc = ric[-2]
        yr = ric[-1]
        return f"{MONTH_CODES.get(mc, mc)}-{yr}"
    return ric


def smooth_series(ric_df: pd.DataFrame, ltd: pd.Timestamp) -> pd.Series | None:
    """
    Return a Series indexed by integer days-until-LTD, filled across weekends/holidays,
    but only within the range of actual observations (never extrapolated forward).
    """
    out = ric_df[["Date", "open_interest"]].copy()
    out["days_exp"] = (ltd - out["Date"]).dt.days
    out = out[out["days_exp"] >= 0]
    if out.empty:
        return None
    out = out.sort_values("Date").drop_duplicates("days_exp", keep="last")
    s = out.set_index("days_exp")["open_interest"].astype(float)

    # Fill only within the actual data window — never beyond the last real observation
    min_days = int(s.index.min())
    max_days = int(s.index.max())
    full_range = np.arange(max_days, min_days - 1, -1)
    s = s.reindex(full_range).sort_index(ascending=False).ffill().bfill()
    return s


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    commodity = st.selectbox(
        "Commodity", list(COMMODITIES.keys()),
        format_func=lambda x: f"{x} — {COMMODITIES[x]}"
    )

df = load(commodity)
if df.empty:
    st.error(f"No data found for {commodity}")
    st.stop()

latest_date = df["Date"].max()

active_rics = (
    df[df["Date"] == latest_date]
    .query("open_interest > 0")
    .sort_values("open_interest", ascending=False)["RIC"]
    .tolist()
)

col_sel, col_info = st.columns([2, 3])
with col_sel:
    sel_contract = st.selectbox(
        "Contract",
        active_rics,
        format_func=lambda r: f"{r}  ({contract_label(r)})",
    )

if not sel_contract or len(sel_contract) < 4:
    st.stop()

base       = sel_contract[:-2]   # e.g. "KC"
month_code = sel_contract[-2]    # e.g. "U"

# Get the exact LTD for the selected contract from the latest date row
sel_ltd_row = df[(df["base_ric"] == sel_contract) & (df["Date"] == latest_date)]
if sel_ltd_row.empty:
    st.error("Cannot find LTD for selected contract.")
    st.stop()
sel_ltd = sel_ltd_row["LTD"].iloc[0]

# Find historical contracts: same commodity prefix + same month code, different LTD
# Group by (base_ric, LTD) to correctly handle decade ambiguity (e.g. KCZ6 = 2016 and 2026)
contract_ltds = (
    df[df["base_ric"].str.startswith(base) & df["base_ric"].str[-2].eq(month_code)]
    .groupby("base_ric")["LTD"].first()
)
hist_contracts = contract_ltds[contract_ltds != sel_ltd]  # exclude current contract's LTD

hist_matrix: dict[str, pd.Series] = {}
for ric, ltd in hist_contracts.items():
    ric_data = df[(df["base_ric"] == ric) & (df["LTD"] == ltd)]
    s = smooth_series(ric_data, ltd)
    if s is not None and len(s) > 20:
        label_key = f"{ric}({ltd.year})"
        hist_matrix[label_key] = s

with col_info:
    st.caption(
        f"Latest data: **{latest_date.strftime('%d %b %Y')}** | "
        f"LTD: **{sel_ltd.strftime('%d %b %Y')}** | "
        f"Historical contracts: **{len(hist_matrix)}**"
    )

if not hist_matrix:
    st.warning("Not enough historical contracts with the same month code to build bands.")
    st.stop()

# Align all series on a common integer index, compute stats
hist_df = pd.DataFrame(hist_matrix)   # index = days_exp (may have NaN where contracts differ)
hist_df = hist_df.sort_index(ascending=False)

# Stats — skip NaN so short-lived contracts don't drag down counts at high day counts
stats = pd.DataFrame({
    "min":    hist_df.min(axis=1),
    "q1":     hist_df.quantile(0.25, axis=1),
    "median": hist_df.median(axis=1),
    "q3":     hist_df.quantile(0.75, axis=1),
    "max":    hist_df.max(axis=1),
}).dropna()

cur_s = smooth_series(
    df[(df["base_ric"] == sel_contract) & (df["LTD"] == sel_ltd)],
    sel_ltd
)
if cur_s is not None:
    cur_s = cur_s.sort_index(ascending=False)

days_axis = stats.index  # descending (700 → 0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Max line (top of outer band)
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["max"],
    mode="lines", name="Max",
    line=dict(color="rgba(100,200,100,0.8)", width=1.2),
))
# Min line — fills back up to Max creating the outer envelope
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["min"],
    mode="lines", name="Min",
    fill="tonexty", fillcolor="rgba(100,200,100,0.10)",
    line=dict(color="rgba(200,160,100,0.8)", width=1.2),
))

# Upper quartile
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["q3"],
    mode="lines", name="Upper Quartile",
    line=dict(color="rgba(150,150,150,0.9)", width=1.2),
))
# Lower quartile — fills back up to Q3
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["q1"],
    mode="lines", name="Lower Quartile",
    fill="tonexty", fillcolor="rgba(150,150,150,0.20)",
    line=dict(color="rgba(150,150,150,0.9)", width=1.2),
))

# Median
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["median"],
    mode="lines", name="Median",
    line=dict(color="#444444", width=2, dash="dot"),
))

# Current contract — black on top
if cur_s is not None:
    fig.add_trace(go.Scatter(
        x=cur_s.index, y=cur_s.values,
        mode="lines", name=sel_contract,
        line=dict(color="black", width=2.5),
    ))

fig.update_layout(
    title=dict(
        text=f"OI Progression of : {sel_contract}",
        x=0.5, xanchor="center",
        font=dict(size=17, color="#222222"),
    ),
    xaxis=dict(
        title="Days Until Expiry",
        autorange="reversed",
        nticks=25,
        showgrid=True, gridcolor="rgba(0,0,0,0.07)",
        linecolor="#cccccc", tickcolor="#888888",
    ),
    yaxis=dict(
        title="Open Interest",
        showgrid=True, gridcolor="rgba(0,0,0,0.07)",
        linecolor="#cccccc", tickcolor="#888888",
        tickformat=",",
    ),
    legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                font=dict(color="#333333")),
    hovermode="x unified",
    height=580,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#333333"),
    margin=dict(t=60, b=80),
)

st.plotly_chart(fig, use_container_width=True)
