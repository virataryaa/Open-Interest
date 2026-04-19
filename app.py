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
    if len(ric) >= 2:
        mc, yr = ric[-2], ric[-1]
        return f"{MONTH_CODES.get(mc, mc)}-2{yr}0"
    return ric


def days_to_expiry_series(ric_df: pd.DataFrame):
    ltd = ric_df["LTD"].dropna()
    if ltd.empty:
        return None
    ltd = ltd.iloc[0]
    out = ric_df[["Date", "open_interest"]].copy()
    out["days_exp"] = (ltd - out["Date"]).dt.days
    return out[out["days_exp"] >= 0].set_index("days_exp")["open_interest"]


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

base = sel_contract[:-2]
month_code = sel_contract[-2]

hist_rics = df[
    df["base_ric"].str.startswith(base) &
    df["base_ric"].str[-2].eq(month_code) &
    ~df["base_ric"].eq(sel_contract)
]["base_ric"].unique()

hist_matrix = {}
for ric in hist_rics:
    s = days_to_expiry_series(df[df["base_ric"] == ric])
    if s is not None and len(s) > 10:
        hist_matrix[ric] = s

with col_info:
    st.caption(
        f"Latest data: **{latest_date.strftime('%d %b %Y')}** | "
        f"Historical contracts used: **{len(hist_matrix)}** "
        f"({', '.join(sorted(hist_matrix.keys()))})"
    )

if not hist_matrix:
    st.warning("Not enough historical contracts with the same month code to build bands.")
    st.stop()

hist_df = pd.DataFrame(hist_matrix).sort_index(ascending=False)

stats = pd.DataFrame({
    "min":    hist_df.min(axis=1),
    "q1":     hist_df.quantile(0.25, axis=1),
    "median": hist_df.median(axis=1),
    "q3":     hist_df.quantile(0.75, axis=1),
    "max":    hist_df.max(axis=1),
})

cur_s = days_to_expiry_series(df[df["base_ric"] == sel_contract])
days_axis = stats.index

fig = go.Figure()

# Outer band: Min to Max
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["max"],
    mode="lines", name="Max",
    line=dict(color="rgba(144,238,144,0.7)", width=1.5),
))
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["min"],
    mode="lines", name="Min",
    fill="tonexty", fillcolor="rgba(144,238,144,0.10)",
    line=dict(color="rgba(210,180,140,0.7)", width=1.5),
))

# IQR band: Q1 to Q3
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["q3"],
    mode="lines", name="Upper Quartile",
    line=dict(color="rgba(180,180,180,0.9)", width=1.5),
))
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["q1"],
    mode="lines", name="Lower Quartile",
    fill="tonexty", fillcolor="rgba(180,180,180,0.22)",
    line=dict(color="rgba(180,180,180,0.9)", width=1.5),
))

# Median
fig.add_trace(go.Scatter(
    x=days_axis, y=stats["median"],
    mode="lines", name="Median",
    line=dict(color="rgba(255,255,255,0.85)", width=2, dash="dot"),
))

# Current contract
if cur_s is not None:
    cur_s = cur_s.sort_index(ascending=False)
    fig.add_trace(go.Scatter(
        x=cur_s.index, y=cur_s.values,
        mode="lines", name=sel_contract,
        line=dict(color="black", width=2.5),
    ))

fig.update_layout(
    title=dict(
        text=f"OI Progression of : {sel_contract}",
        x=0.5, xanchor="center",
        font=dict(size=18),
    ),
    xaxis=dict(
        title="Days Until Expiry",
        autorange="reversed",
        nticks=25,
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
    ),
    yaxis=dict(
        title="Open Interest",
        showgrid=True, gridcolor="rgba(255,255,255,0.08)",
        tickformat=",",
    ),
    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    hovermode="x unified",
    height=580,
    plot_bgcolor="rgba(30,34,45,1)",
    paper_bgcolor="rgba(30,34,45,1)",
    font=dict(color="white"),
)

st.plotly_chart(fig, use_container_width=True)
