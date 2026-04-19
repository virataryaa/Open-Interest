import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import date

st.set_page_config(page_title="OI Monitor", layout="wide")

st.title("Open Interest Monitor — Softs")

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

CONTRACT_COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24


@st.cache_data(ttl=3600)
def load(commodity: str) -> pd.DataFrame:
    path = DATA_DIR / f"{commodity}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["settlement"] = pd.to_numeric(df["settlement"], errors="coerce")
    df["LTD"] = pd.to_datetime(df["LTD"], errors="coerce")
    df["FND"] = pd.to_datetime(df["FND"], errors="coerce")
    return df


def contract_label(ric: str) -> str:
    if len(ric) >= 2:
        mc = ric[-2]
        yr = ric[-1]
        return f"{MONTH_CODES.get(mc, mc)}-2{yr}0"
    return ric


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    commodity = st.selectbox(
        "Commodity", list(COMMODITIES.keys()),
        format_func=lambda x: f"{x} — {COMMODITIES[x]}"
    )
    lookback_years = st.slider("Lookback (years)", 1, 10, 5)
    roll_warn_days = st.slider("Roll watch window (days to FND/LTD)", 10, 90, 30)
    min_oi_pct = st.slider(
        "Min OI % of total for contract display", 0, 10, 1,
        help="Hide contracts below this % of total OI"
    )

df = load(commodity)
if df.empty:
    st.error(f"No data found for {commodity}")
    st.stop()

latest_date = df["Date"].max()
prev_date = df[df["Date"] < latest_date]["Date"].max()
cutoff = latest_date - pd.DateOffset(years=lookback_years)
df_window = df[df["Date"] >= cutoff].copy()
today = pd.Timestamp(date.today())
latest_df = df[df["Date"] == latest_date].copy()

st.caption(
    f"Latest data: **{latest_date.strftime('%d %b %Y')}** | "
    f"Showing last **{lookback_years}yr** | Commodity: **{commodity}**"
)

total_oi = (
    df_window.groupby("Date")["open_interest"]
    .sum().reset_index()
    .rename(columns={"open_interest": "total_oi"})
)


# ── Section 1: Total OI Historical View ──────────────────────────────────────
with st.expander("Section 1 — Total OI: Historical View", expanded=True):
    col1, col2, col3 = st.columns(3)

    row_latest = total_oi[total_oi["Date"] == latest_date]
    row_prev = total_oi[total_oi["Date"] == prev_date]
    latest_total = int(row_latest["total_oi"].values[0]) if not row_latest.empty else 0
    prev_total = int(row_prev["total_oi"].values[0]) if not row_prev.empty else 0
    delta_total = latest_total - prev_total

    full_total_oi = df.groupby("Date")["open_interest"].sum()
    pct_rank = int((full_total_oi < latest_total).mean() * 100)
    avg_1y = int(total_oi[total_oi["Date"] >= latest_date - pd.DateOffset(years=1)]["total_oi"].mean())
    avg_3y = int(total_oi[total_oi["Date"] >= latest_date - pd.DateOffset(years=3)]["total_oi"].mean())

    col1.metric("Current Total OI", f"{latest_total:,.0f}", f"{delta_total:+,.0f} vs prev day")
    col2.metric("Percentile (full history)", f"{pct_rank}th")
    col3.metric("1Y avg / 3Y avg", f"{avg_1y:,.0f} / {avg_3y:,.0f}")

    full_total_df = full_total_oi.reset_index().rename(columns={"open_interest": "total_oi"})
    full_total_df["doy"] = full_total_df["Date"].dt.dayofyear
    bands = full_total_df.groupby("doy")["total_oi"].quantile([0.1, 0.25, 0.75, 0.9]).unstack()
    bands.columns = ["p10", "p25", "p75", "p90"]

    plot_df = total_oi.copy()
    plot_df["doy"] = plot_df["Date"].dt.dayofyear
    plot_df = plot_df.merge(bands, on="doy", how="left")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["p90"],
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig1.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["p10"],
                              name="P10–P90", fill="tonexty",
                              fillcolor="rgba(100,149,237,0.15)", line=dict(width=0), hoverinfo="skip"))
    fig1.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["p75"],
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig1.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["p25"],
                              name="P25–P75", fill="tonexty",
                              fillcolor="rgba(100,149,237,0.28)", line=dict(width=0), hoverinfo="skip"))
    fig1.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["total_oi"],
                              name="Total OI", line=dict(color="#1f77b4", width=2)))
    fig1.add_hline(y=avg_1y, line_dash="dot", line_color="orange",
                   annotation_text="1Y avg", annotation_position="right")
    fig1.add_hline(y=avg_3y, line_dash="dot", line_color="limegreen",
                   annotation_text="3Y avg", annotation_position="right")
    fig1.update_layout(
        title=f"{commodity} — Total OI (all contracts aggregated)",
        xaxis_title="Date", yaxis_title="Open Interest",
        legend=dict(orientation="h"), hovermode="x unified",
        height=420, margin=dict(r=80)
    )
    st.plotly_chart(fig1, use_container_width=True)


# ── Section 2: OI by Contract ─────────────────────────────────────────────────
with st.expander("Section 2 — OI by Contract: Curve Breakdown", expanded=True):
    total_latest = latest_df["open_interest"].sum()
    if total_latest > 0:
        latest_df["oi_pct"] = latest_df["open_interest"] / total_latest * 100
        active_contracts = latest_df[latest_df["oi_pct"] >= min_oi_pct]["RIC"].tolist()
    else:
        active_contracts = latest_df["RIC"].tolist()

    contract_ts = df_window[df_window["RIC"].isin(active_contracts)].copy()
    contract_pivot = contract_ts.pivot_table(
        index="Date", columns="RIC", values="open_interest", aggfunc="sum"
    ).fillna(0)

    col_order = (
        latest_df[latest_df["RIC"].isin(active_contracts)]
        .sort_values("open_interest", ascending=False)["RIC"].tolist()
    )
    col_order = [c for c in col_order if c in contract_pivot.columns]
    contract_pivot = contract_pivot[col_order]

    fig2 = go.Figure()
    for i, ric in enumerate(col_order):
        color = CONTRACT_COLORS[i % len(CONTRACT_COLORS)]
        fig2.add_trace(go.Scatter(
            x=contract_pivot.index, y=contract_pivot[ric],
            name=ric, stackgroup="one",
            line=dict(width=0.5, color=color), fillcolor=color,
            hovertemplate=f"{ric}: %{{y:,.0f}}<extra></extra>"
        ))
    fig2.update_layout(
        title=f"{commodity} — OI by Contract (stacked, ≥{min_oi_pct}% of total)",
        xaxis_title="Date", yaxis_title="Open Interest",
        legend=dict(orientation="h", y=-0.2),
        height=450, hovermode="x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)

    snap = latest_df[latest_df["RIC"].isin(col_order)].sort_values("open_interest", ascending=False).copy()
    snap["label"] = snap["RIC"].apply(contract_label)
    fig2b = go.Figure(go.Bar(
        x=snap["label"], y=snap["open_interest"],
        text=snap["open_interest"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
        marker_color=[CONTRACT_COLORS[i % len(CONTRACT_COLORS)] for i in range(len(snap))]
    ))
    fig2b.update_layout(
        title=f"{commodity} — OI Snapshot as of {latest_date.strftime('%d %b %Y')}",
        xaxis_title="Contract", yaxis_title="Open Interest", height=350
    )
    st.plotly_chart(fig2b, use_container_width=True)


# ── Section 3: Daily OI Change Monitor ───────────────────────────────────────
with st.expander("Section 3 — Daily OI Change Monitor", expanded=True):
    latest_d = df[df["Date"] == latest_date][["RIC", "open_interest", "volume", "settlement"]].copy()
    prev_d = df[df["Date"] == prev_date][["RIC", "open_interest"]].rename(
        columns={"open_interest": "oi_prev"})
    chg = latest_d.merge(prev_d, on="RIC", how="left")
    chg["delta_oi"] = chg["open_interest"] - chg["oi_prev"]
    chg["delta_oi_pct"] = (chg["delta_oi"] / chg["oi_prev"].replace(0, np.nan) * 100).round(2)
    chg["oi_vol_ratio"] = (chg["open_interest"] / chg["volume"].replace(0, np.nan)).round(1)
    chg = chg[chg["open_interest"] > 0].sort_values("delta_oi", key=abs, ascending=False)
    chg["contract"] = chg["RIC"].apply(contract_label)

    display = chg[["contract", "RIC", "open_interest", "oi_prev", "delta_oi",
                    "delta_oi_pct", "volume", "oi_vol_ratio", "settlement"]].copy()
    display.columns = ["Contract", "RIC", "OI", "Prev OI", "D OI", "D OI %",
                       "Volume", "OI/Vol", "Settlement"]

    def color_delta(val):
        if pd.isna(val) or val == 0:
            return "color: gray"
        return "color: #2ecc71; font-weight:bold" if val > 0 else "color: #e74c3c; font-weight:bold"

    styled = (
        display.style
        .applymap(color_delta, subset=["D OI", "D OI %"])
        .format({"OI": "{:,.0f}", "Prev OI": "{:,.0f}", "D OI": "{:+,.0f}",
                 "D OI %": "{:+.2f}%", "Volume": "{:,.0f}", "OI/Vol": "{:.1f}",
                 "Settlement": "{:.2f}"}, na_rep="—")
    )
    st.caption(f"Sorted by |D OI| — {latest_date.strftime('%d %b')} vs {prev_date.strftime('%d %b')}")
    st.dataframe(styled, use_container_width=True, height=420)

    top_g = chg.nlargest(6, "delta_oi")
    top_l = chg.nsmallest(6, "delta_oi")
    top = pd.concat([top_g, top_l]).drop_duplicates("RIC").sort_values("delta_oi")
    colors_bar = ["#e74c3c" if v < 0 else "#2ecc71" for v in top["delta_oi"]]
    fig3 = go.Figure(go.Bar(
        x=top["RIC"], y=top["delta_oi"],
        marker_color=colors_bar,
        text=top["delta_oi"].apply(lambda x: f"{x:+,.0f}"),
        textposition="outside"
    ))
    fig3.update_layout(
        title=f"{commodity} — Top OI Movers ({latest_date.strftime('%d %b')} vs {prev_date.strftime('%d %b')})",
        xaxis_title="Contract", yaxis_title="D OI", height=380
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Section 4: OI Progression vs Historical Bands ────────────────────────────
with st.expander("Section 4 — OI Progression: Current Contract vs History", expanded=True):
    st.caption(
        "X-axis = days until LTD. Each historical contract of the same month code is aligned "
        "on the same axis, then Min / Q1 / Median / Q3 / Max are computed across those years."
    )

    active_rics = (
        df[df["Date"] == latest_date]
        .query("open_interest > 0")
        .sort_values("open_interest", ascending=False)["RIC"].tolist()
    )

    sel_contract = st.selectbox(
        "Select contract", active_rics,
        format_func=lambda r: f"{r}  ({contract_label(r)})",
        key="prog_contract"
    )

    if sel_contract and len(sel_contract) >= 4:
        base = sel_contract[:-2]          # e.g. "KC"
        month_code = sel_contract[-2]     # e.g. "K"

        # All historical contracts with same base + month code
        hist_rics = df[
            df["base_ric"].str.startswith(base) &
            df["base_ric"].str[-2].eq(month_code) &
            ~df["base_ric"].eq(sel_contract)
        ]["base_ric"].unique()

        def days_to_expiry_series(ric_df):
            ltd = ric_df["LTD"].dropna().iloc[0] if not ric_df["LTD"].dropna().empty else None
            if ltd is None:
                return None
            out = ric_df[["Date", "open_interest"]].copy()
            out["days_exp"] = (ltd - out["Date"]).dt.days
            return out[out["days_exp"] >= 0].set_index("days_exp")["open_interest"]

        # Build matrix: rows = days_exp, cols = historical years
        hist_matrix = {}
        for ric in hist_rics:
            s = days_to_expiry_series(df[df["base_ric"] == ric])
            if s is not None and len(s) > 10:
                hist_matrix[ric] = s

        if not hist_matrix:
            st.warning("Not enough historical contracts with the same month code to build bands.")
        else:
            hist_df = pd.DataFrame(hist_matrix)
            hist_df = hist_df.sort_index(ascending=False)  # high days first

            # Statistics across historical years at each days_exp
            stats = pd.DataFrame({
                "min":    hist_df.min(axis=1),
                "q1":     hist_df.quantile(0.25, axis=1),
                "median": hist_df.median(axis=1),
                "q3":     hist_df.quantile(0.75, axis=1),
                "max":    hist_df.max(axis=1),
            })

            # Current contract aligned to days_exp
            cur_s = days_to_expiry_series(df[df["base_ric"] == sel_contract])

            days_axis = stats.index  # sorted descending

            fig4 = go.Figure()

            # Max band (outer fill)
            fig4.add_trace(go.Scatter(
                x=days_axis, y=stats["max"],
                mode="lines", name="Max",
                line=dict(color="rgba(144,238,144,0.6)", width=1.5),
            ))
            # Min band (fill to max)
            fig4.add_trace(go.Scatter(
                x=days_axis, y=stats["min"],
                mode="lines", name="Min",
                fill="tonexty", fillcolor="rgba(144,238,144,0.12)",
                line=dict(color="rgba(210,180,140,0.7)", width=1.5),
            ))
            # Q3
            fig4.add_trace(go.Scatter(
                x=days_axis, y=stats["q3"],
                mode="lines", name="Upper Quartile",
                line=dict(color="rgba(180,180,180,0.9)", width=1.5),
            ))
            # Q1 (fill to Q3)
            fig4.add_trace(go.Scatter(
                x=days_axis, y=stats["q1"],
                mode="lines", name="Lower Quartile",
                fill="tonexty", fillcolor="rgba(180,180,180,0.25)",
                line=dict(color="rgba(180,180,180,0.9)", width=1.5),
            ))
            # Median
            fig4.add_trace(go.Scatter(
                x=days_axis, y=stats["median"],
                mode="lines", name="Median",
                line=dict(color="white", width=2, dash="dot"),
            ))
            # Current contract
            if cur_s is not None:
                cur_s = cur_s.sort_index(ascending=False)
                fig4.add_trace(go.Scatter(
                    x=cur_s.index, y=cur_s.values,
                    mode="lines", name=sel_contract,
                    line=dict(color="black", width=2.5),
                ))

            n_hist = len(hist_matrix)
            fig4.update_layout(
                title=f"OI Progression of: {sel_contract}  ({n_hist} historical contracts)",
                xaxis=dict(
                    title="Days Until Expiry",
                    autorange="reversed",
                    tickmode="auto", nticks=20,
                ),
                yaxis_title="Open Interest",
                legend=dict(orientation="h", y=-0.18),
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig4, use_container_width=True)

            with st.expander("Historical contracts used", expanded=False):
                st.write(sorted(hist_matrix.keys()))


# ── Section 5: Roll Watch ─────────────────────────────────────────────────────
with st.expander(f"Section 5 — Roll Watch (within {roll_warn_days} days of FND/LTD)", expanded=True):
    roll_df = df[df["Date"] == latest_date][
        ["RIC", "open_interest", "volume", "settlement", "FND", "LTD"]].copy()
    roll_df = roll_df[roll_df["open_interest"] > 0].copy()
    roll_df["days_to_FND"] = (roll_df["FND"] - today).dt.days
    roll_df["days_to_LTD"] = (roll_df["LTD"] - today).dt.days
    roll_df["contract"] = roll_df["RIC"].apply(contract_label)

    watch = roll_df[
        ((roll_df["days_to_FND"] >= 0) & (roll_df["days_to_FND"] <= roll_warn_days)) |
        ((roll_df["days_to_LTD"] >= 0) & (roll_df["days_to_LTD"] <= roll_warn_days))
    ].sort_values("days_to_FND")

    if watch.empty:
        st.success(f"No contracts within {roll_warn_days} days of FND or LTD.")
    else:
        pct_total = watch["open_interest"].sum() / roll_df["open_interest"].sum() * 100
        st.warning(
            f"**{len(watch)} contract(s)** near expiry — "
            f"**{watch['open_interest'].sum():,.0f} lots** ({pct_total:.1f}% of total OI)"
        )
        disp = watch[["contract", "RIC", "open_interest", "volume", "settlement",
                       "FND", "days_to_FND", "LTD", "days_to_LTD"]].copy()
        disp.columns = ["Contract", "RIC", "OI", "Volume", "Settlement",
                        "FND", "Days to FND", "LTD", "Days to LTD"]
        disp["FND"] = disp["FND"].dt.strftime("%d %b %Y")
        disp["LTD"] = disp["LTD"].dt.strftime("%d %b %Y")

        def style_urgency(row):
            d = min(
                (v for v in [row["Days to FND"], row["Days to LTD"]] if pd.notna(v) and v >= 0),
                default=999
            )
            if d <= 7:
                return ["background-color: #c0392b; color: white"] * len(row)
            if d <= 14:
                return ["background-color: #e67e22; color: white"] * len(row)
            return ["background-color: #f39c12; color: black"] * len(row)

        styled_roll = (
            disp.style.apply(style_urgency, axis=1)
            .format({"OI": "{:,.0f}", "Volume": "{:,.0f}", "Settlement": "{:.2f}",
                     "Days to FND": "{:.0f}", "Days to LTD": "{:.0f}"}, na_rep="—")
        )
        st.dataframe(styled_roll, use_container_width=True)


# ── Section 6: OI / Volume Ratio ─────────────────────────────────────────────
with st.expander("Section 6 — OI / Volume Ratio (Commitment Proxy)", expanded=False):
    st.caption(
        "High OI/Vol = positions held, low turnover.  "
        "Low OI/Vol = heavy activity relative to open positions."
    )
    oi_vol = df[df["Date"] == latest_date][["RIC", "open_interest", "volume"]].copy()
    oi_vol = oi_vol[(oi_vol["open_interest"] > 0) & (oi_vol["volume"] > 0)].copy()
    oi_vol["oi_vol"] = (oi_vol["open_interest"] / oi_vol["volume"]).round(1)
    oi_vol["contract"] = oi_vol["RIC"].apply(contract_label)
    oi_vol = oi_vol.sort_values("oi_vol", ascending=False)

    avg_ratio = oi_vol["oi_vol"].mean()
    colors_ov = ["#e74c3c" if v > avg_ratio * 1.5 else "#3498db" for v in oi_vol["oi_vol"]]

    fig5 = go.Figure(go.Bar(
        x=oi_vol["contract"], y=oi_vol["oi_vol"],
        marker_color=colors_ov,
        text=oi_vol["oi_vol"].apply(lambda x: f"{x:.1f}x"),
        textposition="outside"
    ))
    fig5.add_hline(y=avg_ratio, line_dash="dot", line_color="white",
                   annotation_text=f"avg {avg_ratio:.1f}x", annotation_position="right")
    fig5.update_layout(
        title=f"{commodity} — OI/Volume by Contract ({latest_date.strftime('%d %b %Y')})",
        xaxis_title="Contract", yaxis_title="OI / Volume",
        height=380, margin=dict(r=80)
    )
    st.plotly_chart(fig5, use_container_width=True)

    front_ric = latest_df.sort_values("open_interest", ascending=False).iloc[0]["RIC"] if not latest_df.empty else None
    if front_ric:
        front_ts = df_window[df_window["RIC"] == front_ric].copy()
        front_ts = front_ts[front_ts["volume"] > 0].copy()
        front_ts["oi_vol"] = front_ts["open_interest"] / front_ts["volume"]
        front_ts["ma20"] = front_ts["oi_vol"].rolling(20).mean()
        fig5b = go.Figure()
        fig5b.add_trace(go.Scatter(x=front_ts["Date"], y=front_ts["oi_vol"],
                                   name="Daily", line=dict(color="#3498db", width=1), opacity=0.5))
        fig5b.add_trace(go.Scatter(x=front_ts["Date"], y=front_ts["ma20"],
                                   name="20D MA", line=dict(color="orange", width=2)))
        fig5b.update_layout(
            title=f"{front_ric} — OI/Volume Ratio (historical)",
            xaxis_title="Date", yaxis_title="OI / Volume",
            height=350, hovermode="x unified"
        )
        st.plotly_chart(fig5b, use_container_width=True)


# ── Section 7: Cross-Commodity OI Snapshot ───────────────────────────────────
with st.expander("Section 7 — Cross-Commodity OI Snapshot", expanded=False):
    st.caption("Total OI and day-on-day change across all loaded commodities.")
    rows = []
    for sym in COMMODITIES:
        d = load(sym)
        if d.empty:
            continue
        ld = d["Date"].max()
        pd_ = d[d["Date"] < ld]["Date"].max()
        cur = int(d[d["Date"] == ld]["open_interest"].sum())
        prv = int(d[d["Date"] == pd_]["open_interest"].sum()) if not pd.isna(pd_) else 0
        dlt = cur - prv
        full_s = d.groupby("Date")["open_interest"].sum()
        pct_r = int((full_s < cur).mean() * 100)
        rows.append({
            "Commodity": sym, "Name": COMMODITIES[sym],
            "Latest Date": ld.strftime("%d %b %Y"),
            "Total OI": cur, "D OI": dlt,
            "D OI %": round(dlt / prv * 100, 2) if prv else 0,
            "Hist Percentile": pct_r,
        })

    cross_df = pd.DataFrame(rows)

    def color_cross(val):
        if pd.isna(val) or val == 0:
            return "color: gray"
        return "color: #2ecc71; font-weight:bold" if val > 0 else "color: #e74c3c; font-weight:bold"

    styled_cross = (
        cross_df.style
        .applymap(color_cross, subset=["D OI", "D OI %"])
        .format({"Total OI": "{:,.0f}", "D OI": "{:+,.0f}",
                 "D OI %": "{:+.2f}%", "Hist Percentile": "{}th"})
    )
    st.dataframe(styled_cross, use_container_width=True)

    fig7 = go.Figure(go.Bar(
        x=cross_df["Commodity"],
        y=cross_df["Hist Percentile"],
        marker_color=[
            "#e74c3c" if v < 20 else "#f39c12" if v < 40 else
            "#2ecc71" if v > 60 else "#3498db"
            for v in cross_df["Hist Percentile"]
        ],
        text=cross_df["Hist Percentile"].apply(lambda x: f"{x}th"),
        textposition="outside"
    ))
    fig7.add_hline(y=50, line_dash="dot", line_color="white", annotation_text="median")
    fig7.update_layout(
        title="Historical OI Percentile by Commodity (full history)",
        xaxis_title="Commodity", yaxis_title="Percentile",
        yaxis_range=[0, 110], height=380
    )
    st.plotly_chart(fig7, use_container_width=True)
