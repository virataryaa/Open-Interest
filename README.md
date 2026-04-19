# Open Interest Monitor — Softs

Streamlit dashboard for monitoring open interest across soft commodity futures (KC, CC, CT, SB, OJ, RC, LCC, LSU).

## Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Total OI Historical** | Aggregate OI with seasonal percentile bands (P10–P90), 1Y/3Y averages, current percentile rank |
| 2 | **OI by Contract** | Stacked area showing roll migration across the curve + snapshot bar chart |
| 3 | **Daily OI Change Monitor** | Δ OI table (green/red) sorted by absolute change + top movers bar chart |
| 4 | **Roll Watch** | Contracts within N days of FND/LTD, colour-coded by urgency |
| 5 | **OI / Volume Ratio** | Commitment proxy by contract + rolling front-month ratio |
| 6 | **Cross-Commodity Snapshot** | Side-by-side OI and historical percentile for all 8 commodities |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
