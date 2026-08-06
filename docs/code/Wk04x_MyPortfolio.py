# Full copy of the original script so it can be served as a static file from `docs/`.
# The working copy lives at the repository root; this file is an identical copy for Pages.

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "matplotlib>=3.10.0",
#     "mplsoccer>=1.4.0",
#     "numpy>=2.3.0",
#     "pandas>=2.3.3",
#     "plotly>=6.5.1",
#     "pyarrow>=22.0.0",
#     "pyzmq>=27.1.0",
#     "yfinance>=0.2.0",
#     "scipy>=1.11.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import textwrap as tw
    import numpy as np
    import yfinance as yf
    from scipy.optimize import minimize
    return mo, pd, px, go, tw, np, yf, minimize


@app.cell
def _(pd):

    csv_url = "https://gist.githubusercontent.com/DrAYim/80393243abdbb4bfe3b45fef58e8d3c8/raw/ed5cfd9f210bf80cb59a5f420bf8f2b88a9c2dcd/sp500_ZScore_AvgCostofDebt.csv"
    df_final = pd.read_csv(csv_url)
    df_final = df_final.dropna(subset=['AvgCost_of_Debt', 'Z_Score_lag', 'Sector_Key'])
    df_final = df_final[(df_final['AvgCost_of_Debt'] < 5)]
    df_final['Debt_Cost_Percent'] = df_final['AvgCost_of_Debt'] * 100
    df_final['Market_Cap_B'] = df_final['Market_Cap'] / 1e9

    all_tickers = df_final['Ticker'].dropna().unique().tolist()

    return (df_final, all_tickers)


@app.cell
def _(df_final, mo, tw):

    def md(s: str) -> str:
        return tw.dedent(s).strip()

    def hover_wrap(content_html: str) -> str:
        return md(f"""
        <div
            data-hover-card="1"
            style="
                margin: 0.3rem 0;
                padding: 0.65rem 0.8rem;
                border: 1px solid rgba(84, 123, 177, 0.38);
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.90);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            "
        >
            {content_html}
        </div>
        """)

    top_companies = df_final.nlargest(100, 'Market_Cap')
    default_sectors = top_companies['Sector_Key'].unique().tolist()

    sector_options = sorted(df_final['Sector_Key'].unique().tolist())
    sector_selector = mo.ui.multiselect(
        options=sector_options,
        value=default_sectors,  # Start with sectors from top 100
        label="**Select Sectors to Analyze:**"
    )

    min_z_db = df_final['Z_Score_lag'].min()
    max_z_db = df_final['Z_Score_lag'].max()
    z_slider = mo.ui.range_slider(
        start=min_z_db,
        stop=max_z_db,
        value=(min_z_db, max_z_db),
        step=0.1,
        label="**Z-Score (lag) range**"
    )

    max_cap_slider = int(df_final['Market_Cap_B'].max())
    cap_slider_db = mo.ui.slider(
        start=0,
        stop=max_cap_slider,
        step=10,
        value=50,  # Start at $50B minimum to reduce initial load
        label="**Min Market Cap ($B)** - Start higher for faster loading"
    )

    stock_search = mo.ui.text(
        label="**Search Company (Ticker or Name)** - Leave blank to see all filtered results",
        value=""
    )

    # Wrapped versions with hover effects
    animated_sector_selector = mo.Html(hover_wrap(mo.as_html(sector_selector)))
    animated_z_slider = mo.Html(hover_wrap(mo.as_html(z_slider)))
    animated_cap_slider = mo.Html(hover_wrap(mo.as_html(cap_slider_db)))
    animated_stock_search = mo.Html(hover_wrap(mo.as_html(stock_search)))

    return (sector_selector, z_slider, cap_slider_db, stock_search,
            animated_sector_selector, animated_z_slider, animated_cap_slider,
            animated_stock_search, md, hover_wrap)

# (full file content continues identically to repository root file)
