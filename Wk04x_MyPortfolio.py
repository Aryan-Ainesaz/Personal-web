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


@app.cell
def _(df_final, sector_selector, z_slider, cap_slider_db, stock_search, mo, pd):
    # Dashboard filtering logic with proper error handling
    z_min_db, z_max_db = z_slider.value
    cap_min = cap_slider_db.value

    # Base filtering
    df_filtered = df_final[
        (df_final['Sector_Key'].isin(sector_selector.value)) &
        (df_final['Z_Score_lag'] >= z_min_db) &
        (df_final['Z_Score_lag'] <= z_max_db) &
        (df_final['Market_Cap_B'] >= cap_min)
    ]

    # Search filtering with error handling
    search_term = stock_search.value.strip().lower()
    search_applied = False
    search_result_message = ""

    if search_term:
        search_applied = True
        df_search = df_filtered[
            df_filtered['Ticker'].str.lower().eq(search_term) |
            df_filtered['Name'].str.lower().str.contains(search_term, na=False)
        ]

        if df_search.empty:
            search_result_message = f"⚠️ No companies found matching '{search_term}'. Showing all filtered results instead."
        else:
            search_result_message = f"✓ Found {len(df_search)} company(ies) matching '{search_term}'"
            df_filtered = df_search
    else:
        search_result_message = f"Showing {len(df_filtered)} companies based on current filters"

    distress_threshold = 1.81
    safe_threshold = 2.99
    risk_distress = int((df_filtered['Z_Score_lag'] < distress_threshold).sum())
    risk_intermediate = int(((df_filtered['Z_Score_lag'] >= distress_threshold) & (df_filtered['Z_Score_lag'] <= safe_threshold)).sum())
    risk_safe = int((df_filtered['Z_Score_lag'] > safe_threshold).sum())

    df_distress = df_filtered[df_filtered['Z_Score_lag'] < distress_threshold]
    df_intermediate = df_filtered[(df_filtered['Z_Score_lag'] >= distress_threshold) & (df_filtered['Z_Score_lag'] <= safe_threshold)]
    df_safe = df_filtered[df_filtered['Z_Score_lag'] > safe_threshold]

    # Top 5 companies by risk
    top5_safest = df_filtered.nlargest(5, 'Z_Score_lag')[['Name', 'Ticker', 'Z_Score_lag', 'Debt_Cost_Percent']] if len(df_filtered) > 0 else pd.DataFrame()
    top5_riskiest = df_filtered.nsmallest(5, 'Z_Score_lag')[['Name', 'Ticker', 'Z_Score_lag', 'Debt_Cost_Percent']] if len(df_filtered) > 0 else pd.DataFrame()
    top5_distress = df_distress.nsmallest(5, 'Z_Score_lag')[['Name', 'Ticker', 'Z_Score_lag', 'Debt_Cost_Percent']] if len(df_distress) > 0 else pd.DataFrame()
    top5_intermediate = df_intermediate.nsmallest(5, 'Z_Score_lag')[['Name', 'Ticker', 'Z_Score_lag', 'Debt_Cost_Percent']] if len(df_intermediate) > 0 else pd.DataFrame()
    top5_safe = df_safe.nlargest(5, 'Z_Score_lag')[['Name', 'Ticker', 'Z_Score_lag', 'Debt_Cost_Percent']] if len(df_safe) > 0 else pd.DataFrame()

    company_count = len(df_filtered)
    avg_cost = df_filtered['Debt_Cost_Percent'].mean() if len(df_filtered) > 0 else 0
    median_cost = df_filtered['Debt_Cost_Percent'].median() if len(df_filtered) > 0 else 0
    avg_zscore = df_filtered['Z_Score_lag'].mean() if len(df_filtered) > 0 else 0
    max_cap_filtered = df_filtered['Market_Cap_B'].max() if len(df_filtered) > 0 else 0

    if search_applied:
        if "No companies found" in search_result_message:
            search_message_display = mo.callout(mo.md(search_result_message), kind="warn")
        else:
            search_message_display = mo.callout(mo.md(search_result_message), kind="success")
    else:
        search_message_display = mo.callout(mo.md(search_result_message), kind="info")

    return (df_filtered, distress_threshold, safe_threshold, risk_distress,
            risk_intermediate, risk_safe, top5_safest, top5_riskiest,
            company_count, avg_cost, median_cost, avg_zscore, max_cap_filtered,
            top5_distress, top5_intermediate, top5_safe, search_message_display)


@app.cell
def _(df_filtered, px, tw, distress_threshold, safe_threshold, np, pd):

    def tidy_xy_figure(fig, title_left=False):
        fig.update_layout(
            font=dict(size=13),
            margin=dict(l=60, r=30, t=95, b=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hoverlabel=dict(bgcolor="white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            transition=dict(duration=320, easing="cubic-in-out"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", zeroline=False)
        if title_left:
            fig.update_layout(title=dict(y=0.96, x=0, xanchor='left'))
        return fig

    title = "Are higher Z-Scores last year associated with lower average costs of debt this year?"
    wrapped_title = "<br>".join(tw.wrap(title, width=50))

    if len(df_filtered) > 0:
        df_scatter = df_filtered.copy()
        positive_caps = df_scatter.loc[df_scatter['Market_Cap'] > 0, 'Market_Cap']
        min_positive_cap = float(positive_caps.min()) if not positive_caps.empty else 1.0
        df_scatter['Market_Cap_Render'] = df_scatter['Market_Cap'].clip(lower=min_positive_cap * 0.2)

        DB_fig = px.scatter(
            df_scatter,
            x='Z_Score_lag',
            y='Debt_Cost_Percent',
            range_x=[-5, 20],
            range_y=[-1, 15],
            color='Sector_Key',
            size='Market_Cap_Render',
            hover_name='Name',
            hover_data=['Ticker', 'Market_Cap'],
            title=wrapped_title,
            labels={'Z_Score_lag': 'Altman Z-Score (lagged)', 'Debt_Cost_Percent': 'Avg. Cost of Debt (%)'},
            template='presentation',
            width=900,
            height=600,
            opacity=0.7
        )
        DB_fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey'), sizemin=6))
        tidy_xy_figure(DB_fig, title_left=True)
        DB_fig.update_layout(
            margin=dict(l=60, r=240, t=95, b=60),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )
        DB_fig.add_vline(x=1.81, line_dash="dash", line_color="red",
            annotation=dict(text="Distress Threshold (Z-Score = 1.81)", font=dict(color="red"),
                            x=1.5, xref="x", y=1.07, yref="paper", showarrow=False, yanchor="top"))
        DB_fig.add_vline(x=2.99, line_dash="dash", line_color="green",
            annotation=dict(text="Safe Threshold (Z-Score = 2.99)", font=dict(color="green"),
                            x=3.10, xref="x", y=1.02, yref="paper", showarrow=False, yanchor="top"))

        df_regline = df_filtered[(df_filtered['Debt_Cost_Percent'] < 5)]
        if not df_regline.empty and len(df_regline) > 1:
            x = df_regline['Z_Score_lag'].astype(float)
            y = df_regline['Debt_Cost_Percent'].astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = intercept + slope * x_line
            line_trace = px.line(x=x_line, y=y_line).data[0]
            line_trace.update(line=dict(width=0.5, color='black'))
            DB_fig.add_trace(line_trace)
    else:
        DB_fig = px.scatter(title="No data matches your current filters. Try adjusting the filters above.")
        DB_fig.update_layout(height=600)

    if len(df_filtered) > 0:
        box_fig = px.box(
            df_filtered,
            x='Sector_Key',
            y='Debt_Cost_Percent',
            color='Sector_Key',
            title="<br>".join(tw.wrap("Cost of Debt Distribution by Sector", width=45)),
            labels={'Debt_Cost_Percent': 'Avg. Cost of Debt (%)', 'Sector_Key': 'Sector'},
            template='presentation',
            width=900,
            height=450
        )
        tidy_xy_figure(box_fig)
        box_fig.update_layout(
            xaxis_tickangle=-35,
            margin=dict(l=60, r=30, t=90, b=95),
            font=dict(size=12),
            xaxis=dict(tickfont=dict(size=10)),
            showlegend=False,
        )
    else:
        box_fig = px.box(title="No data available")
        box_fig.update_layout(height=450)

    if len(df_filtered) > 0:
        risk_plot_df = df_filtered.copy()
        risk_plot_df["Risk_Zone"] = np.select(
            [
                risk_plot_df["Z_Score_lag"] < distress_threshold,
                risk_plot_df["Z_Score_lag"] > safe_threshold,
            ],
            ["Distress", "Safe"],
            default="Intermediate",
        )

        risk_dist_fig = px.violin(
            risk_plot_df,
            x="Risk_Zone",
            y="Debt_Cost_Percent",
            color="Risk_Zone",
            box=True,
            points="outliers",
            category_orders={"Risk_Zone": ["Distress", "Intermediate", "Safe"]},
            color_discrete_map={"Distress": "red", "Intermediate": "gold", "Safe": "green"},
            title="<br>".join(tw.wrap("Debt Cost Distribution by Risk Zone", width=45)),
            labels={"Risk_Zone": "Risk Zone", "Debt_Cost_Percent": "Average Cost of Debt (%)"},
            template="presentation",
            height=450,
        )
        tidy_xy_figure(risk_dist_fig)
        risk_dist_fig.update_layout(
            autosize=True,
            margin=dict(l=60, r=30, t=90, b=55),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
            legend_title_text="Risk Zone",
        )
    else:
        risk_dist_fig = px.violin(title="No data available")
        risk_dist_fig.update_layout(height=450)

    heatmap_cols = ["Z_Score_lag", "Debt_Cost_Percent", "Market_Cap_B"]
    available_cols = [c for c in heatmap_cols if c in df_filtered.columns]

    if len(available_cols) >= 2 and not df_filtered.empty:
        corr_df = df_filtered[available_cols].corr(numeric_only=True)
        pretty_labels = {
            "Z_Score_lag": "Altman Z-Score (Lagged)",
            "Debt_Cost_Percent": "Average Cost of Debt (%)",
            "Market_Cap_B": "Market Capitalisation ($B)",
        }
        corr_df = corr_df.rename(index=pretty_labels, columns=pretty_labels)
        heatmap_fig = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="<br>".join(tw.wrap("Financial Signal Map: Correlations Across Key Metrics", width=45)),
            labels={"x": "Variables", "y": "Variables", "color": "Correlation"},
            height=450,
            template="presentation",
        )
        tidy_xy_figure(heatmap_fig)
        heatmap_fig.update_layout(
            autosize=True,
            title=dict(x=0.5, xanchor="center", y=0.97),
            title_font=dict(size=16),
            margin=dict(l=60, r=30, t=95, b=45),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
            coloraxis_colorbar=dict(len=0.8),
            showlegend=False,
        )
    else:
        heatmap_fig = px.imshow(
            [[1.0]],
            text_auto=".2f",
            title="<br>".join(tw.wrap("Financial Signal Map: Not Enough Data After Filters", width=45)),
            labels={"x": "Variables", "y": "Variables", "color": "Correlation"},
            height=450,
            template="presentation",
        )
        tidy_xy_figure(heatmap_fig)

    return (DB_fig, box_fig, risk_dist_fig, heatmap_fig, tidy_xy_figure)


@app.cell
def _(pd, px, tw, yf, mo):

    try:
        gspc = yf.Ticker("^GSPC").history(period="20y")
        gspc = gspc[["Close"]].reset_index()
        gspc.columns = ["Date", "Close"]
        gspc["Date"] = pd.to_datetime(gspc["Date"]).dt.tz_convert(None)
        gspc = gspc.dropna().sort_values("Date")
        one_year_cutoff = gspc["Date"].max() - pd.DateOffset(years=1)
        gspc_recent = gspc[gspc["Date"] >= one_year_cutoff].copy()
    except Exception:
        gspc = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "Close": pd.Series(dtype="float64")})
        gspc_recent = gspc.copy()

    trailing_pe = "27.5 – 29.2"
    forward_pe = "~21.2"
    div_yield = "~1.17%"

    def tidy_timeseries(fig):
        fig.update_layout(
            font=dict(size=13),
            margin=dict(l=60, r=25, t=85, b=55),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hoverlabel=dict(bgcolor="white"),
            dragmode="zoom",
            uirevision="sp-timeseries",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            transition=dict(duration=0),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", zeroline=False)
        return fig

    _fig_recent = px.line(
        gspc_recent,
        x="Date",
        y="Close",
        title="<br>".join(tw.wrap("S&P 500 Index Daily Closing Price (^GSPC, Last 1 Year)", width=50)),
        labels={"Close": "S&P 500 Index (Points)", "Date": "Date"},
        template="presentation",
        width=900,
        height=400,
    )
    tidy_timeseries(_fig_recent)
    _fig_recent.update_xaxes(rangeslider_visible=True)

    sp_fig_recent = mo.ui.plotly(_fig_recent)

    _fig_historic = px.line(
        gspc,
        x="Date",
        y="Close",
        title="S&P 500 Index Daily Closing Price (^GSPC, Last 20 Years)",
        labels={"Close": "S&P 500 Index (Points)", "Date": "Date"},
        template="presentation",
        width=900,
        height=400,
    )
    tidy_timeseries(_fig_historic)
    _fig_historic.update_xaxes(rangeslider_visible=True, fixedrange=True)
    _fig_historic.update_yaxes(fixedrange=False)

    sp_fig_historic = mo.ui.plotly(_fig_historic)

    return (sp_fig_recent, sp_fig_historic, trailing_pe, forward_pe, div_yield, tidy_timeseries)


@app.cell
def _(mo, top5_safest, top5_riskiest, top5_intermediate):

    intermediate_table = mo.ui.table(top5_intermediate, label="Select firms to highlight in chart") if len(top5_intermediate) > 0 else None
    distress_table = mo.ui.table(top5_riskiest, label="Select firms to highlight in chart") if len(top5_riskiest) > 0 else None
    safe_table = mo.ui.table(top5_safest, label="Select firms to highlight in chart") if len(top5_safest) > 0 else None

    return (intermediate_table, distress_table, safe_table)


@app.cell
def _(intermediate_table, distress_table, safe_table, top5_safest, top5_riskiest, top5_intermediate, pd, px, tw, yf):

    def fetch_returns(tickers, label):
        if not tickers:
            fig = px.line(title=f"No companies selected for {label}")
            fig.update_layout(dragmode="zoom", transition=dict(duration=0), height=450)
            return fig

        frames = []
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(period="1y")
                if hist.empty:
                    continue
                hist = hist[["Close"]].reset_index()
                hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_convert(None)
                hist = hist.sort_values("Date")
                hist["Cumulative_Return"] = hist["Close"] / hist["Close"].iloc[0]
                hist["Ticker"] = ticker
                frames.append(hist[["Date", "Cumulative_Return", "Ticker"]])
            except Exception:
                pass
        if frames:
            df_ret = pd.concat(frames, ignore_index=True)
            fig = px.line(
                df_ret, x="Date", y="Cumulative_Return", color="Ticker",
                title="<br>".join(tw.wrap(f"{label} — Cumulative Returns (Last 1 Year)", width=50)),
                labels={"Cumulative_Return": "Growth of $1 Invested", "Date": "Date"},
                template="presentation", width=900, height=450,
            )
            fig.update_layout(
                font=dict(size=13),
                margin=dict(l=60, r=25, t=85, b=55),
                paper_bgcolor="white",
                plot_bgcolor="white",
                dragmode="zoom",
                transition=dict(duration=0)
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", rangeslider_visible=True)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)")
            fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Breakeven")
        else:
            fig = px.line(title=f"No data available for {label}")
            fig.update_layout(dragmode="zoom", transition=dict(duration=0), height=450)
        return fig

    def _get_tickers(table_widget, fallback_df):
        if table_widget is None:
            return list(fallback_df["Ticker"]) if len(fallback_df) > 0 else []
        sel = table_widget.value
        if sel is not None and not sel.empty:
            return list(sel["Ticker"])
        return list(fallback_df["Ticker"]) if len(fallback_df) > 0 else []

    safe_returns_fig = fetch_returns(_get_tickers(safe_table, top5_safest), "Top 5 Safest Firms")
    distress_returns_fig = fetch_returns(_get_tickers(distress_table, top5_riskiest), "Top 5 Riskiest Firms")
    intermediate_returns_fig = fetch_returns(_get_tickers(intermediate_table, top5_intermediate), "Top 5 Intermediate Firms")

    return (safe_returns_fig, distress_returns_fig, intermediate_returns_fig, fetch_returns)


@app.cell
def _(all_tickers, mo):

    sorted_tickers = sorted(all_tickers)

    _preferred = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    _default_tickers = [t for t in _preferred if t in sorted_tickers]

    if not _default_tickers:
        _default_tickers = sorted_tickers[:5]

    ticker_selector = mo.ui.multiselect(
        options=sorted_tickers,
        value=_default_tickers,
        label="**Select Stocks for Portfolio (3-10 recommended):**"
    )

    start_date = mo.ui.date(
        value="2020-01-01",
        label="**Start Date:**"
    )

    end_date = mo.ui.date(
        value="2023-12-31",
        label="**End Date:**"
    )

    risk_free_rate = mo.ui.slider(
        start=0,
        stop=5,
        step=0.1,
        value=1.0,
        label="**Risk-Free Rate (%):**"
    )

    import time

    optimise_button = mo.ui.button(
        label=" Optimise Portfolio",
        on_click=lambda: time.time()
    )

    return (ticker_selector, start_date, end_date, risk_free_rate, optimise_button)


@app.cell
def _(ticker_selector, start_date, end_date, risk_free_rate, optimise_button, yf, np, minimize, pd, px, mo):

    def fetch_data(tickers, start, end):
        """Fetches historical stock closing prices from Yahoo Finance.
        Handles both old yfinance ('Adj Close') and new yfinance ('Close') column names.
        """
        try:
            if len(tickers) == 1:

                raw = yf.download(tickers[0], start=start, end=end, progress=False, auto_adjust=True)
                if raw.empty:
                    return pd.DataFrame()

                data = raw[['Close']].copy()
                data.columns = tickers
            else:
                raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
                if raw.empty:
                    return pd.DataFrame()

                if 'Close' in raw.columns:
                    data = raw['Close'].copy()
                elif ('Close', tickers[0]) in raw.columns:

                    data = raw['Close'].copy()
                else:
                    return pd.DataFrame()
                if isinstance(data, pd.Series):
                    data = data.to_frame()

            data = data.dropna(axis=1, how='all')
            return data
        except Exception as e:
            return pd.DataFrame()

    def calculate_portfolio_performance(weights, returns, cov_matrix, rf_rate):
        """Calculates portfolio annual return, volatility, and Sharpe Ratio."""
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - rf_rate) / portfolio_std_dev if portfolio_std_dev > 0 else 0
        return portfolio_return, portfolio_std_dev, sharpe_ratio

    def neg_sharpe_ratio(weights, returns, cov_matrix, rf_rate):
        """Returns the negative Sharpe ratio (for minimization)."""
        _, _, sharpe_ratio = calculate_portfolio_performance(weights, returns, cov_matrix, rf_rate)
        return -sharpe_ratio

    def portfolio_volatility(weights, returns, cov_matrix, rf_rate):
        """Returns the portfolio volatility (for minimization)."""
        _, portfolio_std_dev, _ = calculate_portfolio_performance(weights, returns, cov_matrix, rf_rate)
        return portfolio_std_dev

    optimisation_result = None
    optimisation_error = None

    # Compute when the button is clicked, and also when a valid default
    # selection is present so static HTML exports can include results.
    should_optimise = optimise_button.value is not None or len(ticker_selector.value) >= 2

    if should_optimise:
        tickers = list(ticker_selector.value)
        start = str(start_date.value)
        end = str(end_date.value)
        rf_rate = risk_free_rate.value / 100

        if len(tickers) < 2:
            optimisation_error = "Please select at least 2 stocks for optimisation."
        elif len(tickers) > 20:
            optimisation_error = "Please select no more than 20 stocks to ensure timely optimisation."
        else:
            try:
                data = fetch_data(tickers, start, end)

                if data.empty:
                    optimisation_error = "Unable to fetch data for selected stocks. Please check tickers and date range."
                else:
                    returns = data.pct_change().dropna()
                    cov_matrix = returns.cov()

                    num_assets = len(tickers)
                    initial_weights = np.array(num_assets * [1. / num_assets])

                    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0, 1) for _ in range(num_assets))

                    sharpe_results = minimize(
                        neg_sharpe_ratio, initial_weights,
                        args=(returns, cov_matrix, rf_rate),
                        method='SLSQP', bounds=bounds, constraints=constraints
                    )
                    max_sharpe_weights = sharpe_results.x
                    max_sharpe_return, max_sharpe_std_dev, max_sharpe_ratio = calculate_portfolio_performance(
                        max_sharpe_weights, returns, cov_matrix, rf_rate
                    )

                    min_vol_results = minimize(
                        portfolio_volatility, initial_weights,
                        args=(returns, cov_matrix, rf_rate),
                        method='SLSQP', bounds=bounds, constraints=constraints
                    )
                    min_vol_weights = min_vol_results.x
                    min_vol_return, min_vol_std_dev, min_vol_ratio = calculate_portfolio_performance(
                        min_vol_weights, returns, cov_matrix, rf_rate
                    )

                    sharpe_df = pd.DataFrame({
                        'Ticker': tickers,
                        'Weight (%)': max_sharpe_weights * 100
                    }).sort_values('Weight (%)', ascending=False)

                    minvol_df = pd.DataFrame({
                        'Ticker': tickers,
                        'Weight (%)': min_vol_weights * 100
                    }).sort_values('Weight (%)', ascending=False)

                    sharpe_chart = px.bar(
                        sharpe_df,
                        x='Ticker',
                        y='Weight (%)',
                        title=f" Return: {max_sharpe_return:.2%} | Volatility: {max_sharpe_std_dev:.2%} | Sharpe: {max_sharpe_ratio:.2f}",
                        labels={'Weight (%)': 'Allocation (%)'},
                        color='Weight (%)',
                        color_continuous_scale='Blues',
                        template='presentation'
                    )
                    sharpe_chart.update_layout(showlegend=False, height=400)

                    minvol_chart = px.bar(
                        minvol_df,
                        x='Ticker',
                        y='Weight (%)',
                        title=f"| Return: {min_vol_return:.2%} | Volatility: {min_vol_std_dev:.2%} | Sharpe: {min_vol_ratio:.2f}",
                        labels={'Weight (%)': 'Allocation (%)'},
                        color='Weight (%)',
                        color_continuous_scale='Greens',
                        template='presentation'
                    )
                    minvol_chart.update_layout(showlegend=False, height=400)

                    optimisation_result = {
                        'sharpe_df': sharpe_df,
                        'minvol_df': minvol_df,
                        'sharpe_chart': sharpe_chart,
                        'minvol_chart': minvol_chart,
                        'sharpe_metrics': {
                            'return': max_sharpe_return,
                            'volatility': max_sharpe_std_dev,
                            'sharpe': max_sharpe_ratio
                        },
                        'minvol_metrics': {
                            'return': min_vol_return,
                            'volatility': min_vol_std_dev,
                            'sharpe': min_vol_ratio
                        }
                    }

            except Exception as e:
                optimisation_error = f"Optimisation failed: {str(e)}"

    return (optimisation_result, optimisation_error)


@app.cell
def _(
    mo,
    pd,
    px,
    go,
    optimisation_error,
    optimisation_result,
    ticker_selector,
    start_date,
    end_date,
    risk_free_rate,
    optimise_button,
):

    travel_data = pd.DataFrame({
        'City': ['England', 'Italy', 'France', 'Spain', 'Turkey', 'Portugal', 'Netherlands', 'Wales', 'Jamaica', 'Iran'],
        'Lat': [51.5074, 41.9028, 48.8566, 40.4168, 39.9334, 38.7223, 52.3676, 51.4816, 17.9714, 35.6892],
        'Lon': [-0.1278, 12.4964, 2.3522, -3.7038, 32.8597, 32.8597, 4.9041, -3.1791, -76.7922, 51.3890],
        'Visit_Year_str': ['2006', '2018', '2016', '2012', '2022', '2017', '2015', '2019', '2009', '2013']
    })

    years = sorted(travel_data['Visit_Year_str'].unique(), key=int)

    fig_travel = px.scatter_geo(
        travel_data,
        lat='Lat', lon='Lon',
        hover_name='City',
        color='Visit_Year_str',
        category_orders={'Visit_Year_str': years},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        projection="natural earth",
        title="My Travel Footprint",
        labels={'Visit_Year_str': 'Visit Year'}
    )
    fig_travel.update_traces(marker=dict(size=12))

    travel_data2 = pd.DataFrame({
        'City': ['Japan', 'Dubai', 'USA', 'Canada', 'Mexico', 'China', 'Bali'],
        'Lat': [35.6895, 25.2048, 38.9072, 45.4215, 19.4326, 39.9042, -8.6500],
        'Lon': [139.6917, 55.2708, -77.0369, -75.6972, -99.1332, 116.4074, 115.2167],
        'Priority': ['Top 1', 'Top 4', 'Top 3', 'Top 2', 'Top 5', 'Top 6', 'Top 7']
    })

    travel_reasons = {
        'Japan': 'Priority Top 1: \n\n From social media and word of mouth I have heard amazing things about Japan. From its scenic views at Mount Fuji to its shrines and modern streets. I also love Japanese culture and am currently learning the language. I also enjoy watching Naruto, and Japan has a Naruto theme park that I would love to visit.\n\nCities I want to visit: Tokyo, Kyoto, Osaka',
        'Dubai': 'Priority Top 4: \n\n For its modern architecture, luxury shopping, and vibrant nightlife.',
        'USA': 'Priority Top 3: \n\n For major financial centers and a wide range of cities to explore. America is probably the most well rounded country in the world. \n\nCities I want to visit: New York, Wyoming, Chicago',
        'Canada': 'Priority Top 2 (Visited ✓): \n\n Canada is where I visited this summer. I saw family there for the first time and it was amazing to finally meet them. It is also home to many beautiful views and ski resorts, and I am someone who really enjoys skiing.\n\nCities I visited / want to explore more: Toronto, Vancouver, Montreal',
        'Mexico': 'Priority Top 5: \n\n Mexico seems like a perfect holiday destination. From What I have heard Mexico has amazing beaches in places like Cancun, great food and festivals and also rich aztec history.\n\nCities I want to visit: Cancun, Mexico City, Tulum',
        'China': 'Priority Top 6: \n\n For the scale of the country and its history. I have many Chinese friends and hear great things about the country. It seems to have the perfect mix of everything especially in cities like Fuzhou which has city landscape but also Chinese architecture .  \n\nCities I want to visit: Beijing, Shanghai, Fuzhou',
        'Bali': 'Priority Top 7: \n\n Similar to Mexico, Bali offers stunning beaches, relaxation, and a completely different pace of life. This is a holiday I would love to take for 5 or 6 days.'
    }

    travel_seen_reasons = {
        'England': 'My country of residence, however, I havent truely explored Engish cities although I would like to do so.',
        'Italy': 'I have visited Italy on 3 different occasions and have visited many cities including Rome, Florence, and Venice, Naples, San Gimignano and . But I would have to say my personal favourite city was venice because it is truely beautiful and also it home to amazing food.',
        'France': "I've been to Paris twice: once on a school trip and once with family. While the Louvre's art collections, the Eiffel Tower's panoramic views, and Disneyland were highlights, I found the city lost some of its charm after a few days. The South of France remains on my bucket list, though France overall hasn't ranked among my favorite destinations.",
        'Spain': "Visited Barcelona and Madrid during my younger years—both cities left a lasting impression.",
        'Turkey': 'Stayed at a beautiful resort in Antalya with my family. The experience was relaxing and memorable.',
        'Portugal': 'Behind Italy, Portugal is my second favourite holdiay destination, I have a great recollection of vising Benfica Stadium and Club shop where I purchased one of their kits. Aside from football, Portugal has very nice residents and beaches and very underrated food.',
        'Netherlands': 'What struck me most was the clean, modern infrastructure and thoughtful urban planning. The cheese was exceptional too.',
        'Wales': "A peaceful escape with family, where we immersed ourselves in nature and stunning scenery. It offered exactly the quiet get away we needed.",
        'Jamaica': 'I visited Jamaica when i was very young. However, have very little recollection of the experience.',
        'Iran': "Visiting family in Iran is always meaningful, though political tensions sometimes make the experience nerve-wracking. The country's rich history fills me with pride in my cultural heritage."
    }

    fig_travel2 = px.scatter_geo(
        travel_data2,
        lat='Lat', lon='Lon',
        hover_name='City',
        color='Priority',
        category_orders={'Priority': ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        projection="natural earth",
        title="Where I want to travel to the most",
        labels={'Priority': 'Travel Priority'}
    )

    # Mark Canada as already visited with a green tick
    _canada = travel_data2[travel_data2['City'] == 'Canada'].iloc[0]
    fig_travel2.add_scattergeo(
        lat=[_canada['Lat']],
        lon=[_canada['Lon']],
        mode="markers+text",
        text=["✓"],
        textposition="middle center",
        textfont=dict(size=16, color="white"),
        marker=dict(size=24, color="green", line=dict(color="white", width=2)),
        name="Visited ✓",
        hovertemplate="<b>Canada</b><br>Already visited ✓<extra></extra>",
        showlegend=True,
    )

    Country_wishlist_chart = mo.ui.plotly(fig_travel2)

    return (travel_data, travel_data2, travel_reasons, travel_seen_reasons, fig_travel, fig_travel2, Country_wishlist_chart)


@app.cell
def _(mo, travel_reasons):
    city_select = mo.ui.dropdown(options=list(travel_reasons.keys()), label="Select a country")
    return (city_select,)


@app.cell
def _(city_select, mo, travel_reasons):
    city = city_select.value

    if city is None:
        selected_city_reason = mo.callout(
            mo.md("Select a country above to see why I want to visit!"),
            kind="info"
        )
    else:
        content = [mo.md(travel_reasons.get(city, "No data available"))]

        image_map_wishlist = {
            "Japan": ["img/Japan1.JPG", "img/Japan2.JPG"],
            "Dubai": ["img/Dubai1.JPG"],
            "USA": ["img/USA1.JPG"],
            "Canada": ["img/Canada1.JPG"],
            "Mexico": ["img/Mexico1.JPG"],
            "China": ["img/China1.JPG"],
            "Bali": ["img/Bali1.JPG"],
        }

        if city in image_map_wishlist:
            content.append(
                mo.hstack(
                    [mo.image(img, width=300) for img in image_map_wishlist[city]],
                    gap=0
                )
            )

        selected_city_reason = mo.vstack(content)

    return (selected_city_reason,)


@app.cell
def _(mo, travel_data):
    city_seen_select = mo.ui.dropdown(
        options=list(travel_data["City"]),
        label="Select a country",
    )
    return (city_seen_select,)


@app.cell
def _(city_seen_select, fig_travel, travel_data, go):
    selected_travel_map = go.Figure(fig_travel)
    if city_seen_select.value:
        selected_row_seen = travel_data.loc[
            travel_data["City"] == city_seen_select.value
        ].iloc[0]
        selected_travel_map.add_trace(
            go.Scattergeo(
                lat=[selected_row_seen.Lat],
                lon=[selected_row_seen.Lon],
                mode="markers+text",
                text=[selected_row_seen.City],
                textposition="top center",
                marker=dict(size=18, color="red", line=dict(color="white", width=2)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{selected_row_seen.City}</b><br>"
                    f"Visit year: {selected_row_seen.Visit_Year_str}<extra></extra>"
                ),
            )
        )
    return (selected_travel_map,)


@app.cell
def _(city_select, fig_travel2, travel_data2, go):
    selected_wishlist_map = go.Figure(fig_travel2)
    if city_select.value:
        selected_index = travel_data2.index[
            travel_data2["City"] == city_select.value
        ][0]
        selected_row_wishlist = travel_data2.iloc[selected_index]
        selected_wishlist_map.add_trace(
            go.Scattergeo(
                lat=[selected_row_wishlist.Lat],
                lon=[selected_row_wishlist.Lon],
                mode="markers+text",
                text=[selected_row_wishlist.City],
                textposition="top center",
                marker=dict(size=18, color="red", line=dict(color="white", width=2)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{selected_row_wishlist.City}</b><br>"
                    f"Priority: {selected_row_wishlist.Priority}<extra></extra>"
                ),
            )
        )
    return (selected_wishlist_map,)


@app.cell
def _(city_seen_select, mo, travel_seen_reasons):
    seen_city = city_seen_select.value

    if seen_city is None:
        selected_seen_city_reason = mo.callout(
            mo.md("Select a country to view details."),
            kind="info"
        )
    else:
        seen_content = [mo.md(travel_seen_reasons.get(seen_city, "No data available"))]

        image_map = {
            "Italy": ["img/Italy1.jpg", "img/Italy3.jpg", "img/Italy4jpg.jpg", "img/Italy5.jpg"],
            "France": ["img/France1.jpg", "img/France2.jpg"],
            "Spain": ["img/Spain1.jpg", "img/Spain2.jpg"],
            "Turkey": ["img/Turkey1.jpg"],
            "Wales": ["img/Wales1.jpg", "img/Wales2.jpg"],
            "Portugal": ["img/Port.jpg", "img/Port2.jpg", "img/Port3.jpg"],
            "Iran": ["img/Iran1.jpg", "img/Iran2.jpg"],
        }

        if seen_city in image_map:
            seen_content.append(
                mo.hstack(
                    [mo.image(img, width=300) for img in image_map[seen_city]],
                    gap=0
                )
            )

        selected_seen_city_reason = mo.vstack(seen_content)

    return (selected_seen_city_reason,)


@app.cell
def _():
    skill_to_sources = {
        "Python Programming": {
            "education": [
                "Research Methods (First Year Module)",
                "Introduction to Data Science and AI Tools (First Year Module)",
            ],
            "experiences": [
                "Personal learning",
                "London Finance Committee: built finance games and a portfolio optimiser in Python",
            ],
            "societies": ["...."],
        },
        "Data Visualization": {
            "education": [
                "Research Methods of accounting and finance (First Year Module)",
                "Introduction to Data Science and AI Tools (First Year Module)",
            ],
            "experiences": [
                "London Finance Committee: built interactive dashboards with Plotly and Marimo",
                "ZS Spring Insight Day: Presented trends with Plotly dashboards",
                "Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS",
                "Bloomberg Finance Fundamental, Market Concept and ESG Certificates",
            ],
            "societies": ["City Innovation Hub", "Bayes Finance & Banking Club"],
        },
        "Financial Analysis": {
            "education": [
                "Intro to Finance",
                "Financial Institutions",
                "Intro Financial Accounting",
                "Intro to Data Science and AI Tools",
                "Research Methods of Accounting and Finance",
            ],
            "experiences": [
                "London Finance Committee: published market analysis across equities and commodities",
                "BDO Virtual Work Experience: Audit case studies and financial statement analysis",
                "Bloomberg Finance Fundamental, Market Concept and ESG Certificates",
                "Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS",
                "Goldman Sachs Risk Forage",
            ],
            "societies": ["Bayes Finance & Banking Club", "Target Finance & Investment Society"],
        },
        "Commercial Awareness": {
            "education": ["Intro to Finance", "Financial Institutions"],
            "experiences": [
                "London Finance Committee: tracking macroeconomic trends and forming market views",
                "JP Morgan MyPlus Future You Insight Event: networking with professionals across JP Morgan, Capgemini, KPMG and NESO",
                "Morgan Stanley Early Insight Series: Sales and Trading sessions",
                "ZS Spring Insight Day: Understanding client priorities in healthcare",
                "PIMCO Spring Insight: Learning about fixed income markets and macro trends",
                "Bloomberg Finance Fundamental, Market Concept and ESG Certificates",
                "Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS",
                "Goldman Sachs Risk Forage",
            ],
            "societies": [
                "Finance Podcast Society",
                "Bayes Finance & Banking Club",
                "Target Finance & Investment Society",
                "City Innovation Hub",
            ],
        },
        "Attention to Detail": {
            "education": ["Intro Financial Accounting", "Research Methods"],
            "experiences": [
                "London Finance Committee: data cleaning and accuracy in published analysis",
                "BDO Virtual Work Experience: Audit testing and tax concepts",
                "Morgan Stanley Early Insight Series: Sales and Trading sessions",
                "ZS Spring Insight Day: Understanding client priorities in healthcare",
                "PIMCO Spring Insight: Learning about fixed income markets and macro trends",
                "Bloomberg Finance Fundamental, Market Concept and ESG Certificates",
                "Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS",
                "Goldman Sachs Risk Forage",
                "Volounteering at the PEEL institute",
                "Carer Responsibilities",
            ],
            "societies": ["City Innovation Hub", "Bayes Finance & Banking Club"],
        },
        "Communication": {
            "education": ["Research Methods"],
            "experiences": [
                "London Finance Committee: publishing written market analysis for readers",
                "JP Morgan MyPlus Future You Insight Event: professional networking",
                "ZS Spring Insight Day: Presented findings clearly",
                "NHS Shadowing: Professional communication in clinical settings",
                "Volounteering at the PEEL institute",
                "Carer Responsibilities",
                "City Buddies Mentor and Program representative for Accounting and Finance, City University of London",
            ],
            "societies": ["City Buddies Mentor Program", "City Christian Union Society"],
        },
        "Resilience": {
            "education": ["Each of my modules respectively"],
            "experiences": [
                "Carer Responsibilities",
                "Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS",
                "City Buddies Mentor and Program representative for Accounting and Finance, City University of London",
            ],
            "societies": ["City Christian Union Society", "Bayes Finance & Banking Club"],
        },
        "Problem Solving": {
            "education": ["Each of my modules respectively"],
            "experiences": [
                "London Finance Committee: designed and built a portfolio optimiser and credit-risk dashboard",
                "Blackmont Consulting: selective consulting assessment centre",
                "BDO Virtual Work Experience: Applied accounting logic to case problems",
                "Goldman Sachs Risk Forage: Completed risk case studies and simulations",
                "City Buddies Mentor and Program representative for Accounting and Finance, City University of London",
                "Project Management Training Course",
                "Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS",
            ],
            "societies": ["City Innovation Hub", "Target Finance & Investment Society"],
        },
    }
    return (skill_to_sources,)


@app.cell
def _(mo, skill_to_sources):
    skill_picker = mo.ui.radio(
        options=list(skill_to_sources.keys()),
        value="Python Programming",
        label="Click a skill to see its source",
    )
    return (skill_picker,)


@app.cell
def _(mo, skill_picker, skill_to_sources):
    selected_skill = skill_picker.value
    selected_sources = skill_to_sources[selected_skill]
    education_lines = "\n".join(f"- {item}" for item in selected_sources["education"])
    experience_lines = "\n".join(f"- {item}" for item in selected_sources["experiences"])
    society_lines = "\n".join(f"- {item}" for item in selected_sources["societies"])

    selected_skill_sources = mo.callout(
        mo.md(
            f"""### {selected_skill}

**Education that built this skill:**

{education_lines}

**Experiences that built this skill:**

{experience_lines}

**Societies that built this skill:**

{society_lines}
"""
        ),
        kind="success",
    )
    return (selected_skill_sources,)


@app.cell
def _():
    stage_to_reason = {
        "Y12": "Y12: I reset my mindset and focused on building discipline from the ground up after suffering such unexpeccted downturn the year prior.",
        "Y13": "Y13: I became more consistent with revision and improved my study structure. I learned Anki which is a key revision tool I use today. I completed my A-Levels and although I didnt achieve the grades I expect from myself, I learned a lot about myself and my capabilities. Most importantly, I never stopped believing in my hard work and that once my tough times settled, I would progress exponentially.",
        "Foundation Year": "Foundation Year: I built a stronger academic base and regained confidence. However, my mother underwent Chemotherapy during this time and despite working very hard during this time, the responsibility i had to carry with trying to be a strong academic and to also take care of me proved very challenging. Although this was the most difficult year for me, the summer was the most important period for me. My mother finished her treatment and I noticed that I became very conscious of my future. I wanted to make sure that my suffering had a strong story at the end of me, so i rested and thought for the future during that summer, preparing me for the year ahead. ",
        "First year": "First year: I achieved a major leap through resilience, preparation, and effort.",
        "Second year projection": "Second year projection: I plan do maintain my strong momentum and add to my extra curriculur activities while maintaining strong grades and land a summer internship.",
        "Third year projection": "Third year projection: I aim to convert my internships, to stay commercially aware build strong professional networks and be consistent with my grades.",
    }
    return (stage_to_reason,)


@app.cell
def _(mo, stage_to_reason):
    progress_stage_select = mo.ui.dropdown(
        options=list(stage_to_reason.keys()),
        label="Select a stage",
    )
    return (progress_stage_select,)


@app.cell
def _(mo, progress_stage_select, stage_to_reason):
    selected_stage_reason = mo.callout(
        mo.md(
            stage_to_reason[progress_stage_select.value]
            if progress_stage_select.value
            else "Select a stage above to see my self-improvement reflection!"
        ),
        kind="info",
    )
    return (selected_stage_reason,)


@app.cell
def _(mo, px):
    grades = {
        "Financial Institutions": 85.8,
        "Introductory Financial Accounting": 87.5,
        "Principles of Economics": 79,
        "Introductory Management Accounting": 92.7,
        "Research Methods": 76,
        "Principles of Taxation": 94.8,
        "Introduction to Data Science and AI Tools": 73,
        "Introduction to Finance": 86.8,
    }

    fig = px.bar(
        x=list(grades.values()),
        y=list(grades.keys()),
        orientation='h',
        text_auto=True,
        title="First Year Module Grades (%)"
    )

    fig.add_vline(x=70, line_dash="dash", line_color="green",
        annotation=dict(
            text="First Class (70%)",
            font=dict(color="green"),
            x=70, xref="x",
            y=1.07, yref="paper",
            showarrow=False,
            yanchor="top"
        )
    )

    fig.add_vline(x=60, line_dash="dash", line_color="orange",
        annotation=dict(
            text="2:1 (60%)",
            font=dict(color="orange"),
            x=60, xref="x",
            y=1.07, yref="paper",
            showarrow=False,
            yanchor="top"
        )
    )

    fig.update_layout(xaxis_title="Percentage", yaxis_title="Modules")
    grades_chart = mo.ui.plotly(fig)
    average_grade = sum(grades.values()) / len(grades)
    average_grade_text = mo.md(
        f"<div style='text-align:center; font-size:1.6rem; font-weight:700;'>Average grade: {average_grade:.1f}%</div>"
    )

    from pathlib import Path
    screenshot_path = Path("public/education_screenshot.png")
    if screenshot_path.exists():
        education_download = mo.download(
            screenshot_path.read_bytes(),
            filename="education_screenshot.png",
            mimetype="image/png",
            label="📥 Evidence Screenshot",
        )
    else:
        education_download = mo.md("")

    extenuating_expand = mo.accordion(
        {
            "Extenuating Circumstances": mo.md(
                """ When I was 15 years old, during my GCSE's my father passed away. My mother was also diagnosed with cancer during this time. This was a very difficult time for me and my family, and it had a significant impact on my academic performance during that period. I struggled to focus on my studies and my grades suffered as a result. However, I was able to overcome this challenge with the support of my family and teachers, and I was able to improve my grades in the following years. I am proud of how I was able to persevere through such a difficult time and come out stronger on the other side. """
            )
        }
    )

    screenshot_urls = [
        "https://i.ibb.co/Q6gdfRkc/098-EB7-DE-8-F24-41-BE-98-B6-3364-C617-BE0-C.jpg",
        "https://i.ibb.co/XfFZLmQL/642-F0-E0-F-77-BB-4-D32-9-D38-7-D47-A3-F8-F01-B.jpg",
        "https://i.ibb.co/TDvzffk3/1-A4-B50-B9-8325-4-CE6-8-B86-77-D906-DEC389.jpg",
    ]
    education_view_link = mo.md("\n".join([f"- [Screenshot Evidence {i+1}]({url})" for i, url in enumerate(screenshot_urls)]))

    return (grades_chart, average_grade_text, education_download, extenuating_expand, education_view_link)


@app.cell
def _(mo, px):

    progression_stages = ['Y12', 'Y13', 'Foundation', 'First year', 'Second year projection', 'Third year projection']
    progression_scores = [0, 10, 20, 65, 85, 100]
    progression_plot = px.line(
        x=progression_stages,
        y=progression_scores,
        markers=True,
        title='Self Improvement Monitor as a student',
        labels={'x': 'Education Stage', 'y': 'Self Improvement Monitor'},
        template='plotly_white',
        width=900,
        height=550
    )
    progression_notes = [
        "Starting point and reset mindset.",
        "Built consistency and better study habits.",
        "Foundation year created stronger academic base.",
        "Major breakthrough in confidence and results.",
        "Continuing upward momentum with discipline.",
        "Targeting sustained excellence and career readiness.",
    ]
    progression_plot.update_traces(
        mode='markers',
        marker=dict(size=10),
        customdata=progression_notes,
        hovertemplate=(
            "<b>Self Improvement Profile</b><br>"
            "Stage: %{x}<br>"
            "Monitor score: %{y}<br>"
            "Note: %{customdata}"
            "<extra></extra>"
        ),
    )

    progression_plot.add_scatter(
        x=progression_stages[:4],
        y=progression_scores[:4],
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        hoverinfo='skip',
        showlegend=False,
    )

    progression_plot.add_scatter(
        x=progression_stages[3:],
        y=progression_scores[3:],
        mode='lines',
        line=dict(color='#1f77b4', width=3, dash='dot'),
        hoverinfo='skip',
        showlegend=False,
    )

    progression_plot.update_layout(hovermode='closest')
    progression_chart = mo.ui.plotly(progression_plot)

    checklist_ui = mo.vstack(
        [
            mo.ui.checkbox(label="Build two portfolio projects with Python and data visualization."),
            mo.ui.checkbox(label="Strengthen technical interview preparation."),
            mo.ui.checkbox(label="Expand professional network through events and mentoring."),
            mo.ui.checkbox(label="Improve commercial awareness with weekly market reviews."),
            mo.ui.checkbox(label="Start a Finance related society."),
            mo.ui.checkbox(label="Final goal Secure a summer internship in finance."),
        ],
        gap=0.25,
    )

    return (progression_chart, checklist_ui)


@app.cell
def _(mo, go):

    fig_pitch = go.Figure()
    fig_pitch.update_layout(
        title="Football Pitch",
        plot_bgcolor="#177245",
        paper_bgcolor="#177245",
        xaxis=dict(range=[0, 120], visible=False),
        yaxis=dict(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=45, b=10),
        width=760,
        height=500,
        showlegend=False,
    )
    fig_pitch.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="white", width=2))
    fig_pitch.add_shape(type="circle", x0=59, y0=39, x1=61, y1=41, fillcolor="white", line=dict(color="white"))
    fig_pitch.add_shape(type="circle", x0=11, y0=39, x1=13, y1=41, fillcolor="white", line=dict(color="white"))
    fig_pitch.add_shape(type="circle", x0=107, y0=39, x1=109, y1=41, fillcolor="white", line=dict(color="white"))
    fig_pitch.add_trace(
        go.Scatter(
            x=[88],
            y=[70],
            mode="markers+text",
            text=["11"],
            textposition="top center",
            textfont=dict(color="white", size=12),
            hovertemplate=(
                "<b>My player profile</b><br>"
                "Position: Left Winger (Lw)<br>"
                "Playstyle: Enjoys doing skills, but also loves to press and track back to help my defenders.<br>"
                "Strong foot: Right<br>"
                "Jersey number: 11"
                "<extra></extra>"
            ),
            marker=dict(size=22, color="black", line=dict(color="white", width=2)),
            showlegend=False,
        )
    )

    sports_visual = mo.ui.plotly(fig_pitch)

    notes = mo.ui.tabs({
        "Notes on Philippians 4:6-7": mo.md('Philippians 4:6-7 (NIV): "Do not be anxious about anything, but in every situation, by prayer and petition, with thanksgiving, present your requests to God. And the peace of God, which transcends all understanding, will guard your hearts and your minds in Christ Jesus." \n\n This verse was one of my favourite verses upon introduction to Christianity. I used to be someone who was governed by anxiety, however, this verse shifted my mentality completely and it felt like I was taking the weight off if my struggles off of my own shoulders and getting support in my growth journey, giving a positive externality on my confidence and the way I carry myself .'),
        "Notes on Hebrews 11:1": mo.md("Hebrews 11:1: Now faith is confidence in what we hope for and assurance about what we do not see.\n\n This verse is very similar to the verse present in Philippians, and it also had a very strong impact on me. It made me realise that I can have confidence in the future I am working towards, even if I cannot see it yet. It also made me realise that I can have confidence in the person I am becoming, even if I cannot see that yet either. But more importantly, it means that despite not being able to see God, he is always there, and in times where I feel distant from him, he is testing my faith as the scripture tells me to believe even though I cannot see with my 2 eyes. Gods silences in tests are similar to the silence of a teacher during a test and that revising his word optimises the . "),
        "Notes on Romans 1:16": mo.md('"For I am not ashamed of the gospel, because it is the power of God that brings salvation to everyone who believes" \n\n Making your faith in God known as a Christian is something very important in order to enter the afterlife, so by being proud of your faith and spreading your love of the gospel it provides many positive externalities such as building a strong community of believers around you, and also being a light to others who may be struggling with their faith or going through tough times. I want to be proud of my faith and share it with others, and this verse reminds me of the importance of doing so has a positive impact on others.'),
    })

    return (sports_visual, notes)


@app.cell
def _(
    mo, grades_chart, average_grade_text, education_download, education_view_link,
    extenuating_expand, skill_picker, selected_skill_sources, progression_chart,
    progress_stage_select, selected_stage_reason, checklist_ui, city_seen_select,
    city_select, selected_city_reason, selected_seen_city_reason, selected_travel_map,
    selected_wishlist_map, travel_data, travel_data2, sports_visual, notes,
    animated_sector_selector, animated_z_slider, animated_cap_slider, animated_stock_search,
    DB_fig, box_fig, risk_dist_fig, heatmap_fig, company_count, avg_cost, median_cost,
    avg_zscore, max_cap_filtered, distress_threshold, safe_threshold, risk_distress,
    risk_intermediate, risk_safe, distress_table, intermediate_table, safe_table,
    distress_returns_fig, intermediate_returns_fig, safe_returns_fig, sp_fig_recent,
    sp_fig_historic, trailing_pe, forward_pe, div_yield, df_filtered,
    md, px, pd, ticker_selector, start_date, end_date, risk_free_rate, optimise_button,
    optimisation_result, optimisation_error, search_message_display
):

    dashboard_instructions = mo.callout(
        mo.md("""
## How to Use This Dashboard

**This dashboard analyzes S&P 500 companies' credit risk and debt costs.**

### Quick Start:
1. **Adjust Filters** - Use the controls below to filter companies by sector, Z-Score, and market cap
2. **Search** - Type a ticker (e.g., 'AAPL') or company name to focus on specific companies
3. **Explore Tabs** - Navigate between Analysis, Metrics, and Risk Zones for different insights

### Performance Tips:
- Start with **higher market cap thresholds** ($50B+) for faster loading
- Select **fewer sectors** to reduce data points
- Use the **search box** to highlight individual companies

### What You'll Find:
- **Analysis Tab**: Visual correlations between Z-Scores and debt costs
- **Key Metrics Tab**: Summary statistics of filtered companies
- **Risk Zones Tab**: Companies grouped by financial health (Distress/Intermediate/Safe)
"""),
        kind="neutral"
    )

    tab_cv = mo.vstack([
        mo.md("### First year student at Bayes Business School | ZS, Morgan Stanley and JP Morgan Spring Insight Participant | Interested in Sales & Trading, Financial Analysis, and Data Science"),
        mo.md("**Summary:**\n\n• Energetic and resilient student with a strong fascination for finance. Eager to develop practical analytical skills, contribute effectively to a team, and build strong commercial awareness. Particularly interested in Finance, where I aim to apply and further develop my analytical skills. Passionate about uncovering market insights using modern data tools like Python, Marimo, and Plotly."),
        mo.md("**Education:**\n\n**BSc Accounting & Finance**, Bayes Business School (2024 - 2028)\n\n**Secondary School & 6th Form**, St John's School, London (2017 - 2024)\n\n**Grades:**\n\n• GCSEs: Grades 8-5, including Mathematics (8) and English Language (6)\n\n• A Levels: Mathematics: (B), Chemistry (C), Biology (C)\n\n• First Year Modules: Financial Institutions: 85.8%, Introductory Financial Accounting: 87.5%, Principles of Economics: 79%, Introductory Management Accounting: 92.7%, Research Methods: 76%, Principles of Taxation: 94.8%, Introduction to Data Science and AI Tools: 73%, Introduction to Finance: 86.8%. Overall first-year average: 84.5%.\n\n• Context behind GCSE's and A Levels"),
        extenuating_expand,
        grades_chart,
        average_grade_text,
        education_download,
        education_view_link,
        mo.md("**Relevant Websites:**\n\n• LinkedIn: [LinkedIn](https://www.linkedin.com/in/aryan-ainesaz/)\n\n• GitHub: [GitHub](https://github.com/aryan-ainesaz)\n\n Email: Aryan.Ainesaz@bayes.city.ac.uk")
    ])

    work_experience_md = mo.md("""##Work Experience:

**London Finance Committee** (June 2026)

• Founded a financial news platform and published monthly market analysis, tracking macroeconomic trends across equities and commodities with an independent view on market direction.

• Developing finance-related games and a stock-portfolio competition, planned for launch this summer.


**Blackmont Consulting** (June 2026)

• Selected as one of 7 from 32 candidates at the assessment centre for this competitive consulting internship.


**ZS Spring Insight Day** (April 2026)

• Gained understanding of consultant workflows and the responsibility of facing clients from managers.

• Developed awareness of associate expectations and the structured onboarding approach ZS uses to integrate new members.


**Morgan Stanley Early Insight Series** (March 2026)

• Learned about the different sectors and roles at Morgan Stanley. Improved personal commercial awareness skills.

• Got valuable information from Sales and Trading directors and analysts about the application process, the impact of current macroeconomic conditions and the standards expected from a candidate in the field.


**JP Morgan MyPlus Future You Insight Event** (March 2026)

• Selected for the in-person MyPlus Future You insight event at JP Morgan's headquarters.

• Networked with professionals from JP Morgan, Capgemini, KPMG and NESO, building industry connections and commercial awareness.


**The Peel Institute Volunteer** (March 2026)

• Distributed food and essentials to vulnerable individuals, demonstrating empathy and community engagement.

• Supported the team in events, showcasing teamwork and organisational skills.


**Accounting, Virtual Work Experience, BDO** (October 2025)

• Completed virtual audit and accounting case studies, developing an understanding of financial statements, audit testing and tax concepts.

• Gained insights into interview expectations and the skills required for early-career roles. Picked up valuable employment information and interview techniques.


**Shadowing NHS Plastic Surgeons** (August 2023)

• Observed communication and teamwork in a professional clinical setting.

• Learned to stay focused in a busy, demanding environment and improved organisational skills.
""")

    tab_work_experience_skills = mo.vstack([
        work_experience_md,
        mo.md("## Skills Explorer"),
        skill_picker,
        selected_skill_sources,
    ])

    tab_personal_interests = mo.md("""
**Extracurriculum Activities:** \n\n • Bloomberg Finance Fundamental, Market Concept and ESG Certificates \n\n • Amplify Me Finance Accelerator in Partnership with Morgan Stanley and UBS \n\n • City Buddies Mentor and Program representative for Accounting and Finance, City University of London \n\n • Goldman Sachs Risk Forage \n\n • Project Management Training Course

**Societies** \n\n • City Innovation hub \n\n • Finance Podcast Society \n\n • Bayes Finance & Banking Club \n\n • City Christian Union socity \n\n • Target Finance & Investment Society
""")

    tab_future_expectation_checklist = mo.vstack([
        mo.md("""##My Ambitions
- My goal is to build a career in finance. While my GCSE and A-Level results were not a true reflection of my potential due to challenging circumstances at the time, they have become a source of motivation rather than limitation. Since then, I have focused on proving my ability through my university performance, work experience, and personal projects.

- I am driven by the belief that progress and resilience can be just as powerful as a flawless academic record. The discipline and determination I have developed through my experiences continue to shape my approach as I work towards a career in finance.
"""),
        progression_chart,
        progress_stage_select,
        selected_stage_reason,
        mo.md("## Summer Checklist: Building Momentum for Year 2"),
        checklist_ui,
    ])

    tab_personal = mo.ui.tabs({
        "Countries I've Seen": mo.vstack([
            mo.md("## 🌍 The Countries I have visited \n\n (If visited multiple times date applies to first visit)"),
            mo.ui.plotly(selected_travel_map),
            city_seen_select,
            selected_seen_city_reason,
            mo.callout(mo.md(f"I have seen {len(travel_data)} countries so far."), kind="info"),
        ]),
        "Places I Want to Visit": mo.vstack([
            mo.md("## ✈️ My Travel Wishlist"),
            mo.ui.plotly(selected_wishlist_map),
            city_select,
            selected_city_reason,
            mo.callout(mo.md(f"These are the {len(travel_data2)} places I want to visit most."), kind="info")
        ]),
        "Sports": mo.vstack([
            mo.md("## Football"),
            sports_visual,
            mo.md("• I enjoy both watching and playing football.\n\n• The plot above shows the position I usually played for my secondary school team.\n\n• I support Manchester United, and I have been following the team for as long as I can remember.\n\n• I also enjoy watching the Premier League in general, and I try to keep up with the latest news and matches where I can."),
            mo.md("## Gym"),
            mo.md("• I have been going to the gym for about 3 years now and I really enjoy it. I usually go 4 or 5 times a week and I like to do a mix of weight training and cardio. I find that going to the gym is a great way to relieve stress and stay healthy, and I also enjoy the sense of accomplishment that comes with seeing progress in my strength and fitness levels. Unfortunately, Recently, I have not been able to go to the gym because of my jaw surgery and am excited to start again soon.\n\n• I have also contributed by encouraging others to stay consistent, sharing workout routines, and helping create a positive training environment. ")
        ]),
        "Religion": mo.vstack([
            mo.md("## Religion"),
            mo.md("• I have recenty converted to Christianity and am actively learning about the faith and how to better myself."),
            mo.md("• I intend on going to church regularly to grow closer to my faith."),
            mo.md("• I like to read the Bible, make notes on my readings and pray daily."),
            mo.md("• Here are some of the notes ive taken from my Bible study."),
            notes,
        ]),
    })

    tab_dashboard_analysis = mo.vstack([
        mo.md("# S&P500 Credit Risk Analysis"),
        dashboard_instructions,
        animated_sector_selector,
        animated_z_slider,
        animated_cap_slider,
        animated_stock_search,
        search_message_display,
        mo.md("---"),
        mo.ui.plotly(DB_fig),
        mo.md("## Cost of Debt by Sector"),
        mo.ui.plotly(box_fig),
        mo.md("## Financial Signal Map (Filtered Data)"),
        mo.ui.plotly(heatmap_fig),
        mo.md("## Debt Cost Distribution by Risk Zone"),
        mo.ui.plotly(risk_dist_fig),
    ])

    tab_dashboard_metrics = mo.vstack([
        animated_sector_selector,
        animated_z_slider,
        animated_cap_slider,
        animated_stock_search,
        search_message_display,
        mo.md("---"),
        mo.md(md(f"""
## Key Metrics Summary

| Companies Analyzed | Avg. Cost of Debt | Median Cost | Avg Z-Score | Max Market Cap ($B) |
| :---: | :---: | :---: | :---: | :---: |
| **{company_count}** | **{avg_cost:.2f}%** | **{median_cost:.2f}%** | **{avg_zscore:.2f}** | **{max_cap_filtered:.2f}** |

## Safety Levels
| Safety Level | Meaning |
| :---: | :---: |
| Distress | Z-Score below {distress_threshold} |
| Intermediate | Z-Score between {distress_threshold} and {safe_threshold} |
| Safe | Z-Score above {safe_threshold} |
"""))
    ])

    risk_summary_md_text = md(f"""
## Risk Summary
- **Distress:** {risk_distress} firms
- **Intermediate:** {risk_intermediate} firms
- **Safe:** {risk_safe} firms
""")

    pie_fig = px.pie(
        names=["Distress", "Intermediate", "Safe"],
        values=[risk_distress, risk_intermediate, risk_safe],
        color=["Distress", "Intermediate", "Safe"],
        color_discrete_map={"Distress": "red", "Intermediate": "gold", "Safe": "green"},
        title="Firm Risk Distribution"
    )
    pie_fig.update_traces(textinfo='label+percent', pull=[0.05, 0.05, 0.05])
    pie_fig.update_layout(
        font=dict(size=13),
        paper_bgcolor="white",
        margin=dict(t=55, b=20, l=20, r=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    )

    tab_dashboard_risk = mo.ui.tabs({
        "Firm Distribution": mo.vstack([
            animated_sector_selector,
            animated_z_slider,
            animated_cap_slider,
            animated_stock_search,
            search_message_display,
            mo.md("---"),
            mo.md(risk_summary_md_text),
            mo.ui.plotly(pie_fig),
        ]),
        "Distress": mo.vstack([
            mo.md(md(f"""
## Distress Zone (Z-Score < {distress_threshold})
### Top 5 Riskiest Firms
*Select rows in the table to highlight those firms in the chart below.*
""")),
            distress_table if distress_table is not None else mo.md("No firms in distress zone with current filters."),
            mo.md("## Top 5 Riskiest Firms — Returns (Last 1 Year)"),
            mo.ui.plotly(distress_returns_fig),
        ]),
        "Intermediate": mo.vstack([
            mo.md(md(f"""
## Intermediate Zone (Z-Score between {distress_threshold} and {safe_threshold})
### Top 5 Intermediate Firms
*Select rows in the table to highlight those firms in the chart below.*
""")),
            intermediate_table if intermediate_table is not None else mo.md("No firms in intermediate zone with current filters."),
            mo.md("## Top 5 Intermediate Firms — Returns (Last 1 Year)"),
            mo.ui.plotly(intermediate_returns_fig),
        ]),
        "Safe": mo.vstack([
            mo.md(md(f"""
## Safe Zone (Z-Score > {safe_threshold})
### Top 5 Safest Firms
*Select rows in the table to highlight those firms in the chart below.*
""")),
            safe_table if safe_table is not None else mo.md("No firms in safe zone with current filters."),
            mo.md("## Top 5 Safest Firms — Returns (Last 1 Year)"),
            mo.ui.plotly(safe_returns_fig),
        ]),
    })

    tab_dashboard_sp_info = mo.vstack([
        mo.md("## S&P 500 Time Series (Closing Price)"),
        sp_fig_recent,
        mo.md("## S&P 500 Firms by Sector (Current Filters)"),
        mo.ui.plotly(
            px.pie(
                df_filtered.groupby("Sector_Key")["Ticker"].nunique().reset_index().rename(columns={"Ticker": "Firms", "Sector_Key": "Sector"}) if len(df_filtered) > 0 else pd.DataFrame({"Sector": ["No Data"], "Firms": [1]}),
                names="Sector",
                values="Firms",
                title="S&P 500 Firms by Sector",
                template="presentation",
                height=450,
            ).update_traces(textinfo="label+percent", textposition="inside").update_layout(
                font=dict(size=13),
                paper_bgcolor="white",
                margin=dict(t=50, b=20, l=20, r=20),
                showlegend=False,
            )
        ),
    ])

    tab_dashboard_fundamentals = mo.ui.tabs({
        "Historic Data": mo.vstack([
            mo.md("## Historic Price Data (20 Years)"),
            sp_fig_historic,
        ]),
        "Fundamentals": mo.md(md(f"""
## S&P 500 Fundamentals (^GSPC, as of March 2026)

| Metric | Value | What it tells you |
| :--- | :---: | :--- |
| **Trailing P/E Ratio** | {trailing_pe} | Market price relative to the last 12 months of actual earnings. High values may indicate overvaluation. |
| **Forward P/E Ratio** | {forward_pe} | Market price relative to next 12 months of *expected* earnings. Lower than trailing P/E suggests earnings growth is anticipated. |
| **Dividend Yield** | {div_yield} | Annual dividend payout as a % of index price. Down from a historical average of ~2.87%, reflecting the market's growth-oriented composition. |
| **Headline CPI** | +0.3% MoM | Rose 0.3% month-over-month, matching economist expectations. Signals continued but stabilising inflationary pressure. |
| **Performance** | Volatile | The S&P 500 has experienced volatility, reflecting uncertainty around interest rates, earnings expectations, and macro conditions. |

*Values as of March 2026.*
"""))
    })

    portfolio_optimiser_instructions = mo.callout(
        mo.md("""
##  Portfolio Optimisation Tool

**This tool helps you build optimal portfolios using Modern Portfolio Theory.**

### How It Works:
1. **Select Stocks** - Choose 3-10 stocks from the S&P 500
2. **Set Date Range** - Define the historical period for analysis
3. **Set Risk-Free Rate** - Current 10 year treasury rate is approximately 4.25 for reference
4. **Click Optimise** - Calculate two optimal portfolios:
   - **Maximum Sharpe Ratio**: Best risk-adjusted returns
   - **Minimum Volatility**: Lowest risk portfolio

### What You'll Get:
- Optimal weight allocations for each stock
- Expected annual returns and volatility
- Sharpe ratios for portfolio comparison
- Visual charts showing allocations
"""),
        kind="neutral"
    )

    if optimisation_error:
        optimisation_display = mo.callout(mo.md(f"**Error:** {optimisation_error}"), kind="danger")
    elif optimisation_result:
        optimisation_display = mo.vstack([
            mo.md("## Optimisation Complete"),
            mo.md("### Maximum Sharpe Ratio Portfolio"),
            mo.ui.plotly(optimisation_result['sharpe_chart']),
            mo.ui.table(optimisation_result['sharpe_df']),
            mo.md("---"),
            mo.md("### Minimum Volatility Portfolio"),
            mo.ui.plotly(optimisation_result['minvol_chart']),
            mo.ui.table(optimisation_result['minvol_df']),
        ])
    else:
        optimisation_display = mo.md("*Click the optimise button to calculate optimal portfolios.*")

    tab_portfolio_optimiser = mo.vstack([
        mo.md("# Portfolio Optimisation"),
        portfolio_optimiser_instructions,
        mo.md("### Configuration"),
        ticker_selector,
        mo.hstack([start_date, end_date], justify="start"),
        risk_free_rate,
        optimise_button,
        mo.md("---"),
        optimisation_display,
    ])

    tab_data_content = mo.ui.tabs({
        "S&P 500 Dashboard": mo.ui.tabs({
            "S&P Information": tab_dashboard_sp_info,
            "Credit Risk Analysis": mo.ui.tabs({
                "Analysis": tab_dashboard_analysis,
                "Key Metrics": tab_dashboard_metrics,
                "Risk Zones": tab_dashboard_risk,
            }),
            "S&P Fundamentals": tab_dashboard_fundamentals,
        }),
        "Portfolio Optimiser": tab_portfolio_optimiser,
    })

    tab_profile = mo.ui.tabs({
        "Academics & Experience": tab_cv,
        "Extracurriculars & Personal Interests": tab_personal_interests,
        "Work Experience & Skills": tab_work_experience_skills,
        "Future expectation checklist": tab_future_expectation_checklist,
    })

    return (tab_profile, tab_data_content, tab_personal)


@app.cell
def _(mo, tab_profile, tab_data_content, tab_personal):
    # Final assembly (WHITE BACKGROUND - no gradient layers)
    app_tabs = mo.ui.tabs({
        "Personal Profile": tab_profile,
        " Passion Projects": tab_data_content,
        " Personal Interests": tab_personal,
    })

    final_output = mo.vstack([
        mo.md("# **Aryan Ainesaz Personal Portfolio page**"),
        app_tabs,
    ])

    return (final_output,)


@app.cell
def _(final_output):
    final_output
    return


if __name__ == "__main__":
    app.run()