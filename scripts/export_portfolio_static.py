#!/usr/bin/env python3
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import yfinance as yf
from pathlib import Path

OUT = Path('docs')
OUT.mkdir(parents=True, exist_ok=True)

def load_data():
    csv_url = "https://gist.githubusercontent.com/DrAYim/80393243abdbb4bfe3b45fef58e8d3c8/raw/ed5cfd9f210bf80cb59a5f420bf8f2b88a9c2dcd/sp500_ZScore_AvgCostofDebt.csv"
    df = pd.read_csv(csv_url)
    df = df.dropna(subset=['AvgCost_of_Debt', 'Z_Score_lag', 'Sector_Key'])
    df = df[df['AvgCost_of_Debt'] < 5]
    df['Debt_Cost_Percent'] = df['AvgCost_of_Debt'] * 100
    df['Market_Cap_B'] = df['Market_Cap'] / 1e9
    return df

def make_db_fig(df):
    df_scatter = df.copy()
    positive_caps = df_scatter.loc[df_scatter['Market_Cap'] > 0, 'Market_Cap']
    min_positive_cap = float(positive_caps.min()) if not positive_caps.empty else 1.0
    df_scatter['Market_Cap_Render'] = df_scatter['Market_Cap'].clip(lower=min_positive_cap * 0.2)

    title = 'Z-Score vs Avg Cost of Debt'
    fig = px.scatter(
        df_scatter,
        x='Z_Score_lag', y='Debt_Cost_Percent', color='Sector_Key', size='Market_Cap_Render',
        hover_name='Name', hover_data=['Ticker', 'Market_Cap'], title=title, opacity=0.7
    )
    fig.update_layout(height=600)
    fig.add_vline(x=1.81, line_dash='dash', line_color='red')
    fig.add_vline(x=2.99, line_dash='dash', line_color='green')
    return fig

def make_box_fig(df):
    fig = px.box(df, x='Sector_Key', y='Debt_Cost_Percent', color='Sector_Key', title='Cost of Debt by Sector')
    fig.update_layout(height=450)
    return fig

def make_risk_violin(df):
    distress_threshold = 1.81
    safe_threshold = 2.99
    df2 = df.copy()
    df2['Risk_Zone'] = np.select([
        df2['Z_Score_lag'] < distress_threshold,
        df2['Z_Score_lag'] > safe_threshold], ['Distress', 'Safe'], default='Intermediate')
    fig = px.violin(df2, x='Risk_Zone', y='Debt_Cost_Percent', color='Risk_Zone', box=True, title='Debt Cost by Risk Zone')
    fig.update_layout(height=450)
    return fig

def make_heatmap(df):
    cols = ['Z_Score_lag', 'Debt_Cost_Percent', 'Market_Cap_B']
    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu', zmin=-1, zmax=1, title='Correlation Map')
    fig.update_layout(height=450)
    return fig

def make_sp_figs():
    try:
        gspc = yf.Ticker('^GSPC').history(period='20y')[['Close']].reset_index()
        gspc.columns = ['Date', 'Close']
        gspc['Date'] = pd.to_datetime(gspc['Date']).dt.tz_convert(None)
        one_year = gspc[gspc['Date'] >= (gspc['Date'].max() - pd.DateOffset(years=1))]
    except Exception:
        gspc = pd.DataFrame({'Date': [], 'Close': []})
        one_year = gspc

    fig_recent = px.line(one_year, x='Date', y='Close', title='S&P 500 — Last 1 Year')
    fig_recent.update_layout(height=400)
    fig_hist = px.line(gspc, x='Date', y='Close', title='S&P 500 — 20 Years')
    fig_hist.update_layout(height=400)
    return fig_recent, fig_hist

def make_travel_figs():
    travel_data = pd.DataFrame({
        'City': ['England', 'Italy', 'France', 'Spain', 'Turkey', 'Portugal', 'Netherlands', 'Wales', 'Jamaica', 'Iran'],
        'Lat': [51.5074, 41.9028, 48.8566, 40.4168, 39.9334, 38.7223, 52.3676, 51.4816, 17.9714, 35.6892],
        'Lon': [-0.1278, 12.4964, 2.3522, -3.7038, 32.8597, 32.8597, 4.9041, -3.1791, -76.7922, 51.3890],
        'Visit_Year_str': ['2006','2018','2016','2012','2022','2017','2015','2019','2009','2013']
    })
    fig = px.scatter_geo(travel_data, lat='Lat', lon='Lon', hover_name='City', color='Visit_Year_str', projection='natural earth', title='Travel Footprint')
    fig.update_layout(height=450)
    return fig

def make_grades_chart():
    grades = {
        'Financial Institutions': 85.8,
        'Introductory Financial Accounting': 87.5,
        'Principles of Economics': 79,
        'Introductory Management Accounting': 92.7,
        'Research Methods': 76,
        'Principles of Taxation': 94.8,
        'Introduction to Data Science and AI Tools': 73,
        'Introduction to Finance': 86.8,
    }
    fig = px.bar(x=list(grades.values()), y=list(grades.keys()), orientation='h', text_auto=True, title='First Year Module Grades (%)')
    fig.update_layout(height=450)
    return fig

def assemble_html(snippets):
    head = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Static export — Wk04x_MyPortfolio</title><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"><style>body{font-family:Inter,system-ui,Arial;background:#071827;color:#e6eef8;margin:0;padding:28px} .wrap{max-width:1100px;margin:0 auto} h1{margin:0 0 14px} .section{background:#071a28;padding:20px;border-radius:10px;margin-bottom:18px}</style></head><body><div class="wrap"><h1>Wk04x_MyPortfolio — Static export</h1>'''
    foot = '</div></body></html>'
    body = ''.join(f'<section class="section">{s}</section>' for s in snippets)
    html = head + body + foot
    p = OUT / 'Wk04x_MyPortfolio_static.html'
    p.write_text(html, encoding='utf-8')
    print('Wrote', p)

def main():
    df = load_data()
    db = make_db_fig(df)
    box = make_box_fig(df)
    violin = make_risk_violin(df)
    heat = make_heatmap(df)
    sp_recent, sp_hist = make_sp_figs()
    travel = make_travel_figs()
    grades = make_grades_chart()

    snippets = []
    for title, fig in [
        ('Z-Score vs Avg Cost of Debt', db),
        ('Cost of Debt by Sector', box),
        ('Debt Cost by Risk Zone', violin),
        ('Correlation Map', heat),
        ('S&P 500 — Last 1 Year', sp_recent),
        ('S&P 500 — 20 Years', sp_hist),
        ('Travel Footprint', travel),
        ('Education Grades', grades),
    ]:
        html_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        snippets.append(f'<h2>{title}</h2>' + html_div)

    assemble_html(snippets)

if __name__ == '__main__':
    main()
