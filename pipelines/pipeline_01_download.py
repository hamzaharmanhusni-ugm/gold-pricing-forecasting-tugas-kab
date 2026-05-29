"""
Pipeline 01 – Download Data
============================
Download data 6 aset dari Yahoo Finance dan simpan ke CSV.
Output: data/raw/{nama}_close.csv
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
START_DATE = '2020-01-01'
END_DATE   = datetime.today().strftime('%Y-%m-%d')

ASSETS = {
    'Emas':      {'ticker': 'GC=F',    'category': 'Komoditas'},
    'Bitcoin':   {'ticker': 'BTC-USD', 'category': 'Kripto'},
    'Minyak':    {'ticker': 'CL=F',    'category': 'Komoditas'},
    'Apple':     {'ticker': 'AAPL',    'category': 'Saham'},
    'Microsoft': {'ticker': 'MSFT',    'category': 'Saham'},
    'Ethereum':  {'ticker': 'ETH-USD', 'category': 'Kripto'},
}

DATA_DIR   = Path('reports/hasil/result_multi_asset/data/raw')
OUTPUT_DIR = Path('reports/hasil/result_multi_asset')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Download ──────────────────────────────────────────────────────────────────
print('=' * 65)
print('PIPELINE 01 – DOWNLOAD DATA MULTI-ASET')
print('=' * 65)
print(f'Periode  : {START_DATE} s.d. {END_DATE}')
print(f'Aset     : {len(ASSETS)} aset')
print(f'Output   : {DATA_DIR}')
print()

summary = []
for name, info in ASSETS.items():
    print(f'  Mengunduh {name} ({info["ticker"]})...', end=' ', flush=True)
    df = yf.download(info['ticker'], start=START_DATE, end=END_DATE, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(subset=['Close'])

    out_path = DATA_DIR / f'{name}.csv'
    df.to_csv(out_path)

    row = {
        'Aset':     name,
        'Ticker':   info['ticker'],
        'Kategori': info['category'],
        'Baris':    len(df),
        'Mulai':    str(df.index[0].date()),
        'Akhir':    str(df.index[-1].date()),
        'Min Close': round(df['Close'].min(), 2),
        'Max Close': round(df['Close'].max(), 2),
        'File':     str(out_path),
    }
    summary.append(row)
    print(f'{len(df)} baris  [{df.index[0].date()} – {df.index[-1].date()}]')

# ─── Ringkasan ─────────────────────────────────────────────────────────────────
print()
print('RINGKASAN')
print('-' * 65)
df_summary = pd.DataFrame(summary)
print(df_summary[['Aset', 'Ticker', 'Kategori', 'Baris', 'Mulai', 'Akhir']].to_string(index=False))

meta_path = OUTPUT_DIR / 'pipeline_meta.csv'
df_summary.to_csv(meta_path, index=False)
print(f'\nMeta tersimpan: {meta_path}')
print('\n[PIPELINE 01 SELESAI]')
