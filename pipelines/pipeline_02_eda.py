"""
Pipeline 02 – Exploratory Data Analysis (EDA)
===============================================
Load data dari CSV, buat 4 jenis visualisasi:
  1. Harga historis + MA-30
  2. Moving Average 7 & 30 hari
  3. Distribusi return harian
  4. Volume perdagangan
Output: charts di reports/hasil/result_multi_asset/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATA_DIR   = Path('reports/hasil/result_multi_asset/data/raw')
OUTPUT_DIR = Path('reports/hasil/result_multi_asset')

ASSETS = {
    'Emas':      {'ticker': 'GC=F',    'category': 'Komoditas', 'color': '#FFD700'},
    'Bitcoin':   {'ticker': 'BTC-USD', 'category': 'Kripto',    'color': '#F7931A'},
    'Minyak':    {'ticker': 'CL=F',    'category': 'Komoditas', 'color': '#2E8B57'},
    'Apple':     {'ticker': 'AAPL',    'category': 'Saham',     'color': '#555555'},
    'Microsoft': {'ticker': 'MSFT',    'category': 'Saham',     'color': '#00A4EF'},
    'Ethereum':  {'ticker': 'ETH-USD', 'category': 'Kripto',    'color': '#627EEA'},
}

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

# ─── Load Data ─────────────────────────────────────────────────────────────────
print('=' * 65)
print('PIPELINE 02 – EXPLORATORY DATA ANALYSIS')
print('=' * 65)

all_data = {}
for name in ASSETS:
    csv_path = DATA_DIR / f'{name}.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    all_data[name] = df

print(f'Data dimuat: {len(all_data)} aset\n')

# ─── Visualisasi 1: Harga Historis + MA-30 ────────────────────────────────────
print('[1/4] Membuat grafik harga historis + MA-30...')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, df) in enumerate(all_data.items()):
    info  = ASSETS[name]
    ax    = axes[i]
    close = df['Close']
    ma30  = close.rolling(30).mean()

    ax.plot(close.index, close, color=info['color'], lw=1, label='Close', alpha=0.8)
    ax.plot(ma30.index,  ma30,  color='black', lw=1.8, ls='--', label='MA-30', alpha=0.7)

    ax.set_title(f'{name} ({info["ticker"]})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (USD)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.text(0.02, 0.95, info['category'], transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Pergerakan Harga Historis & MA-30 (2020–Sekarang)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out1 = OUTPUT_DIR / 'eda_01_historical_prices.png'
plt.savefig(out1, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out1}')

# ─── Visualisasi 2: MA 7 & 30 ─────────────────────────────────────────────────
print('[2/4] Membuat grafik MA 7 & 30 hari...')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, df) in enumerate(all_data.items()):
    info  = ASSETS[name]
    ax    = axes[i]
    close = df['Close']
    ma7   = close.rolling(7).mean()
    ma30  = close.rolling(30).mean()

    ax.plot(close.index, close, color=info['color'], lw=0.8, alpha=0.4, label='Close')
    ax.plot(ma7.index,   ma7,   color='darkorange', lw=1.5, label='MA-7')
    ax.plot(ma30.index,  ma30,  color='navy',       lw=1.8, ls='--', label='MA-30')

    ax.set_title(f'{name} – MA 7 & MA 30', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (USD)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle('Moving Average 7 Hari dan 30 Hari – Semua Aset',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out2 = OUTPUT_DIR / 'eda_02_moving_averages.png'
plt.savefig(out2, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out2}')

# ─── Visualisasi 3: Distribusi Return Harian ──────────────────────────────────
print('[3/4] Membuat grafik distribusi return harian...')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, df) in enumerate(all_data.items()):
    info    = ASSETS[name]
    ax      = axes[i]
    returns = df['Close'].pct_change().dropna() * 100

    ax.hist(returns, bins=60, color=info['color'], alpha=0.75, edgecolor='white')
    ax.axvline(returns.mean(), color='red', lw=2, ls='--',
               label=f'Rata-rata: {returns.mean():.3f}%')
    ax.axvline(0, color='black', lw=1, alpha=0.5)

    stats_txt = f'Std: {returns.std():.3f}%\nMin: {returns.min():.2f}%\nMax: {returns.max():.2f}%'
    ax.text(0.98, 0.95, stats_txt, transform=ax.transAxes, fontsize=8,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title(f'{name} – Distribusi Return Harian', fontsize=12, fontweight='bold')
    ax.set_xlabel('Return Harian (%)')
    ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8)

plt.suptitle('Distribusi Return Harian per Aset', fontsize=15, fontweight='bold')
plt.tight_layout()
out3 = OUTPUT_DIR / 'eda_03_daily_returns.png'
plt.savefig(out3, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out3}')

# ─── Visualisasi 4: Volume Perdagangan ────────────────────────────────────────
print('[4/4] Membuat grafik volume perdagangan...')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, df) in enumerate(all_data.items()):
    info = ASSETS[name]
    ax   = axes[i]
    vol  = df['Volume'].fillna(0)

    ax.bar(vol.index, vol, color=info['color'], alpha=0.6, width=1)
    ax.set_title(f'{name} – Volume Perdagangan', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Volume')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle('Volume Perdagangan Harian – Semua Aset', fontsize=15, fontweight='bold')
plt.tight_layout()
out4 = OUTPUT_DIR / 'eda_04_volume.png'
plt.savefig(out4, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out4}')

# ─── Ringkasan Statistik ───────────────────────────────────────────────────────
print()
print('RINGKASAN STATISTIK HARGA CLOSE (USD)')
print('=' * 70)
stats = {}
for name, df in all_data.items():
    s = df['Close'].describe()
    stats[name] = s
df_stats = pd.DataFrame(stats).round(2)
print(df_stats.to_string())

print('\n[PIPELINE 02 SELESAI]')
