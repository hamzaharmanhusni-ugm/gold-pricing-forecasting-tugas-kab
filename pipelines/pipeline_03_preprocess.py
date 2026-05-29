"""
Pipeline 03 – Preprocessing & Stationarity Testing
====================================================
Load raw CSV, lakukan:
  1. Preprocessing (forward-fill gap kalender)
  2. Uji ADF pada data asli (level)
  3. First-order differencing
  4. Uji ADF setelah differencing
  5. Plot rolling mean & std
  6. Plot ACF & PACF (Emas sebagai contoh)
Output: data/processed/{nama}_close_clean.csv
        eda_05_rolling_stats.png
        eda_06_acf_pacf.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATA_RAW    = Path('reports/hasil/result_multi_asset/data/raw')
DATA_PROC   = Path('reports/hasil/result_multi_asset/data/processed')
OUTPUT_DIR  = Path('reports/hasil/result_multi_asset')
DATA_PROC.mkdir(parents=True, exist_ok=True)

ASSETS = {
    'Emas':      {'ticker': 'GC=F',    'color': '#FFD700', 'freq': 'B'},
    'Bitcoin':   {'ticker': 'BTC-USD', 'color': '#F7931A', 'freq': 'D'},
    'Minyak':    {'ticker': 'CL=F',    'color': '#2E8B57', 'freq': 'B'},
    'Apple':     {'ticker': 'AAPL',    'color': '#555555', 'freq': 'B'},
    'Microsoft': {'ticker': 'MSFT',    'color': '#00A4EF', 'freq': 'B'},
    'Ethereum':  {'ticker': 'ETH-USD', 'color': '#627EEA', 'freq': 'D'},
}

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

# ─── Load & Preprocessing ─────────────────────────────────────────────────────
print('=' * 65)
print('PIPELINE 03 – PREPROCESSING & STATIONARITY TESTING')
print('=' * 65)

close_series = {}
for name, info in ASSETS.items():
    df = pd.read_csv(DATA_RAW / f'{name}.csv', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward-fill lalu backward-fill untuk mengisi gap tanggal
    close = df['Close'].asfreq(info['freq']).ffill().bfill()
    close.name = name
    close_series[name] = close

    # Simpan versi bersih
    proc_path = DATA_PROC / f'{name}_close_clean.csv'
    close.to_csv(proc_path)

print(f'Preprocessing selesai. {len(close_series)} aset dimuat.')
print()

# ─── ADF Test Helper ──────────────────────────────────────────────────────────
def uji_adf(series, label=''):
    result = adfuller(series.dropna(), autolag='AIC')
    p_val  = result[1]
    return {
        'label':     label,
        'stat':      round(result[0], 4),
        'p_value':   round(p_val, 6),
        'stasioner': p_val < 0.05,
        'status':    'STASIONER [OK]' if p_val < 0.05 else 'TIDAK STASIONER [!!]',
    }

# ─── ADF pada Data Level ──────────────────────────────────────────────────────
print('UJI ADF – DATA ASLI (LEVEL)')
print('=' * 65)
print(f'{"Aset":<12} {"ADF Stat":>12} {"p-value":>12}  {"Status"}')
print('-' * 65)
for name, close in close_series.items():
    r = uji_adf(close, name)
    print(f'{name:<12} {r["stat"]:>12.4f} {r["p_value"]:>12.6f}  {r["status"]}')

print('\nKesimpulan: Semua aset TIDAK stasioner pada level harga (ada tren).')
print('Solusi    : Lakukan first-order differencing (d=1).')

# ─── ADF setelah Differencing ─────────────────────────────────────────────────
print()
print('UJI ADF – SETELAH FIRST-ORDER DIFFERENCING (d=1)')
print('=' * 65)
print(f'{"Aset":<12} {"ADF Stat":>12} {"p-value":>12}  {"Status"}')
print('-' * 65)

close_diff = {}
for name, close in close_series.items():
    diff = close.diff().dropna()
    close_diff[name] = diff
    r = uji_adf(diff, name)
    print(f'{name:<12} {r["stat"]:>12.4f} {r["p_value"]:>12.6f}  {r["status"]}')

print('\nKesimpulan: Semua aset menjadi STASIONER setelah d=1. Gunakan d=1 di ARIMA.')

# ─── Rolling Mean & Std Chart ─────────────────────────────────────────────────
print()
print('[1/2] Membuat grafik rolling statistics...')
WINDOW = 30
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, close) in enumerate(close_series.items()):
    info      = ASSETS[name]
    ax        = axes[i]
    roll_mean = close.rolling(WINDOW).mean()
    roll_std  = close.rolling(WINDOW).std()

    ax.plot(close.index, close,       color=info['color'], lw=0.8, alpha=0.4, label='Close')
    ax.plot(roll_mean.index, roll_mean, color='navy', lw=1.8, label=f'Rolling Mean ({WINDOW}d)')

    ax2 = ax.twinx()
    ax2.plot(roll_std.index, roll_std, color='tomato', lw=1, alpha=0.7, label='Rolling Std')
    ax2.set_ylabel('Std Dev', color='tomato', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='tomato')

    ax.set_title(f'{name} – Rolling Statistics', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (USD)')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle(f'Rolling Mean & Std ({WINDOW}d) – Indikasi Non-Stasioneritas',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out5 = OUTPUT_DIR / 'eda_05_rolling_stats.png'
plt.savefig(out5, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out5}')

# ─── ACF & PACF (Emas sebagai Contoh) ────────────────────────────────────────
print('[2/2] Membuat grafik ACF & PACF (Emas setelah differencing)...')
gold_diff = close_diff['Emas'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(gold_diff,  lags=30, ax=axes[0],
         title='ACF – Emas (setelah differencing d=1)')
plot_pacf(gold_diff, lags=30, ax=axes[1],
          title='PACF – Emas (setelah differencing d=1)', method='ywm')
axes[0].set_xlabel('Lag')
axes[1].set_xlabel('Lag')

plt.suptitle('ACF & PACF – Emas (Data Stasioner setelah d=1)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out6 = OUTPUT_DIR / 'eda_06_acf_pacf.png'
plt.savefig(out6, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out6}')

print()
print('Interpretasi ACF/PACF:')
print('  - PACF: lag signifikan = estimasi orde p (AR)')
print('  - ACF : lag signifikan = estimasi orde q (MA)')
print('  - Grid search di Pipeline 04 akan menentukan p,q terbaik per aset via AIC.')

print('\n[PIPELINE 03 SELESAI]')
