"""
Pipeline 06 – Future Forecasting (30 Hari ke Depan) + Combined Eval Chart
==========================================================================
Perubahan dari versi sebelumnya:
  [FIX] Individual chart diganti dengan combined 2-panel chart:
        Panel atas : historis (60 hari) + overlay actual vs predicted (test)
        Panel bawah: zoom ke test period + 30-day forecast + CI
  [ADD] D+1 dan D+30 annotasi
  [ADD] Combined overview 2x3 grid tetap ada (forecast_chart_all.png)

Output baru:
  charts_individual/combined_{nama}.png  <- chart 2-panel per aset (BARU)
  forecast_chart_all.png                 <- overview 2x3 (tetap)
  forecast_30days_all.csv                <- tabel prediksi
  model_summary.txt                      <- ringkasan teks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import warnings
from datetime import datetime
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATA_PROC     = Path('reports/hasil/result_multi_asset/data/processed')
MODELS_DIR    = Path('reports/hasil/result_multi_asset/models')
OUTPUT_DIR    = Path('reports/hasil/result_multi_asset')
CHARTS_DIR    = OUTPUT_DIR / 'charts_individual'
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_DAYS = 30
LOOKBACK_HIST = 60   # hari historis di panel atas (bukan test)
TEST_SIZE     = 10

ASSETS = {
    'Emas':      {'ticker': 'GC=F',    'category': 'Komoditas', 'color': '#C9A800', 'freq': 'B'},
    'Bitcoin':   {'ticker': 'BTC-USD', 'category': 'Kripto',    'color': '#F7931A', 'freq': 'D'},
    'Minyak':    {'ticker': 'CL=F',    'category': 'Komoditas', 'color': '#2E8B57', 'freq': 'B'},
    'Apple':     {'ticker': 'AAPL',    'category': 'Saham',     'color': '#555555', 'freq': 'B'},
    'Microsoft': {'ticker': 'MSFT',    'category': 'Saham',     'color': '#00A4EF', 'freq': 'B'},
    'Ethereum':  {'ticker': 'ETH-USD', 'category': 'Kripto',    'color': '#627EEA', 'freq': 'D'},
}

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

# ─── Load best params ─────────────────────────────────────────────────────────
with open(MODELS_DIR / 'best_params.json') as f:
    best_params = json.load(f)

# ─── Forecast Per Aset ────────────────────────────────────────────────────────
print('=' * 72)
print(f'PIPELINE 06 – FORECAST {FORECAST_DAYS} HARI KE DEPAN + COMBINED CHART')
print('=' * 72)
print()

future_forecasts = {}

for name, info in ASSETS.items():
    params = best_params[name]
    order  = tuple(params['order'])
    trend  = params['trend']
    freq   = info['freq']

    csv_path = DATA_PROC / f'{name}_close_clean.csv'
    close = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze('columns')

    print(f'  {name:<12} order={order} trend={trend!r}  ...', end=' ', flush=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        model = ARIMA(close, order=order, trend=trend,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        fitted = model.fit()

    # Buat tanggal masa depan
    last_date = close.index[-1]
    if freq == 'B':
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=FORECAST_DAYS
        )
    else:
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=FORECAST_DAYS,
            freq=freq
        )

    fc     = fitted.get_forecast(steps=FORECAST_DAYS)
    fc_val = fc.predicted_mean
    fc_ci  = fc.conf_int(alpha=0.05)

    fc_val.index = future_dates
    fc_ci.index  = future_dates

    forecast_df = pd.DataFrame({
        'Prediksi':    fc_val.values,
        'CI_Bawah_95': fc_ci.iloc[:, 0].values,
        'CI_Atas_95':  fc_ci.iloc[:, 1].values,
    }, index=future_dates)
    forecast_df.index.name = 'Tanggal'

    last_price = float(close.iloc[-1])
    d1_price   = float(forecast_df['Prediksi'].iloc[0])
    d30_price  = float(forecast_df['Prediksi'].iloc[-1])
    change_pct = (d30_price - last_price) / last_price * 100

    future_forecasts[name] = {
        'forecast_df': forecast_df,
        'last_price':  last_price,
        'close':       close,
        'd1_price':    d1_price,
        'd30_price':   d30_price,
        'change_pct':  change_pct,
    }

    print(f'D+1={d1_price:>9.2f}  D+30={d30_price:>9.2f}  ({change_pct:>+.2f}%)')

# ─── Ringkasan ────────────────────────────────────────────────────────────────
print()
print('RINGKASAN PREDIKSI 30 HARI')
print('=' * 78)
print(f'  {"Aset":<12} {"Sekarang":>12} {"D+1":>10} {"D+15":>10} {"D+30":>10} {"Perubahan":>11}')
print('  ' + '-' * 65)
for name, fdata in future_forecasts.items():
    fc  = fdata['forecast_df']
    cur = fdata['last_price']
    d1  = float(fc['Prediksi'].iloc[0])
    d15 = float(fc['Prediksi'].iloc[14]) if len(fc) >= 15 else float(fc['Prediksi'].iloc[-1])
    d30 = float(fc['Prediksi'].iloc[-1])
    chg = (d30 - cur) / cur * 100
    print(f'  {name:<12} {cur:>12.2f} {d1:>10.2f} {d15:>10.2f} {d30:>10.2f} {chg:>+10.2f}%')

# ─── Simpan Forecast CSV ──────────────────────────────────────────────────────
dfs_list = []
for name, fdata in future_forecasts.items():
    tmp = fdata['forecast_df'].copy().reset_index()
    tmp['Aset']   = name
    tmp['Ticker'] = ASSETS[name]['ticker']
    dfs_list.append(tmp)

df_all_fc = pd.concat(dfs_list, ignore_index=True)
fc_path = OUTPUT_DIR / 'forecast_30days_all.csv'
df_all_fc.to_csv(fc_path, index=False)
print(f'\nForecast CSV tersimpan: {fc_path}')

# ─── Overview 2x3 Grid (forecast saja) ───────────────────────────────────────
print()
print('[1/3] Membuat overview chart 2x3 (forecast_chart_all.png)...')

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, (name, fdata) in enumerate(future_forecasts.items()):
    ax          = axes[i]
    info        = ASSETS[name]
    close       = fdata['close']
    forecast_df = fdata['forecast_df']
    hist        = close.iloc[-60:]

    ax.plot(hist.index, hist.values,
            color=info['color'], lw=1.8, alpha=0.9, label='Historis (60h)')
    ax.plot(forecast_df.index, forecast_df['Prediksi'],
            color='black', lw=2.5, ls='--', label=f'Prediksi {FORECAST_DAYS}h')
    ax.fill_between(forecast_df.index,
                    forecast_df['CI_Bawah_95'],
                    forecast_df['CI_Atas_95'],
                    alpha=0.18, color='steelblue', label='CI 95%')
    ax.axvline(close.index[-1], color='red', lw=1.5, ls=':', alpha=0.8)

    # Annotasi D+30
    ax.annotate(
        f'D+30\n{fdata["d30_price"]:.0f}',
        xy=(forecast_df.index[-1], fdata['d30_price']),
        xytext=(10, 5), textcoords='offset points',
        fontsize=8, color='black',
        arrowprops=dict(arrowstyle='->', color='black', lw=1)
    )

    ax.set_title(f'{name} ({info["ticker"]}) | {fdata["change_pct"]:+.2f}% (30h)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (USD)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.suptitle(f'Prediksi {FORECAST_DAYS} Hari ke Depan – Semua Aset (CI 95%)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_overview = OUTPUT_DIR / 'forecast_chart_all.png'
plt.savefig(out_overview, bbox_inches='tight', dpi=100)
plt.close()
print(f'   Tersimpan: {out_overview}')

# ─── Combined Chart Per Aset (BARU) ──────────────────────────────────────────
# Panel atas : historis + actual (test) vs predicted (test) – menunjukkan akurasi
# Panel bawah: zoom ke test period + 30-day forecast + CI – menunjukkan prediksi
# ─────────────────────────────────────────────────────────────────────────────
print('[2/3] Membuat combined eval+forecast chart per aset...')

for name, fdata in future_forecasts.items():
    info        = ASSETS[name]
    close       = fdata['close']
    forecast_df = fdata['forecast_df']

    # Load test predictions dari pipeline_05
    pred_path = MODELS_DIR / f'{name}_test_predictions.csv'
    df_test   = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    test_actual = df_test['Aktual']
    test_pred   = df_test['Prediksi']

    # Hitung metrik sederhana untuk anotasi
    mae   = float((test_actual - test_pred).abs().mean())
    dstat = float(
        (np.sign(test_actual.diff().dropna().values) ==
         np.sign(test_pred.diff().dropna().values)).mean()
    ) * 100

    # Data historis: sebelum test period
    hist = close.iloc[-(LOOKBACK_HIST + TEST_SIZE):-TEST_SIZE]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10),
                                   gridspec_kw={'height_ratios': [1.4, 1]})

    # ── PANEL ATAS: Historis + Test Actual vs Predicted ──────────────────────
    ax1.plot(hist.index, hist.values,
             color=info['color'], lw=1.5, alpha=0.7, label=f'Historis ({LOOKBACK_HIST}h)')
    ax1.plot(test_actual.index, test_actual.values,
             'o-', color='steelblue', lw=2.2, markersize=6, label='Aktual (test)')
    ax1.plot(test_pred.index, test_pred.values,
             's--', color='tomato', lw=2.2, markersize=6, label='Prediksi ARIMA (test)')

    ax1.axvspan(test_actual.index[0], test_actual.index[-1],
                alpha=0.08, color='steelblue', label='_test_region')

    # Label akurasi test
    ax1.text(0.01, 0.97,
             f'Test Period:  MAE = {mae:.2f} USD  |  Dstat = {dstat:.1f}%',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax1.set_title(f'{name} ({info["ticker"]}) – Panel 1: Akurasi Model pada Data Uji',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Harga (USD)')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

    # ── PANEL BAWAH: Test Period + 30-Day Forecast ───────────────────────────
    # Gabung: data aktual test + prediksi future
    ax2.plot(test_actual.index, test_actual.values,
             'o-', color='steelblue', lw=2, markersize=6, label='Aktual (test)')
    ax2.plot(test_pred.index, test_pred.values,
             's--', color='tomato', lw=1.8, markersize=5, label='Prediksi (test)')

    # Titik peralihan historis → forecast
    # Tambahkan 1 titik terakhir actual ke forecast agar garis sambung
    last_actual_date  = test_actual.index[-1]
    last_actual_price = float(test_actual.iloc[-1])
    link_dates  = [last_actual_date] + list(forecast_df.index)
    link_prices = [last_actual_price] + list(forecast_df['Prediksi'].values)

    ax2.plot(forecast_df.index, forecast_df['Prediksi'],
             'd-', color='black', lw=2.5, markersize=5,
             label=f'Forecast {FORECAST_DAYS} hari ke depan')
    ax2.fill_between(forecast_df.index,
                     forecast_df['CI_Bawah_95'],
                     forecast_df['CI_Atas_95'],
                     alpha=0.18, color='steelblue', label='CI 95%')

    # Garis pemisah antara test dan forecast
    ax2.axvline(close.index[-1], color='red', lw=2, ls=':', alpha=0.9,
                label='Batas historis / forecast')
    ax2.axvspan(test_actual.index[0], close.index[-1],
                alpha=0.05, color='steelblue')

    # Annotasi D+1 dan D+30
    d1_date  = forecast_df.index[0]
    d30_date = forecast_df.index[-1]
    d1_val   = float(forecast_df['Prediksi'].iloc[0])
    d30_val  = float(forecast_df['Prediksi'].iloc[-1])
    chg      = (d30_val - last_actual_price) / last_actual_price * 100

    ax2.annotate(
        f'D+1\n{d1_val:.2f}',
        xy=(d1_date, d1_val), xytext=(0, -28), textcoords='offset points',
        fontsize=9, ha='center', color='black',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2)
    )
    ax2.annotate(
        f'D+{FORECAST_DAYS}\n{d30_val:.2f}',
        xy=(d30_date, d30_val), xytext=(0, 22), textcoords='offset points',
        fontsize=9, ha='center', color='black',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2)
    )

    ax2.text(0.98, 0.97,
             f'Sekarang : {last_actual_price:.2f} USD\n'
             f'D+{FORECAST_DAYS}      : {d30_val:.2f} USD\n'
             f'Perubahan: {chg:+.2f}%',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax2.set_title(f'Panel 2: Forecast {FORECAST_DAYS} Hari ke Depan dengan CI 95%',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Tanggal')
    ax2.set_ylabel('Harga (USD)')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

    plt.suptitle(f'{name} ({info["ticker"]}) – Evaluasi Akurasi & Prediksi 30 Hari',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_combined = CHARTS_DIR / f'combined_{name}.png'
    plt.savefig(out_combined, bbox_inches='tight', dpi=100)
    plt.close()
    print(f'   {name}: {out_combined}')

# ─── Overview Combined 2x3 (untuk notebook) ──────────────────────────────────
print('[3/3] Membuat overview combined chart 2x3 (combined_chart_all.png)...')

fig, axes = plt.subplots(2, 3, figsize=(21, 12))
axes = axes.flatten()

import matplotlib.image as mpimg
for i, name in enumerate(ASSETS):
    img_path = CHARTS_DIR / f'combined_{name}.png'
    img = mpimg.imread(str(img_path))
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(name, fontsize=13, fontweight='bold', pad=6)

plt.suptitle('Evaluasi Akurasi Model & Prediksi 30 Hari – Semua Aset',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_comb_all = OUTPUT_DIR / 'combined_chart_all.png'
plt.savefig(out_comb_all, bbox_inches='tight', dpi=85)
plt.close()
print(f'   Tersimpan: {out_comb_all}')

# ─── Simpan Ringkasan Teks ────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / 'model_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('RINGKASAN MODEL ARIMA MULTI-ASET\n')
    f.write(f'Tanggal Analisis : {datetime.today().strftime("%Y-%m-%d %H:%M")}\n')
    f.write(f'ARIMA d=1, grid p,q=[0-3], trend=[n,t]\n\n')

    f.write('PARAMETER TERBAIK (per aset)\n')
    f.write('=' * 60 + '\n')
    for name, p in best_params.items():
        f.write(f'{name:<12} order={p["order"]} trend={p["trend"]}  '
                f'AIC={p["aic"]} BIC={p["bic"]}\n')

    f.write('\nPREDIKSI 30 HARI KE DEPAN\n')
    f.write('=' * 60 + '\n')
    for name, fdata in future_forecasts.items():
        fc  = fdata['forecast_df']
        cur = fdata['last_price']
        d30 = float(fc['Prediksi'].iloc[-1])
        ci_low  = float(fc['CI_Bawah_95'].iloc[-1])
        ci_high = float(fc['CI_Atas_95'].iloc[-1])
        f.write(f'{name:<12} Sekarang={cur:.2f}  D+30={d30:.2f}  '
                f'CI=[{ci_low:.2f}, {ci_high:.2f}]\n')

print(f'\nRingkasan teks tersimpan: {summary_path}')
print(f'\n[PIPELINE 06 SELESAI]')
print(f'Output ada di: {OUTPUT_DIR}')
