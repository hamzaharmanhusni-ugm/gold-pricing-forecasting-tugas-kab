"""
Pipeline 05 – Model Evaluation (Walk-Forward)
==============================================
FIX dari notebook lama:
  [BUG 3] result.append() deprecated di statsmodels 0.14
          -> Diganti expanding window refit (clean dan reliable)

Untuk tiap aset:
  1. Load data + best_params.json
  2. Walk-forward one-step-ahead prediction (10 langkah)
  3. Hitung MSE, MAE, RMSE, Dstat
  4. Buat chart Aktual vs Prediksi
Output: metrics_summary.csv, eval_actual_vs_predicted.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import warnings
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATA_PROC  = Path('reports/hasil/result_multi_asset/data/processed')
MODELS_DIR = Path('reports/hasil/result_multi_asset/models')
OUTPUT_DIR = Path('reports/hasil/result_multi_asset')
TEST_SIZE  = 10

ASSETS = {
    'Emas':      {'color': '#FFD700'},
    'Bitcoin':   {'color': '#F7931A'},
    'Minyak':    {'color': '#2E8B57'},
    'Apple':     {'color': '#555555'},
    'Microsoft': {'color': '#00A4EF'},
    'Ethereum':  {'color': '#627EEA'},
}

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

# ─── Load best params ─────────────────────────────────────────────────────────
with open(MODELS_DIR / 'best_params.json') as f:
    best_params = json.load(f)

# ─── Walk-Forward Prediction ──────────────────────────────────────────────────
def walk_forward(close, order, trend, test_size):
    """
    Expanding window walk-forward: setiap langkah refit model
    dengan menambahkan 1 data aktual ke training.
    Lebih lambat tapi tidak bergantung pada API deprecated.
    """
    n_train = len(close) - test_size
    predictions = []

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        for i in range(test_size):
            train_i = close.iloc[:n_train + i]
            model   = ARIMA(train_i, order=order, trend=trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fitted  = model.fit()
            fc      = fitted.forecast(steps=1)
            predictions.append(float(fc.iloc[0]))

    test_index   = close.index[-test_size:]
    return pd.Series(predictions, index=test_index)

# ─── Evaluasi ─────────────────────────────────────────────────────────────────
def hitung_dstat(actual, pred):
    a = np.array(actual)
    p = np.array(pred)
    return float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p)))) * 100

print('=' * 75)
print('PIPELINE 05 – WALK-FORWARD EVALUATION (10 langkah per aset)')
print('=' * 75)
print()

metrics_all = {}

for name in ASSETS:
    params = best_params[name]
    order  = tuple(params['order'])
    trend  = params['trend']

    csv_path = DATA_PROC / f'{name}_close_clean.csv'
    close = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze('columns')
    test  = close.iloc[-TEST_SIZE:]

    print(f'  {name:<12} order={order} trend={trend!r}  ...', end=' ', flush=True)
    forecast_test = walk_forward(close, order, trend, TEST_SIZE)

    mse   = mean_squared_error(test.values, forecast_test.values)
    mae   = mean_absolute_error(test.values, forecast_test.values)
    rmse  = np.sqrt(mse)
    dstat = hitung_dstat(test.values, forecast_test.values)

    metrics_all[name] = {
        'Order':      str(order),
        'Trend':      trend,
        'AIC':        params['aic'],
        'MSE':        round(mse,  2),
        'MAE':        round(mae,  2),
        'RMSE':       round(rmse, 2),
        'Dstat (%)':  round(dstat, 1),
    }
    print(f'RMSE={rmse:>8.2f}  MAE={mae:>8.2f}  Dstat={dstat:>5.1f}%')

    # Simpan prediksi per aset untuk keperluan debug/audit
    df_cmp = pd.DataFrame({'Aktual': test, 'Prediksi': forecast_test})
    df_cmp.to_csv(MODELS_DIR / f'{name}_test_predictions.csv')

# ─── Tabel Metrik ─────────────────────────────────────────────────────────────
print()
df_metrics = pd.DataFrame(metrics_all).T
df_metrics.index.name = 'Aset'

print('TABEL PERBANDINGAN METRIK EVALUASI')
print('=' * 75)
print(df_metrics.to_string())

metrics_path = OUTPUT_DIR / 'metrics_summary.csv'
df_metrics.to_csv(metrics_path)
print(f'\nMetrik tersimpan: {metrics_path}')

print()
print('PERINGKAT DSTAT (lebih tinggi = lebih baik):')
print(df_metrics[['Dstat (%)']].sort_values('Dstat (%)', ascending=False).to_string())

# ─── Visualisasi Aktual vs Prediksi ───────────────────────────────────────────
print()
print('Membuat grafik Aktual vs Prediksi...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, name in enumerate(ASSETS):
    ax    = axes[i]
    color = ASSETS[name]['color']
    res   = metrics_all[name]

    pred_path = MODELS_DIR / f'{name}_test_predictions.csv'
    df_cmp = pd.read_csv(pred_path, index_col=0, parse_dates=True)

    ax.plot(df_cmp.index, df_cmp['Aktual'],   'o-', color='steelblue', lw=2,
            markersize=6, label='Aktual')
    ax.plot(df_cmp.index, df_cmp['Prediksi'], 's--', color=color,      lw=2,
            markersize=6, label='Prediksi')

    ax.set_title(f'{name}  |  Dstat={res["Dstat (%)"]:.1f}%  |  MAE={res["MAE"]:.2f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (USD)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.suptitle('Aktual vs Prediksi – 10 Data Uji Terakhir (Semua Aset)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out = OUTPUT_DIR / 'eval_actual_vs_predicted.png'
plt.savefig(out, bbox_inches='tight', dpi=100)
plt.close()
print(f'Tersimpan: {out}')

# ─── Visualisasi Perbandingan Metrik ──────────────────────────────────────────
print('Membuat grafik perbandingan metrik...')

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
names   = list(ASSETS.keys())
colors  = [ASSETS[n]['color'] for n in names]
rmses   = [metrics_all[n]['RMSE']       for n in names]
maes    = [metrics_all[n]['MAE']        for n in names]
dstats  = [metrics_all[n]['Dstat (%)']  for n in names]

axes[0].barh(names, rmses,  color=colors, alpha=0.85, edgecolor='gray')
axes[0].set_title('RMSE per Aset\n(lebih kecil = lebih baik)', fontweight='bold')
axes[0].set_xlabel('RMSE (USD)')
for i, v in enumerate(rmses):
    axes[0].text(v * 1.01, i, f'{v:.1f}', va='center', fontsize=9)

axes[1].barh(names, maes, color=colors, alpha=0.85, edgecolor='gray')
axes[1].set_title('MAE per Aset\n(lebih kecil = lebih baik)', fontweight='bold')
axes[1].set_xlabel('MAE (USD)')
for i, v in enumerate(maes):
    axes[1].text(v * 1.01, i, f'{v:.1f}', va='center', fontsize=9)

axes[2].barh(names, dstats, color=colors, alpha=0.85, edgecolor='gray')
axes[2].axvline(50, color='red', ls='--', lw=1.5, alpha=0.7, label='50% (random)')
axes[2].set_title('Dstat per Aset (%)\n(lebih tinggi = lebih baik)', fontweight='bold')
axes[2].set_xlabel('Dstat (%)')
axes[2].set_xlim(0, 120)
axes[2].legend(fontsize=8)
for i, v in enumerate(dstats):
    axes[2].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

plt.suptitle('Perbandingan Metrik Evaluasi ARIMA – 6 Aset',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out2 = OUTPUT_DIR / 'eval_metrics_comparison.png'
plt.savefig(out2, bbox_inches='tight', dpi=100)
plt.close()
print(f'Tersimpan: {out2}')

print('\n[PIPELINE 05 SELESAI]')
