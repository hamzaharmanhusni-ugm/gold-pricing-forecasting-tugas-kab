"""
Pipeline 04 – ARIMA Model Training (Grid Search AIC)
======================================================
FIX dari notebook lama:
  [BUG 1] trend='c' tidak valid untuk d=1 di statsmodels 0.14
          -> diganti dengan trend='n' dan trend='t'
  [BUG 2] model.fit(method='lbfgs', maxiter=...) -> parameter tidak valid
          -> cukup model.fit() tanpa kwargs tambahan

Grid search: p in [0,1,2,3], d=1, q in [0,1,2,3], trend in ('n','t')
Pilih kombinasi dengan AIC terendah per aset.
Output: models/best_params.json
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
DATA_PROC  = Path('reports/hasil/result_multi_asset/data/processed')
MODELS_DIR = Path('reports/hasil/result_multi_asset/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 10
ARIMA_D   = 1
P_RANGE   = range(0, 4)
Q_RANGE   = range(0, 4)

# PENTING: trend='c' TIDAK VALID untuk d=1 di statsmodels 0.14
# trend='n' = tanpa drift | trend='t' = linear trend (drift)
TRENDS = ('n', 't')

ASSETS = ['Emas', 'Bitcoin', 'Minyak', 'Apple', 'Microsoft', 'Ethereum']

# ─── Grid Search Per Aset ─────────────────────────────────────────────────────
def grid_search_arima(series, p_range, d, q_range, trends, name=''):
    """
    Grid search ARIMA(p,d,q) dengan trend. Pilih AIC terendah.
    Hanya gunakan data training (bukan test set).
    """
    best = None
    total = len(list(p_range)) * len(list(q_range)) * len(trends)
    tried = 0

    for p in p_range:
        for q in q_range:
            for trend in trends:
                tried += 1
                try:
                    model = ARIMA(
                        series,
                        order=(p, d, q),
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=ConvergenceWarning)
                        warnings.filterwarnings('ignore', category=UserWarning)
                        fitted = model.fit()

                    if not np.isfinite(fitted.aic):
                        continue

                    if best is None or fitted.aic < best['aic']:
                        best = {
                            'order': (p, d, q),
                            'trend': trend,
                            'aic':   round(fitted.aic, 2),
                            'bic':   round(fitted.bic, 2),
                        }
                except Exception:
                    continue

    return best

# ─── Main ─────────────────────────────────────────────────────────────────────
print('=' * 70)
print('PIPELINE 04 – ARIMA GRID SEARCH (p,q in 0-3 | d=1 | trend: n,t)')
print('=' * 70)
print(f'Total kombinasi per aset: {len(list(P_RANGE))} x {len(list(Q_RANGE))} x {len(TRENDS)} = '
      f'{len(list(P_RANGE)) * len(list(Q_RANGE)) * len(TRENDS)} model')
print()

best_params = {}

for name in ASSETS:
    csv_path = DATA_PROC / f'{name}_close_clean.csv'
    close = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze('columns')

    # Gunakan hanya data training (kecuali 10 data terakhir)
    train = close.iloc[:-TEST_SIZE]

    print(f'  {name:<12} (train={len(train)} obs) ...', end=' ', flush=True)
    best = grid_search_arima(train, P_RANGE, ARIMA_D, Q_RANGE, TRENDS, name)

    if best is None:
        print('GAGAL - tidak ada model yang konvergen')
        continue

    best_params[name] = best
    print(f'order={best["order"]}  trend={best["trend"]!r}  '
          f'AIC={best["aic"]:>9.2f}  BIC={best["bic"]:>9.2f}  [OK]')

# ─── Simpan Hasil ─────────────────────────────────────────────────────────────
params_path = MODELS_DIR / 'best_params.json'
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

print()
print(f'Parameter tersimpan: {params_path}')
print()
print('REKAP PARAMETER TERBAIK')
print('=' * 70)
print(f'{"Aset":<12} {"Order":<12} {"Trend":<8} {"AIC":>10} {"BIC":>10}')
print('-' * 70)
for name, p in best_params.items():
    print(f'{name:<12} {str(p["order"]):<12} {p["trend"]!r:<8} {p["aic"]:>10.2f} {p["bic"]:>10.2f}')

print()
print(f'Model ditemukan: {len(best_params)}/{len(ASSETS)} aset')
print('[PIPELINE 04 SELESAI]')
