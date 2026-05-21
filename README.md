# Laporan Proyek: Prediksi Harga Emas (ARIMA)

Notebook ini membangun model peramalan harga emas (GC=F) menggunakan ARIMA dan menghasilkan output analisis serta grafik. Laporan ini merangkum alur kerja dan hasil utama dengan bahasa sederhana.

## Ringkasan Cepat

- **Data**: Gold Futures (GC=F) dari Yahoo Finance, 2020-01-02 sampai 2026-05-20.
- **Jumlah data**: 1606 hari perdagangan.
- **Model**: ARIMA(5,1,0).
- **Hasil evaluasi**: RMSE 89.21 USD, MAE 71.51 USD, Dstat 55.56%.
- **Output**: CSV, TXT, dan 3 grafik disimpan di folder `reports/hasil`.

## Tujuan Proyek

1. Mengunduh data historis harga emas dari Yahoo Finance.
2. Melakukan eksplorasi data dan analisis stasioneritas.
3. Melatih model ARIMA dan mengevaluasi prediksinya.
4. Membuat prediksi 30 hari ke depan beserta confidence interval.
5. Menyimpan hasil untuk keperluan laporan.

## Dataset

- **Ticker**: GC=F (Gold Futures)
- **Interval**: Harian
- **Periode**: 2020-01-02 s.d. 2026-05-20

**Kolom utama**:
- Open, High, Low, Close, Volume
- Target prediksi: **Close**

## Alur Analisis (Singkat dan Mudah Dipahami)

1. **Download data** dari Yahoo Finance.
2. **EDA**: grafik tren harga, moving average, return harian, distribusi harga, dan volume.
3. **Preprocessing**: pastikan index datetime, urutkan data, dan ambil kolom Close.
4. **Uji stasioneritas** (ADF): data asli tidak stasioner, setelah differencing menjadi stasioner.
5. **ACF/PACF**: membantu menentukan parameter ARIMA.
6. **Train-test split**: 10 hari terakhir sebagai data uji.
7. **Model ARIMA**: latih dan prediksi data uji.
8. **Evaluasi**: MSE, RMSE, MAE, Dstat.
9. **Forecast 30 hari** dan simpan hasil + grafik.

## Hasil Utama

### 1) Stasioneritas

- **ADF data asli**: p-value 0.9977 (tidak stasioner).
- **ADF setelah differencing**: p-value 0.0000 (stasioner).
- **Kesimpulan**: nilai **d = 1** pada ARIMA.

### 2) Evaluasi Model (Data Uji)

- **MSE**: 7959.2332
- **RMSE**: 89.2145 USD
- **MAE**: 71.5075 USD
- **Dstat**: 55.56% (sedikit lebih baik dari tebakan acak)

Interpretasi singkat:
- RMSE dan MAE menunjukkan rata-rata selisih prediksi sekitar 70-90 USD.
- Dstat 55.56% berarti arah pergerakan harga seringnya benar, tapi masih bisa ditingkatkan.

### 3) Prediksi 30 Hari ke Depan

- Rata-rata prediksi berada di sekitar **4527 USD**.
- **CI 95% pada H+30**: 4156.72 sampai 4897.30 USD.
- Tren jangka pendek terlihat **datar cenderung turun tipis**.

## Output yang Dihasilkan

- `reports/hasil/gold_data.csv` (dataset hasil preprocessing)
- `reports/hasil/forecast_30days.csv` (prediksi 30 hari)
- `reports/hasil/metrics.txt` (metrik evaluasi)
- `reports/hasil/actual_vs_predicted.png`
- `reports/hasil/forecast_30days.png`
- `reports/hasil/combined_forecast.png`

## Cara Menjalankan Ulang

1. Buka notebook: `Gold_Price_Forecasting_ARIMA.ipynb`
2. Jalankan semua sel dari atas ke bawah.
3. Hasil otomatis tersimpan di `reports/hasil`.

## Catatan

- Data diambil otomatis sampai tanggal saat notebook dijalankan.
- Hasil prediksi bersifat akademis dan bukan saran investasi.