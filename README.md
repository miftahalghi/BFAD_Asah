# Proyek Analisis Data: Bike Sharing Dataset (Capital Bikeshare 2011–2012)

Proyek ini menganalisis perilaku penyewaan sepeda pada sistem **Capital Bikeshare (Washington D.C., 2011–2012)** secara menyeluruh mulai dari perumusan pertanyaan bisnis berbasis **SMART Question**, **Data Wrangling** (*Gathering, Assessing, Cleaning*), **Exploratory Data Analysis (EDA)**, **Visualization & Explanatory Analysis**, **Analisis Lanjutan (Clustering Non-ML: Manual Grouping & Binning)**, hingga **Dashboard Interaktif berbasis Streamlit**.

---

## Struktur Direktori

```text
BFAD_Asah/
├── Bike-sharing-dataset/
│   ├── day.csv                  # Dataset mentah agregasi harian (731 baris)
│   ├── hour.csv                 # Dataset mentah agregasi per jam (17.379 baris)
│   └── Readme.txt               # Dokumentasi resmi dataset
├── dashboard/
│   ├── dashboard.py             # Aplikasi utama dashboard interaktif Streamlit
│   ├── day_clean.csv            # Dataset harian hasil cleaning & clustering
│   ├── hour_clean.csv           # Dataset per jam hasil cleaning & clustering
│   └── main_data.csv            # Dataset utama untuk kebutuhan dashboard
├── Proyek_Analisis_Data.ipynb   # Notebook analisis data lengkap beserta output visualisasi
├── README.md                    # Dokumentasi proyek dan panduan menjalankan dashboard
├── requirements.txt             # Daftar pustaka Python yang dibutuhkan
└── url.txt                      # Tautan deployment Streamlit Cloud
```

---

## Ringkasan Pertanyaan Bisnis (SMART Framework)

1. **Pengaruh Musim & Kondisi Cuaca (2011–2012):** Seberapa besar pengaruh variasi musim (*Spring, Summer, Fall, Winter*) dan kondisi cuaca (*Clear* hingga *Heavy Rain/Snow*) terhadap rata-rata serta total penyewaan sepeda harian (`cnt`, `casual`, dan `registered`), dan pada kondisi apa terjadi penurunan penyewaan paling drastis?
2. **Pola Jam Sibuk & Segmen Pengguna pada Hari Kerja vs Hari Libur:** Bagaimana perbedaan pola distribusi penyewaan sepeda per jam (00:00–23:00) antara segmen `registered` dan `casual` pada hari kerja (`workingday`) dibandingkan hari libur/akhir pekan (`holiday/weekend`), serta pada jam berapa terjadi puncak permintaan tertinggi (*peak hours*)?
3. **Pertumbuhan Tahunan (YoY) & Tren Bulanan (2011 vs 2012):** Berapa persentase tingkat pertumbuhan jumlah penyewaan sepeda secara tahunan (*Year-over-Year*) dan bagaimana tren pergerakan bulanan dari segmen pengguna `casual` dan `registered` antara tahun 2011 dan tahun 2012?

---

## Cara Menjalankan Dashboard di Lokal

### 1. Persiapan Environment (Menggunakan `venv` / `pip`)
Buka terminal atau PowerShell pada direktori proyek ini (`C:\Users\Lenovo\BFAD_Asah`), lalu instal seluruh dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

*(Opsional jika menggunakan Conda)*:
```bash
conda create --name bike-sharing-ds python=3.11
conda activate bike-sharing-ds
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Streamlit
Jalankan perintah berikut dari direktori utama proyek:

```bash
streamlit run dashboard/dashboard.py
```

Atau apabila Anda berada di dalam folder `dashboard`:
```bash
cd dashboard
streamlit run dashboard.py
```

Dashboard akan otomatis terbuka di peramban (*browser*) pada alamat `http://localhost:8501`.

# AI Attribution / Acknowledegemnts:
**Ide & Konsep: Saya menggunakan ChatGPT untuk brainstorm bagian spesifik yang mungkin untuk diekstrak polanya**
**Code Evaluator & Editor: Saya menggunakan Gemini untuk mengubah bentuk visualisasi agar lebih bagus dan menarik**