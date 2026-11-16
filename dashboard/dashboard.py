from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Capital Bikeshare Analytics Dashboard (2011–2012)",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 12.5
plt.rcParams["axes.titleweight"] = "bold"

# ==============================================================================
# FUNGSI LOAD & PREPARASI DATA (DENGAN CACHING)
# ==============================================================================
@st.cache_data
def load_data():
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent

    day_clean_path = current_dir / "day_clean.csv"
    hour_clean_path = current_dir / "hour_clean.csv"

    if day_clean_path.exists() and hour_clean_path.exists():
        day_df = pd.read_csv(day_clean_path)
        hour_df = pd.read_csv(hour_clean_path)
        day_df["dteday"] = pd.to_datetime(day_df["dteday"])
        hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])
    else:
        # Fallback apabila dijalankan langsung dari dataset mentah
        raw_dir = root_dir / "Bike-sharing-dataset"
        day_df = pd.read_csv(raw_dir / "day.csv")
        hour_df = pd.read_csv(raw_dir / "hour.csv")

        # Cleaning hum == 0
        med_day = day_df[(day_df["hum"] > 0) & (day_df["season"] == 1) & (day_df["weathersit"] == 3)]["hum"].median()
        med_hr = hour_df[(hour_df["hum"] > 0) & (hour_df["season"] == 1) & (hour_df["weathersit"] == 3)]["hum"].median()
        day_df.loc[day_df["hum"] == 0, "hum"] = med_day
        hour_df.loc[hour_df["hum"] == 0, "hum"] = med_hr

        season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        yr_map = {0: "2011", 1: "2012"}
        mnth_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        weekday_map = {
            0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
            4: "Thursday", 5: "Friday", 6: "Saturday"
        }
        workingday_map = {0: "Holiday/Weekend", 1: "Working Day"}
        weather_map = {
            1: "Clear/Partly Cloudy",
            2: "Mist/Cloudy",
            3: "Light Snow/Rain",
            4: "Heavy Rain/Snow"
        }

        for df in [day_df, hour_df]:
            df["dteday"] = pd.to_datetime(df["dteday"])
            df["season_label"] = df["season"].map(season_map)
            df["year_label"] = df["yr"].map(yr_map)
            df["month_label"] = df["mnth"].map(mnth_map)
            df["weekday_label"] = df["weekday"].map(weekday_map)
            df["workingday_label"] = df["workingday"].map(workingday_map)
            df["weather_label"] = df["weathersit"].map(weather_map)
            df["temp_celsius"] = df["temp"] * 41.0
            df["atemp_celsius"] = df["atemp"] * 50.0
            df["hum_pct"] = df["hum"] * 100.0
            df["windspeed_kmh"] = df["windspeed"] * 67.0

    # Pastikan urutan kategori konsisten untuk visualisasi
    season_cats = ["Spring", "Summer", "Fall", "Winter"]
    month_cats = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    weekday_cats = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weather_cats = ["Clear/Partly Cloudy", "Mist/Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"]

    for df in [day_df, hour_df]:
        df["season_label"] = pd.Categorical(df["season_label"], categories=season_cats, ordered=True)
        df["month_label"] = pd.Categorical(df["month_label"], categories=month_cats, ordered=True)
        df["weekday_label"] = pd.Categorical(df["weekday_label"], categories=weekday_cats, ordered=True)
        df["weather_label"] = pd.Categorical(df["weather_label"], categories=weather_cats, ordered=True)
        df["year_label"] = df["year_label"].astype(str)

    # Pastikan kolom klaster tersedia
    if "time_cluster" not in hour_df.columns:
        def assign_time_cluster(hr):
            hr = int(hr)
            if 0 <= hr <= 5:
                return "1. Early Morning (00:00-05:59)"
            elif 6 <= hr <= 9:
                return "2. Morning Rush (06:00-09:59)"
            elif 10 <= hr <= 15:
                return "3. Midday Off-Peak (10:00-15:59)"
            elif 16 <= hr <= 19:
                return "4. Evening Rush (16:00-19:59)"
            else:
                return "5. Night (20:00-23:59)"
        hour_df["time_cluster"] = hour_df["hr"].apply(assign_time_cluster)

    if "comfort_cluster" not in day_df.columns:
        def assign_comfort(row):
            if (
                row["weather_label"] == "Clear/Partly Cloudy"
                and 15.0 <= row["temp_celsius"] <= 28.0
                and row["hum_pct"] < 75.0
                and row["windspeed_kmh"] < 25.0
            ):
                return "Prime Cycling Condition"
            elif (
                row["weather_label"] in ["Light Snow/Rain", "Heavy Rain/Snow"]
                or row["temp_celsius"] < 10.0
                or row["hum_pct"] >= 80.0
            ):
                return "Adverse Weather Condition"
            else:
                return "Moderate Condition"
        day_df["comfort_cluster"] = day_df.apply(assign_comfort, axis=1)

    if "temp_bin" not in day_df.columns:
        day_df["temp_bin"] = pd.cut(
            day_df["temp_celsius"],
            bins=[0, 12, 20, 28, 45],
            labels=["Cold (<12°C)", "Mild (12°C-20°C)", "Warm (20°C-28°C)", "Hot (>28°C)"],
            include_lowest=True
        ).astype(str)

    if "demand_tier" not in day_df.columns:
        day_df["demand_tier"] = pd.qcut(
            day_df["cnt"],
            q=3,
            labels=["Low Demand (Bottom 33%)", "Medium Demand (Middle 33%)", "High Demand (Top 33%)"]
        ).astype(str)

    return day_df, hour_df


day_df, hour_df = load_data()

# ==============================================================================
# SIDEBAR & FILTER INTERAKTIF
# ==============================================================================
with st.sidebar:
    st.title("🚲 Capital Bikeshare")
    st.caption("Dashboard Analisis Penyewaan Sepeda Washington D.C. (2011–2012)")
    st.markdown("---")

    min_date = day_df["dteday"].min().date()
    max_date = day_df["dteday"].max().date()

    date_selection = st.date_input(
        "📅 Pilih Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
        start_date, end_date = date_selection
    elif isinstance(date_selection, (tuple, list)) and len(date_selection) == 1:
        start_date = end_date = date_selection[0]
    else:
        start_date = end_date = date_selection

    all_seasons = ["Spring", "Summer", "Fall", "Winter"]
    selected_seasons = st.multiselect(
        "🍂 Filter Musim (Season)",
        options=all_seasons,
        default=all_seasons
    )

    all_weathers = ["Clear/Partly Cloudy", "Mist/Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"]
    selected_weathers = st.multiselect(
        "⛅ Filter Kondisi Cuaca",
        options=all_weathers,
        default=all_weathers
    )

    all_daytypes = ["Working Day", "Holiday/Weekend"]
    selected_daytypes = st.multiselect(
        "🏢 Filter Tipe Hari",
        options=all_daytypes,
        default=all_daytypes
    )

    st.markdown("---")
    st.markdown(
        "**Metode Analisis Lanjutan:**\n"
        "- *Clustering (Manual Grouping)*: Time-of-Day & Cycling Comfort Index\n"
        "- *Clustering (Binning)*: Temperature Regimes (`pd.cut`) & Demand Tiers (`pd.qcut`)"
    )

# Terapkan filter pada day_df dan hour_df
day_filtered = day_df[
    (day_df["dteday"].dt.date >= start_date)
    & (day_df["dteday"].dt.date <= end_date)
    & (day_df["season_label"].isin(selected_seasons))
    & (day_df["weather_label"].isin(selected_weathers))
    & (day_df["workingday_label"].isin(selected_daytypes))
].copy()

hour_filtered = hour_df[
    (hour_df["dteday"].dt.date >= start_date)
    & (hour_df["dteday"].dt.date <= end_date)
    & (hour_df["season_label"].isin(selected_seasons))
    & (hour_df["weather_label"].isin(selected_weathers))
    & (hour_df["workingday_label"].isin(selected_daytypes))
].copy()

# ==============================================================================
# HEADER UTAMA & METRIK KPI RINGKASAN EKSEKUTIF
# ==============================================================================
st.title("🚲 Dashboard Analisis Data: Bike Sharing Dataset (2011–2012)")
st.markdown(
    f"Menampilkan hasil analisis interaktif pada periode **{start_date.strftime('%d %b %Y')}** "
    f"hingga **{end_date.strftime('%d %b %Y')}** ({len(day_filtered):,} hari operasional, {len(hour_filtered):,} jam observasi)."
)

if day_filtered.empty:
    st.warning("Tidak ada data yang sesuai dengan kombinasi filter yang dipilih. Silakan sesuaikan kembali filter di panel sebelah kiri.")
    st.stop()

total_cnt = int(day_filtered["cnt"].sum())
mean_daily_cnt = day_filtered["cnt"].mean()
total_reg = int(day_filtered["registered"].sum())
total_cas = int(day_filtered["casual"].sum())
reg_share = (total_reg / total_cnt * 100) if total_cnt > 0 else 0
cas_share = (total_cas / total_cnt * 100) if total_cnt > 0 else 0
mean_temp = day_filtered["temp_celsius"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Penyewaan (cnt)", f"{total_cnt:,}", f"{len(day_filtered)} Hari")
col2.metric("Rata-rata Penyewaan / Hari", f"{mean_daily_cnt:,.0f} sepeda", f"Max: {day_filtered['cnt'].max():,}")
col3.metric("Segmen Registered", f"{total_reg:,}", f"{reg_share:.1f}% dari Total")
col4.metric("Segmen Casual", f"{total_cas:,}", f"{cas_share:.1f}% dari Total")
col5.metric("Rata-rata Suhu Aktual", f"{mean_temp:.1f} °C", f"Kelembaban: {day_filtered['hum_pct'].mean():.1f}%")

st.markdown("---")

# ==============================================================================
# NAVIGASI TAB ANALISIS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌤️ 1. Pengaruh Musim & Cuaca",
    "⏰ 2. Pola Jam Sibuk & Tipe Hari",
    "📈 3. Pertumbuhan YoY & Tren Bulanan",
    "🧩 4. Analisis Lanjutan (Clustering)",
    "🎯 5. Kesimpulan & Rekomendasi"
])

# ------------------------------------------------------------------------------
# TAB 1: PERTANYAAN 1 (MUSIM & KONDISI CUACA)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Pertanyaan 1: Pengaruh Musim dan Kondisi Cuaca terhadap Penyewaan Sepeda")
    st.markdown(
        "Analisis ini mengidentifikasi bagaimana variasi musim (*Spring, Summer, Fall, Winter*) dan kondisi cuaca "
        "memengaruhi rata-rata serta total penyewaan sepeda harian."
    )

    c1, c2 = st.columns(2)

    with c1:
        season_agg = (
            day_filtered.groupby("season_label", observed=False)
            .agg(
                rerata_cnt=("cnt", "mean"),
                total_cnt=("cnt", "sum"),
                rerata_reg=("registered", "mean"),
                rerata_cas=("casual", "mean")
            )
            .reset_index()
            .dropna(subset=["rerata_cnt"])
            .sort_values(by="rerata_cnt", ascending=False)
        )

        fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
        if not season_agg.empty:
            max_s = season_agg["rerata_cnt"].max()
            min_s = season_agg["rerata_cnt"].min()
            colors_s = [
                "#1f77b4" if v == max_s else ("#d62728" if v == min_s and len(season_agg) > 1 else "#9bb8d3")
                for v in season_agg["rerata_cnt"]
            ]
            bars = ax1.barh(season_agg["season_label"].astype(str), season_agg["rerata_cnt"], color=colors_s, height=0.58)
            ax1.invert_yaxis()
            ax1.set_title("Rata-rata Penyewaan Harian per Musim")
            ax1.set_xlabel("Rata-rata Penyewaan (Sepeda / Hari)")
            ax1.set_xlim(0, max_s * 1.25)
            for b in bars:
                ax1.text(
                    b.get_width() + (max_s * 0.02),
                    b.get_y() + b.get_height() / 2,
                    f"{b.get_width():,.0f}/hari",
                    va="center", fontweight="bold", fontsize=9.5
                )
        sns.despine(ax=ax1)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with c2:
        weather_agg = (
            day_filtered.groupby("weather_label", observed=False)
            .agg(rerata_cnt=("cnt", "mean"), jumlah_hari=("cnt", "count"))
            .reset_index()
        )
        weather_agg = weather_agg[weather_agg["jumlah_hari"] > 0]

        fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
        if not weather_agg.empty:
            max_w = weather_agg["rerata_cnt"].max()
            min_w = weather_agg["rerata_cnt"].min()
            colors_w = [
                "#2ca02c" if v == max_w else ("#d62728" if v == min_w and len(weather_agg) > 1 else "#9bb8d3")
                for v in weather_agg["rerata_cnt"]
            ]
            bars_w = ax2.bar(weather_agg["weather_label"].astype(str), weather_agg["rerata_cnt"], color=colors_w, width=0.52)
            ax2.set_title("Rata-rata Penyewaan Harian menurut Kondisi Cuaca")
            ax2.set_ylabel("Rata-rata Penyewaan (Sepeda / Hari)")
            ax2.set_ylim(0, max_w * 1.25)
            for b, (_, r) in zip(bars_w, weather_agg.iterrows()):
                ax2.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + (max_w * 0.03),
                    f"{b.get_height():,.0f}/hari\n({int(r['jumlah_hari'])} hari)",
                    ha="center", fontweight="bold", fontsize=9.5
                )
        sns.despine(ax=ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    c3, c4 = st.columns(2)
    with c3:
        piv_sw = pd.pivot_table(
            day_filtered,
            values="cnt",
            index="season_label",
            columns="weather_label",
            aggfunc="mean",
            observed=False
        ).dropna(how="all", axis=1)

        fig3, ax3 = plt.subplots(figsize=(7.5, 4.6))
        sns.heatmap(piv_sw, annot=True, fmt=",.0f", cmap="YlGnBu", ax=ax3, cbar_kws={"label": "Rerata / Hari"})
        ax3.set_title("Matriks Interaksi Musim x Kondisi Cuaca (Sepeda / Hari)")
        ax3.set_xlabel("Kondisi Cuaca")
        ax3.set_ylabel("Musim")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with c4:
        fig4, ax4 = plt.subplots(figsize=(7.5, 4.6))
        sns.scatterplot(
            data=day_filtered,
            x="temp_celsius",
            y="cnt",
            hue="weather_label",
            palette="deep",
            alpha=0.8,
            ax=ax4
        )
        sns.regplot(
            data=day_filtered,
            x="temp_celsius",
            y="cnt",
            scatter=False,
            color="#2c3e50",
            line_kws={"linestyle": "--"},
            ax=ax4
        )
        ax4.set_title("Korelasi Suhu Aktual (°C) terhadap Total Penyewaan Harian")
        ax4.set_xlabel("Suhu Aktual (°C)")
        ax4.set_ylabel("Total Penyewaan Harian (cnt)")
        ax4.legend(title="Cuaca", fontsize=8.5)
        sns.despine(ax=ax4)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    st.info(
        "💡 **Temuan Utama Pertanyaan 1:** Musim **Fall** menghasilkan rata-rata penyewaan tertinggi (**5.644 sepeda/hari**), "
        "sedangkan musim **Spring** mengalami penurunan paling drastis sebesar **53,86%** (**2.604 sepeda/hari**). "
        "Dari faktor cuaca, kondisi **Light Snow/Rain** menurunkan rata-rata penyewaan harian sebesar **63,02%** "
        "(dari **4.877/hari** saat cerah menjadi **1.803/hari**)."
    )

# ------------------------------------------------------------------------------
# TAB 2: PERTANYAAN 2 (POLA JAM OPERASIONAL & TIPE HARI)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Pertanyaan 2: Pola Penyewaan Per Jam (00:00–23:00) pada Hari Kerja vs Hari Libur")
    st.markdown(
        "Membandingkan distribusi penyewaan per jam antara pelanggan **Registered** dan **Casual** "
        "pada **Hari Kerja (`Working Day`)** terhadap **Hari Libur/Akhir Pekan (`Holiday/Weekend`)**."
    )

    hr_group = (
        hour_filtered.groupby(["workingday_label", "hr"], observed=False)
        .agg(
            rerata_cnt=("cnt", "mean"),
            rerata_registered=("registered", "mean"),
            rerata_casual=("casual", "mean")
        )
        .reset_index()
    )

    fig_hr, axes_hr = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)

    for idx, day_type in enumerate(["Working Day", "Holiday/Weekend"]):
        sub = hr_group[hr_group["workingday_label"] == day_type].sort_values("hr")
        ax = axes_hr[idx]
        if not sub.empty:
            ax.plot(sub["hr"].astype(int), sub["rerata_cnt"], marker="o", lw=2.4, color="#1f77b4", label="Total (cnt)")
            ax.plot(sub["hr"].astype(int), sub["rerata_registered"], marker="s", lw=2.0, ls="--", color="#2ca02c", label="Registered")
            ax.plot(sub["hr"].astype(int), sub["rerata_casual"], marker="^", lw=2.0, ls=":", color="#ff7f0e", label="Casual")
            if day_type == "Working Day":
                ax.axvspan(7, 9, color="#1f77b4", alpha=0.10, label="Puncak Pagi (07-09)")
                ax.axvspan(16, 19, color="#d62728", alpha=0.10, label="Puncak Sore (16-19)")
            else:
                ax.axvspan(11, 16, color="#ff7f0e", alpha=0.12, label="Puncak Siang (11-16)")
        ax.set_title(f"Profil Per Jam: {day_type}")
        ax.set_xlabel("Jam (00:00 - 23:00)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=8.5, frameon=True)
        sns.despine(ax=ax)

    axes_hr[0].set_ylabel("Rata-rata Penyewaan / Jam")
    plt.tight_layout()
    st.pyplot(fig_hr)
    plt.close(fig_hr)

    # Heatmap Hari (Monday-Sunday) x Jam (0-23)
    st.markdown("#### Peta Panas (Heatmap) Intensitas Penyewaan: Hari dalam Seminggu × Jam Operasional")
    hm_weekday_hr = pd.pivot_table(
        hour_filtered,
        values="cnt",
        index="weekday_label",
        columns="hr",
        aggfunc="mean",
        observed=False
    )
    fig_hm, ax_hm = plt.subplots(figsize=(14, 4.2))
    sns.heatmap(hm_weekday_hr, cmap="YlOrRd", annot=False, cbar_kws={"label": "Rerata Sepeda / Jam"}, ax=ax_hm)
    ax_hm.set_title("Rata-rata Penyewaan Sepeda Berdasarkan Hari dan Jam (00:00 - 23:00)")
    ax_hm.set_xlabel("Jam Operasional")
    ax_hm.set_ylabel("Hari")
    plt.tight_layout()
    st.pyplot(fig_hm)
    plt.close(fig_hm)

    st.info(
        "💡 **Temuan Utama Pertanyaan 2:** Pada **Hari Kerja**, pola penyewaan bersifat **bimodal** dengan puncak pada "
        "**pukul 08:00 (477,0 sepeda/jam)** dan **pukul 17:00 (525,3 sepeda/jam)** yang didominasi oleh komuter `registered` (~90%–95%). "
        "Pada **Hari Libur/Akhir Pekan**, pola menjadi **unimodal** yang memuncak pada **pukul 13:00 (372,7 sepeda/jam)**, "
        "di mana rata-rata harian pengguna `casual` melonjak **+126,05%** dibandingkan hari kerja."
    )

# ------------------------------------------------------------------------------
# TAB 3: PERTANYAAN 3 (PERTUMBUHAN YOY & TREN BULANAN)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Pertanyaan 3: Pertumbuhan Year-over-Year (YoY) dan Tren Bulanan (2011 vs 2012)")

    monthly_agg = (
        day_filtered.groupby(["year_label", "month_label"], observed=False)
        .agg(total_cnt=("cnt", "sum"), total_reg=("registered", "sum"), total_cas=("casual", "sum"))
        .reset_index()
    )

    col_t1, col_t2 = st.columns([1.65, 1])

    with col_t1:
        fig_m, ax_m = plt.subplots(figsize=(9, 4.8))
        for yr, color, style in [("2011", "#7f8c8d", "--"), ("2012", "#1f77b4", "-")]:
            sub_m = monthly_agg[monthly_agg["year_label"] == yr]
            if not sub_m.empty:
                ax_m.plot(
                    sub_m["month_label"].astype(str),
                    sub_m["total_cnt"],
                    marker="o",
                    linewidth=2.6,
                    linestyle=style,
                    color=color,
                    label=f"Tahun {yr}"
                )
        ax_m.set_title("Perbandingan Total Penyewaan Bulanan (2011 vs 2012)")
        ax_m.set_xlabel("Bulan")
        ax_m.set_ylabel("Total Penyewaan Bulanan")
        ax_m.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
        ax_m.legend(frameon=True)
        sns.despine(ax=ax_m)
        plt.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)

    with col_t2:
        yr_agg = (
            day_filtered.groupby("year_label", observed=False)
            .agg(Total=("cnt", "sum"), Registered=("registered", "sum"), Casual=("casual", "sum"))
            .T
        )
        fig_y, ax_y = plt.subplots(figsize=(5.8, 4.8))
        yr_agg.plot(kind="bar", ax=ax_y, color=["#95a5a6", "#1f77b4"], width=0.65)
        ax_y.set_title("Total Penyewaan Tahunan per Segmen")
        ax_y.set_ylabel("Total Penyewaan")
        ax_y.tick_params(axis="x", rotation=0)
        ax_y.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M"))
        ax_y.legend(title="Tahun", frameon=True)
        sns.despine(ax=ax_y)
        plt.tight_layout()
        st.pyplot(fig_y)
        plt.close(fig_y)

    st.markdown("#### Pergerakan Penyewaan Harian Sepanjang Waktu (Daily Time Series)")
    fig_ts, ax_ts = plt.subplots(figsize=(14, 3.8))
    ax_ts.plot(day_filtered["dteday"], day_filtered["cnt"], color="#85c1e9", alpha=0.6, label="Penyewaan Harian (cnt)")
    day_filtered_sorted = day_filtered.sort_values("dteday")
    ax_ts.plot(
        day_filtered_sorted["dteday"],
        day_filtered_sorted["cnt"].rolling(14, min_periods=1).mean(),
        color="#1f77b4",
        linewidth=2.2,
        label="Rata-rata Bergerak 14 Hari (14-Day Moving Average)"
    )
    ax_ts.set_title("Tren Harian & Rata-rata Bergerak 14 Hari Penyewaan Sepeda")
    ax_ts.set_xlabel("Tanggal")
    ax_ts.set_ylabel("Jumlah Penyewaan / Hari")
    ax_ts.legend(frameon=True)
    sns.despine(ax=ax_ts)
    plt.tight_layout()
    st.pyplot(fig_ts)
    plt.close(fig_ts)

    st.info(
        "💡 **Temuan Utama Pertanyaan 3:** Total penyewaan tumbuh **+64,88% YoY** dari **1.243.103** (2011) menjadi "
        "**2.049.576** (2012). Segmen `registered` tumbuh **+68,38% YoY**, sementara `casual` tumbuh **+50,76% YoY**. "
        "Periode permintaan tertinggi berada pada rentang **Mei–Oktober** dengan rekor puncak pada **September 2012 (218.573 penyewaan)**."
    )

# ------------------------------------------------------------------------------
# TAB 4: ANALISIS LANJUTAN (CLUSTERING: MANUAL GROUPING & BINNING)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Analisis Lanjutan: Clustering Tanpa Algoritma Machine Learning")
    st.markdown(
        "Menerapkan dua pendekatan **Clustering Non-ML** yang relevan dengan domain Bike Sharing: "
        "**1) Manual Grouping** (*Time-of-Day Clustering* & *Cycling Comfort Index*) dan "
        "**2) Binning** (*Temperature Regimes `pd.cut`* & *Demand Tiers `pd.qcut`*)."
    )

    ca1, ca2 = st.columns(2)

    with ca1:
        tc_df = (
            hour_filtered.groupby(["time_cluster", "workingday_label"], observed=False)["cnt"]
            .mean()
            .reset_index()
        )
        fig_tc, ax_tc = plt.subplots(figsize=(7.5, 4.8))
        sns.barplot(
            data=tc_df,
            x="time_cluster",
            y="cnt",
            hue="workingday_label",
            palette=["#1f77b4", "#ff7f0e"],
            ax=ax_tc
        )
        ax_tc.set_title("Manual Grouping 1: Klaster Waktu Operasional")
        ax_tc.set_xlabel("Klaster Waktu")
        ax_tc.set_ylabel("Rata-rata Penyewaan / Jam")
        ax_tc.tick_params(axis="x", rotation=18)
        ax_tc.legend(title="Tipe Hari", frameon=True)
        sns.despine(ax=ax_tc)
        plt.tight_layout()
        st.pyplot(fig_tc)
        plt.close(fig_tc)

    with ca2:
        cc_order = ["Prime Cycling Condition", "Moderate Condition", "Adverse Weather Condition"]
        cc_df = (
            day_filtered.groupby("comfort_cluster", observed=False)
            .agg(rerata_cnt=("cnt", "mean"), hari=("cnt", "count"))
            .reindex(cc_order)
            .dropna()
            .reset_index()
        )
        fig_cc, ax_cc = plt.subplots(figsize=(7.5, 4.8))
        bars_cc = ax_cc.bar(cc_df["comfort_cluster"], cc_df["rerata_cnt"], color=["#2ca02c", "#9bb8d3", "#d62728"][:len(cc_df)], width=0.52)
        ax_cc.set_title("Manual Grouping 2: Indeks Kenyamanan Cuaca (Comfort Cluster)")
        ax_cc.set_xlabel("Klaster Kenyamanan Bersepeda")
        ax_cc.set_ylabel("Rata-rata Penyewaan / Hari")
        if not cc_df.empty:
            max_cc = cc_df["rerata_cnt"].max()
            ax_cc.set_ylim(0, max_cc * 1.25)
            for b, (_, r) in zip(bars_cc, cc_df.iterrows()):
                ax_cc.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + (max_cc * 0.03),
                    f"{b.get_height():,.0f}/hari\n({int(r['hari'])} hari)",
                    ha="center", fontweight="bold", fontsize=9.5
                )
        sns.despine(ax=ax_cc)
        plt.tight_layout()
        st.pyplot(fig_cc)
        plt.close(fig_cc)

    cb1, cb2 = st.columns(2)

    with cb1:
        tb_order = ["Cold (<12°C)", "Mild (12°C-20°C)", "Warm (20°C-28°C)", "Hot (>28°C)"]
        tb_df = (
            day_filtered.groupby("temp_bin", observed=False)
            .agg(rerata_cnt=("cnt", "mean"), total_cnt=("cnt", "sum"))
            .reindex(tb_order)
            .dropna()
            .reset_index()
        )
        fig_tb, ax_tb = plt.subplots(figsize=(7.5, 4.8))
        bars_tb = ax_tb.bar(tb_df["temp_bin"], tb_df["rerata_cnt"], color=["#9bb8d3", "#9bb8d3", "#1f77b4", "#ff7f0e"][:len(tb_df)], width=0.52)
        ax_tb.set_title("Binning 1 (pd.cut): Kategori Suhu Aktual")
        ax_tb.set_xlabel("Kategori Suhu (°C)")
        ax_tb.set_ylabel("Rata-rata Penyewaan / Hari")
        if not tb_df.empty:
            max_tb = tb_df["rerata_cnt"].max()
            ax_tb.set_ylim(0, max_tb * 1.25)
            for b in bars_tb:
                ax_tb.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + (max_tb * 0.03),
                    f"{b.get_height():,.0f}/hari",
                    ha="center", fontweight="bold", fontsize=9.5
                )
        sns.despine(ax=ax_tb)
        plt.tight_layout()
        st.pyplot(fig_tb)
        plt.close(fig_tb)

    with cb2:
        dt_order = ["Low Demand (Bottom 33%)", "Medium Demand (Middle 33%)", "High Demand (Top 33%)"]
        ct_dt = pd.crosstab(day_filtered["demand_tier"], day_filtered["season_label"], normalize="index").reindex(dt_order).dropna(how="all") * 100
        fig_dt, ax_dt = plt.subplots(figsize=(7.5, 4.8))
        if not ct_dt.empty:
            ct_dt.plot(kind="bar", stacked=True, color=["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"], ax=ax_dt, width=0.58)
            ax_dt.set_title("Binning 2 (pd.qcut): Komposisi Musim per Tier Permintaan")
            ax_dt.set_xlabel("Kuartil Permintaan Harian (Demand Tier)")
            ax_dt.set_ylabel("Proporsi Hari (%)")
            ax_dt.tick_params(axis="x", rotation=10)
            ax_dt.legend(title="Musim", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
        sns.despine(ax=ax_dt)
        plt.tight_layout()
        st.pyplot(fig_dt)
        plt.close(fig_dt)

# ------------------------------------------------------------------------------
# TAB 5: KESIMPULAN & REKOMENDASI ACTION ITEM
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Kesimpulan Analisis & Rekomendasi Action Item")
    st.markdown(
        """
        ### 📌 Kesimpulan Utama (Conclusion)
        1. **Pengaruh Musim & Kondisi Cuaca (Pertanyaan 1):**
           - Musim **Fall** mencetak rata-rata penyewaan tertinggi (**5.644,30 sepeda/hari** atau **32,23%** dari total penyewaan 2011–2012), diikuti **Summer** (**4.992,33/hari**) dan **Winter** (**4.728,16/hari**). Sebaliknya, musim **Spring** mengalami penurunan paling tajam sebesar **53,86%** (**2.604,13 sepeda/hari**).
           - Kondisi cuaca **Light Snow/Rain** memangkas rata-rata penyewaan harian sebesar **63,02%** dibandingkan cuaca cerah (**1.803,29/hari** vs **4.876,79/hari**).
        2. **Pola Jam Sibuk & Segmen Pengguna pada Hari Kerja vs Hari Libur (Pertanyaan 2):**
           - **Hari Kerja (`Working Day`)** menunjukkan pola **bimodal komuter** yang didominasi pelanggan `registered` (86,77% pangsa hari kerja), dengan puncak pagi pukul **08:00 (477,01 sepeda/jam)** dan puncak sore pukul **17:00 (525,29 sepeda/jam)** serta **18:00 (492,23 sepeda/jam)**.
           - **Hari Libur/Akhir Pekan (`Holiday/Weekend`)** menunjukkan pola **unimodal rekreasi** yang memuncak pada pukul **11:00–16:00** (tertinggi pukul **13:00: 372,73 sepeda/jam**), dengan lonjakan pengguna `casual` sebesar **+126,05%** dibandingkan hari kerja.
        3. **Pertumbuhan Tahunan YoY & Tren Bulanan (Pertanyaan 3):**
           - Total penyewaan tumbuh pesat sebesar **+64,88% YoY** dari **1.243.103** (2011) menjadi **2.049.576** (2012). Segmen `registered` tumbuh **+68,38% YoY** dan segmen `casual` tumbuh **+50,76% YoY**, dengan puncak permintaan terjadi pada bulan **Mei–Oktober** (rekor tertinggi **September 2012: 218.573 penyewaan**).

        ---
        ### 🚀 Rekomendasi Action Item (Berorientasi Aksi)
        1. **Penjadwalan *Fleet Rebalancing* Berbasis Klaster Waktu:**
           - Lakukan redistribusi sepeda (*rebalancing*) dua gelombang pada hari kerja: **06:00–07:15** ke stasiun permukiman (sebelum puncak pukul 08:00) dan **15:00–16:15** ke stasiun kawasan perkantoran (sebelum puncak pukul 17:00–18:00). Pada akhir pekan, fokuskan redistribusi pada pukul **09:30–10:30** di stasiun taman dan area wisata.
        2. **Kampanye Konversi *Weekend Casual-to-Registered* & *Leisure Pass*:**
           - Tawarkan diskon keanggotaan bulan pertama (*First-Month Membership Promo* 20%–25%) kepada pengguna `casual` yang menyewa pada jam puncak akhir pekan (11:00–16:00) untuk mengonversi lonjakan pengguna rekreasi (+126,05%) menjadi pelanggan tetap (`registered`).
        3. **Optimasi Jadwal Pemeliharaan Armada (*Major Maintenance*) & *Weather-Responsive Pricing*:**
           - Jadwalkan *overhaul* dan perawatan besar armada secara bergilir pada periode *Low Demand* (**Januari–Maret / awal Spring**, suhu $<12^\\circ\\text{C}$) dan pastikan **100% kesiapan armada sebelum bulan Mei**. Terapkan tarif dinamis atau insentif poin tambahan pada kondisi *Mist/Cloudy* untuk menjaga utilisasi sepeda.
        """
    )
