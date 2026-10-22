"""Aplikasi klasifikasi asal garam berbasis Streamlit untuk demonstrasi penelitian."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from model_utils import (
    COUNTRY_NAMES, FEATURES, FEATURE_LABELS, SPLITS,
    classification_by_country, fit_three_scenarios, load_dataset, predict_from_minerals
)

st.set_page_config(page_title="Klasifikasi Asal Garam", page_icon="🧂", layout="wide")
DATA_PATH = Path(__file__).resolve().parent / "table_salt_origin.csv"


@st.cache_data

def get_data():
    return load_dataset(DATA_PATH)


@st.cache_resource

def get_models():
    return fit_three_scenarios(get_data())


data = get_data()
results, summary, labels = get_models()
focus = results["70:30"]

st.sidebar.title("🧂 Asal Garam")
st.sidebar.caption("Aplikasi penelitian Random Forest")
page = st.sidebar.radio("Pilih menu", [
    "Beranda", "Dataset", "Hasil Evaluasi", "Confusion Matrix & Fitur",
    "Prediksi Asal Garam", "Tentang Aplikasi"
])
st.sidebar.divider()
st.sidebar.caption("Model: Random Forest · 100 pohon · random_state=42")
st.sidebar.caption("Data sekunder: Zanela Lima dkk. (2025), Chemosensors")


if page == "Beranda":
    st.title("🧂 Sistem Klasifikasi Asal Negara Garam")
    st.write("Aplikasi ini memperlihatkan pengujian **Random Forest** untuk mengenali asal negara garam dari delapan kandungan mineral. Hasil berasal dari dataset sekunder berisi 59 sampel dan empat negara.")
    a, b, c = st.columns(3)
    a.metric("Jumlah sampel", f"{len(data)}")
    b.metric("Negara asal", f"{data['Country'].nunique()}")
    c.metric("Accuracy eksperimen 70:30", "94,44%")
    st.subheader("Cara menggunakan aplikasi")
    st.markdown("""1. Buka **Dataset** untuk mengenal sampel dan mineral yang digunakan.
2. Buka **Hasil Evaluasi** untuk membandingkan skenario 60:40, 70:30, dan 80:20.
3. Lihat **Confusion Matrix & Fitur** untuk memahami kesalahan prediksi dan kepentingan mineral.
4. Buka **Prediksi Asal Garam** untuk memasukkan delapan nilai mineral.""")

elif page == "Dataset":
    st.title("Dataset Garam")
    st.write("Data berasal dari tabel pendukung penelitian Zanela Lima dkk. (2025), berisi delapan fitur mineral, kode sampel, dan negara asal.")
    left, right = st.columns([1, 2])
    with left:
        counts = data["Country"].value_counts().rename(index=COUNTRY_NAMES)
        st.subheader("Jumlah sampel per negara")
        st.bar_chart(counts)
    with right:
        st.subheader("Contoh data")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    st.subheader("Statistik fitur mineral")
    st.dataframe(data[FEATURES].describe().transpose().round(3), use_container_width=True)
    st.download_button("Unduh ringkasan statistik (CSV)", data[FEATURES].describe().transpose().to_csv().encode("utf-8"), file_name="statistik_mineral.csv", mime="text/csv")
    st.caption("Satuan Na dalam mg/g, sedangkan K, Ca, Mg, Fe, Mn, Zn, dan Cu dalam µg/g. Kode Sample tidak digunakan sebagai fitur model.")

elif page == "Hasil Evaluasi":
    st.title("Perbandingan Hasil Evaluasi")
    st.write("Seluruh model dilatih secara terpisah dengan pengaturan yang sama. Tidak ada cross-validation ataupun hyperparameter tuning.")
    st.dataframe(summary.style.format({k: "{:.2f}" for k in ["Accuracy (%)", "Precision macro (%)", "Recall macro (%)", "F1 macro (%)"]}), use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("Skenario")[["Accuracy (%)", "Precision macro (%)", "Recall macro (%)", "F1 macro (%)"]], y_label="Persentase (%)", stack=False)
    st.write("Pada pengujian ini, skenario **70:30** memperoleh accuracy tertinggi, yaitu **94,44%** (17 dari 18 sampel uji benar).")
    st.download_button("Unduh hasil evaluasi (CSV)", summary.to_csv(index=False).encode("utf-8"), "hasil_evaluasi_random_forest.csv", "text/csv")
    st.subheader("Classification report skenario 70:30")
    report = classification_by_country(focus, labels)
    st.dataframe(report.style.format({"Precision (%)": "{:.2f}", "Recall (%)": "{:.2f}", "F1-score (%)": "{:.2f}"}), hide_index=True, use_container_width=True)

elif page == "Confusion Matrix & Fitur":
    st.title("Confusion Matrix dan Feature Importance")
    st.subheader("Confusion matrix tiga skenario")
    abbreviations = ["BR" if s == "Brazil" else "FR" if s == "France" else "PT" if s == "Portugal" else "ES" for s in labels]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, item) in zip(axes, results.items()):
        ConfusionMatrixDisplay(confusion_matrix=item["cm"], display_labels=abbreviations).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{name} · {item['cm'].trace()}/{item['test_size']} benar")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("BR = Brasil, FR = Prancis, PT = Portugal, ES = Spanyol")
    st.subheader("Feature importance model 70:30")
    importance = pd.DataFrame({"Mineral": FEATURES, "Importance": focus["model"].feature_importances_}).sort_values("Importance")
    st.bar_chart(importance.set_index("Mineral"), horizontal=True, x_label="Nilai impurity-based importance")
    st.dataframe(importance.sort_values("Importance", ascending=False).reset_index(drop=True).style.format({"Importance": "{:.4f}"}), hide_index=True, use_container_width=True)
    st.caption("Nilai di atas adalah impurity-based importance, bukan ukuran hubungan sebab-akibat atau penurunan akurasi apabila fitur dihapus.")

elif page == "Prediksi Asal Garam":
    st.title("Prediksi Asal Garam")
    st.write("Masukkan delapan hasil pengukuran mineral. Prediksi menggunakan **model skenario 70:30**, sesuai fokus pembahasan artikel penelitian.")
    st.caption("Angka yang terisi adalah contoh dari median data. Ubah sesuai hasil pengukuran sampel garam.")
    median = data[FEATURES].median()
    values = {}
    with st.form("mineral_form"):
        cols = st.columns(2)
        for idx, key in enumerate(FEATURES):
            with cols[idx % 2]:
                step = max(round(float(median[key]) / 20, 3), 0.001)
                values[key] = st.number_input(FEATURE_LABELS[key], min_value=0.0, value=float(median[key]), step=step, format="%.4f")
        submitted = st.form_submit_button("Prediksi negara asal", type="primary")
    if submitted:
        country, probabilities = predict_from_minerals(focus["model"], values)
        st.success(f"Prediksi model: **{country}**")
        st.dataframe(probabilities.style.format({"Proporsi suara/probabilitas model": "{:.2%}"}), hide_index=True, use_container_width=True)
        st.caption("Hasil prediksi merupakan keluaran model penelitian, bukan bukti resmi asal produk.")

elif page == "Tentang Aplikasi":
    st.title("Tentang Aplikasi dan Sumber Data")
    st.markdown("""**Nama aplikasi:** Sistem Klasifikasi Asal Negara Garam Menggunakan Random Forest Berbasis Web.

**Tujuan:** mengklasifikasikan asal garam berdasarkan delapan mineral dan membandingkan hasil pengujian pada tiga pembagian data.

**Implementasi:** Python, Streamlit, pandas, scikit-learn, dan matplotlib.

**Pengaturan model:** `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`.

**Sumber data:** Zanela Lima, L. R., et al. (2025). *Origin Identification of Table Salt Using Flame Atomic Absorption and Portable Near-Infrared Spectrometries*. **Chemosensors**, 13(7), 231. https://doi.org/10.3390/chemosensors13070231. Data mineral disusun berdasarkan Tabel S2 pada materi pendukung publikasi.

**Batasan:** Model dilatih dari 59 sampel penelitian. Prediksi pada sampel baru masih memerlukan pengujian lebih lanjut.""")
