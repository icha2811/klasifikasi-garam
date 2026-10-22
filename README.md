# Aplikasi Klasifikasi Asal Garam — Random Forest

Aplikasi Streamlit untuk menampilkan eksperimen Random Forest pada 59 sampel garam dari Brasil, Prancis, Portugal, dan Spanyol. Fitur mineral yang digunakan: K, Na, Ca, Mg, Fe, Mn, Zn, Cu.

## Menu

1. Beranda
2. Dataset
3. Hasil Evaluasi
4. Confusion Matrix & Fitur
5. Prediksi Asal Garam
6. Tentang Aplikasi

## Jalankan di komputer

Pastikan Python sudah terpasang. Di folder proyek ini, jalankan:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy via GitHub dan Streamlit Community Cloud

1. Buka GitHub → **New repository**, misalnya `klasifikasi-garam-streamlit`.
2. Klik **Add file → Upload files**. Unggah **isi folder ini** (`app.py`, `model_utils.py`, `requirements.txt`, `table_salt_origin.csv`, dan `README.md`), **bukan file ZIP**.
3. Klik **Commit changes**.
4. Kunjungi https://share.streamlit.io/ → masuk melalui GitHub → **Create app** → **Yup, I have an app**.
5. Pilih repository, branch `main`, entrypoint file `app.py`, lalu **Deploy**.
6. Setelah terbuka, coba menu Prediksi, Dataset, Hasil Evaluasi, dan Confusion Matrix.

## Perhatian

- Dataset adalah **data sekunder**, bersumber dari: Zanela Lima, L. R. et al. (2025), *Origin Identification of Table Salt Using Flame Atomic Absorption and Portable Near-Infrared Spectrometries*, *Chemosensors* 13(7), 231. DOI: https://doi.org/10.3390/chemosensors13070231; data disusun dari Tabel S2 pada bahan pendukung.
- Pastikan persyaratan atribusi dan penggunaan data sumber sebelum mempublikasikan repository secara terbuka.
- Model penelitian ini hanya memakai tiga stratified holdout tunggal. Angka terbaik 94,44% pada skenario 70:30 bukan jaminan akurasi pada data garam di luar dataset.
- Aplikasi merupakan demonstrasi penelitian, bukan alat sertifikasi asal garam.
- Kode aplikasi harus ditinjau dan disepakati dengan pihak yang akan dicantumkan sebagai pencipta/pemegang hak cipta dalam pengajuan HKI.
