import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.title("Dashboard Analitik DJPb")

# Load Data
DATA_PATH = "data/02_realisasi_anggaran_klasifikasi (1).csv"
MODEL_PATH = "model/Best_model.pkcls"

data = pd.read_csv(DATA_PATH)

# Load Model
model = None
model_error = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        model_error = str(e)
else:
    model_error = "File model tidak ditemukan di folder model."

# Page Tabs
tab_data, tab_predict = st.tabs(["Data & Visualisasi", "Prediksi"])

with tab_data:
    st.header("Data Utama")
    st.dataframe(data.head())

    x_column = "skor_ikpa"
    y_column = "deviasi_rpd_persen"
    if x_column in data.columns and y_column in data.columns:
        st.subheader("Visualisasi Scatter Plot")
        st.write(f"X: {x_column}, Y: {y_column}")
        fig, ax = plt.subplots()
        ax.scatter(data[x_column], data[y_column], alpha=0.6, edgecolors="w", linewidth=0.5)
        ax.set_xlabel("IKPA")
        ax.set_ylabel("Deviasi RPD (%)")
        ax.set_title("Scatter Plot IKPA vs Deviasi RPD")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
    else:
        st.warning("Kolom untuk scatter plot tidak ditemukan dalam data.")

with tab_predict:
    st.header("Menu Prediksi")
    st.write("Isi data berikut untuk menjalankan prediksi menggunakan model.")

    jumlah_spm = st.number_input(
        "Jumlah SPM",
        min_value=0,
        max_value=500,
        value=int(data["jumlah_spm"].median()),
        step=1,
        format="%d",
    )
    revisi_dipa = st.number_input(
        "Revisi DIPA",
        min_value=0,
        max_value=10,
        value=int(data["revisi_dipa"].median()),
        step=1,
        format="%d",
    )
    deviasi_rpd = st.number_input(
        "Deviasi RPD (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(round(data["deviasi_rpd_persen"].median(), 2)),
        step=0.1,
        format="%.2f",
    )
    skor_ikpa = st.number_input(
        "Skor IKPA",
        min_value=0.0,
        max_value=100.0,
        value=float(round(data["skor_ikpa"].median(), 2)),
        step=0.01,
        format="%.2f",
    )
    tipe_options = data["tipe_satker"].dropna().unique().tolist()
    tipe_satker = st.selectbox("Tipe Satker", tipe_options)

    if st.button("Hitung Prediksi"):
        if model is None:
            st.error("Model tidak dapat dimuat. " + (model_error or "Periksa file model atau dependensi."))
        else:
            tipe_categories = [
                "Dekonsentrasi",
                "Kantor Daerah",
                "Kantor Pusat",
                "Tugas Pembantuan",
            ]
            row = {
                "jumlah_spm": jumlah_spm,
                "revisi_dipa": revisi_dipa,
                "deviasi_rpd_persen": deviasi_rpd,
                "skor_ikpa": skor_ikpa,
            }
            for category in tipe_categories:
                row[f"tipe_satker={category}"] = 1 if tipe_satker == category else 0

            input_data = pd.DataFrame([row], columns=[
                "jumlah_spm",
                "revisi_dipa",
                "deviasi_rpd_persen",
                "skor_ikpa",
                "tipe_satker=Dekonsentrasi",
                "tipe_satker=Kantor Daerah",
                "tipe_satker=Kantor Pusat",
                "tipe_satker=Tugas Pembantuan",
            ])
            try:
                prediction = model.predict(input_data)
                if isinstance(prediction, tuple) and len(prediction) >= 2:
                    label = prediction[0][0]
                    result_text = "Realisasi tercapai 95%" if str(label).strip().lower() == "ya" else "Realisasi tidak tercapai 95%"
                    st.success("Prediksi berhasil dijalankan.")
                    st.write(result_text)
                else:
                    label = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
                    result_text = "Realisasi tercapai 95%" if str(label).strip().lower() == "ya" else "Realisasi tidak tercapai 95%"
                    st.success("Prediksi berhasil dijalankan.")
                    st.write(result_text)
            except Exception as e:
                st.error("Prediksi gagal: " + str(e))
