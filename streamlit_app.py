import pickle
from pathlib import Path

import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Dashboard Realisasi Anggaran",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi (1).csv")
MODEL_PATH = Path("model/Best_model.pkcls")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, "File model tidak ditemukan di folder model."

    try:
        import Orange  # noqa: F401
    except ModuleNotFoundError:
        return None, "Modul Orange tidak ditemukan. Tambahkan `Orange3` ke requirements dan jalankan `pip install -r requirements.txt`."

    try:
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as exc:
        return None, f"Model gagal dimuat: {exc}"


def normalize_prediction(value):
    if isinstance(value, (list, tuple, pd.Series)):
        if len(value) > 0:
            value = value[0]
    return str(value).strip().lower()


def prediction_label(raw):
    normalized = normalize_prediction(raw)
    if normalized in {"ya", "yes", "1", "true", "True"}:
        return "Realisasi tercapai 95%"
    return "Realisasi tidak tercapai 95%"


data = load_data()
model, model_error = load_model()

st.title("Dashboard Analitik Realisasi Anggaran")
st.markdown(
    "Aplikasi ini menampilkan ringkasan dataset, visualisasi, dan prediksi realisasi pencapaian 95% berdasarkan model yang tersedia."
)

with st.sidebar:
    st.header("Filter Data")
    tipe_options = ["Semua"] + sorted(data["tipe_satker"].dropna().unique())
    prov_options = ["Semua"] + sorted(data["provinsi"].dropna().unique())
    jenis_options = ["Semua"] + sorted(data["jenis_belanja_utama"].dropna().unique())

    selected_tipe = st.selectbox("Tipe Satker", tipe_options)
    selected_prov = st.selectbox("Provinsi", prov_options)
    selected_jenis = st.selectbox("Jenis Belanja Utama", jenis_options)

    st.markdown("---")
    st.write(
        "Gunakan filter untuk melihat subset data yang relevan. Jika model tidak dapat dimuat, periksa instalasi `Orange3`."
    )

filtered_data = data.copy()
if selected_tipe != "Semua":
    filtered_data = filtered_data[filtered_data["tipe_satker"] == selected_tipe]
if selected_prov != "Semua":
    filtered_data = filtered_data[filtered_data["provinsi"] == selected_prov]
if selected_jenis != "Semua":
    filtered_data = filtered_data[filtered_data["jenis_belanja_utama"] == selected_jenis]

tab_overview, tab_visual, tab_predict = st.tabs(["Overview", "Visualisasi", "Prediksi"])

with tab_overview:
    st.header("Ringkasan Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Satker", len(filtered_data))
    ratio_95 = 100 * (filtered_data["realisasi_tercapai_95persen"] == "Ya").mean()
    col2.metric("Rasio Tercapai 95%", f"{ratio_95:.1f}%")
    col3.metric("Median Skor IKPA", f"{filtered_data['skor_ikpa'].median():.2f}")
    col4.metric("Median Deviasi RPD", f"{filtered_data['deviasi_rpd_persen'].median():.2f}%")

    st.markdown("#### Statistik Numerik")
    st.dataframe(
        filtered_data[
            [
                "pagu_miliar",
                "jumlah_pegawai",
                "jumlah_spm",
                "revisi_dipa",
                "realisasi_tw1_persen",
                "realisasi_tw2_persen",
                "realisasi_tw3_persen",
                "deviasi_rpd_persen",
                "skor_ikpa",
            ]
        ]
        .describe()
        .T
        .style.format("{:.2f}")
    )

    st.markdown("#### Preview Data")
    st.dataframe(filtered_data.head(15))

with tab_visual:
    st.header("Visualisasi Utama")

    target_counts = (
        filtered_data["realisasi_tercapai_95persen"]
        .value_counts()
        .reindex(["Ya", "Tidak"])
        .fillna(0)
    )

    st.subheader("Distribusi Target Realisasi 95%")
    bar_fig = px.bar(
        x=target_counts.index,
        y=target_counts.values,
        color=target_counts.index,
        color_discrete_map={"Ya": "#2ca02c", "Tidak": "#d62728"},
        labels={"x": "Realisasi 95%", "y": "Jumlah Satker"},
        title="Distribusi Realisasi 95%",
    )
    bar_fig.update_layout(showlegend=False)
    st.plotly_chart(bar_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        hist_ikpa = px.histogram(
            filtered_data,
            x="skor_ikpa",
            nbins=20,
            title="Distribusi Skor IKPA",
            labels={"skor_ikpa": "Skor IKPA", "count": "Frekuensi"},
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(hist_ikpa, use_container_width=True)

    with col2:
        hist_deviasi = px.histogram(
            filtered_data,
            x="deviasi_rpd_persen",
            nbins=20,
            title="Distribusi Deviasi RPD (%)",
            labels={"deviasi_rpd_persen": "Deviasi RPD (%)", "count": "Frekuensi"},
            color_discrete_sequence=["#ff7f0e"],
        )
        st.plotly_chart(hist_deviasi, use_container_width=True)

    st.subheader("IKPA vs Deviasi RPD")
    scatter_fig = px.scatter(
        filtered_data,
        x="skor_ikpa",
        y="deviasi_rpd_persen",
        color="realisasi_tercapai_95persen",
        color_discrete_map={"Ya": "#2ca02c", "Tidak": "#d62728"},
        hover_data=["tipe_satker", "provinsi", "pagu_miliar"],
        labels={"skor_ikpa": "Skor IKPA", "deviasi_rpd_persen": "Deviasi RPD (%)"},
        title="IKPA vs Deviasi RPD",
        height=450,
    )
    scatter_fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Histogram Dinamis")
    hist_columns = [
        "skor_ikpa",
        "deviasi_rpd_persen",
        "pagu_miliar",
        "jumlah_pegawai",
        "jumlah_spm",
    ]
    hist_column = st.selectbox("Pilih variabel untuk histogram", hist_columns, index=0)
    hist_by = st.selectbox(
        "Kelompokkan berdasarkan",
        ["Tanpa Pemisahan", "Realisasi 95%", "Tipe Satker"],
        index=0,
    )

    if hist_by == "Tanpa Pemisahan":
        hist_fig = px.histogram(
            filtered_data,
            x=hist_column,
            nbins=20,
            title=f"Histogram {hist_column.replace('_', ' ').title()}",
            labels={hist_column: hist_column.replace('_', ' ').title(), "count": "Frekuensi"},
            color_discrete_sequence=["#4c78a8"],
        )
    elif hist_by == "Realisasi 95%":
        hist_fig = px.histogram(
            filtered_data,
            x=hist_column,
            color="realisasi_tercapai_95persen",
            barmode="overlay",
            nbins=20,
            title=f"Histogram {hist_column.replace('_', ' ').title()} per Realisasi 95%",
            labels={hist_column: hist_column.replace('_', ' ').title(), "count": "Frekuensi"},
            color_discrete_map={"Ya": "#2ca02c", "Tidak": "#d62728"},
            opacity=0.7,
        )
    else:
        hist_fig = px.histogram(
            filtered_data,
            x=hist_column,
            color="tipe_satker",
            barmode="overlay",
            nbins=20,
            title=f"Histogram {hist_column.replace('_', ' ').title()} per Tipe Satker",
            labels={hist_column: hist_column.replace('_', ' ').title(), "count": "Frekuensi"},
            opacity=0.6,
        )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("#### Rata-rata per Tipe Satker")
    grouped = (
        filtered_data.groupby("tipe_satker")
        .agg(
            skor_ikpa_mean=("skor_ikpa", "mean"),
            deviasi_rpd_mean=("deviasi_rpd_persen", "mean"),
            ratio_95=("realisasi_tercapai_95persen", lambda x: (x == "Ya").mean() * 100),
        )
        .round(2)
        .sort_values("ratio_95", ascending=False)
        .reset_index()
    )

    avg_chart = alt.Chart(grouped).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("tipe_satker:N", sort="-y", title="Tipe Satker"),
        y=alt.Y("ratio_95:Q", title="Persentase Tercapai 95%"),
        color=alt.Color("ratio_95:Q", scale=alt.Scale(scheme="greens"), legend=None),
        tooltip=["tipe_satker", "skor_ikpa_mean", "deviasi_rpd_mean", "ratio_95"],
    ).properties(height=360)
    st.altair_chart(avg_chart, use_container_width=True)

    st.dataframe(
        grouped.style.format(
            {"ratio_95": "{:.1f}%", "skor_ikpa_mean": "{:.2f}", "deviasi_rpd_mean": "{:.2f}"}
        )
    )

with tab_predict:
    st.header("Prediksi Realisasi 95%")

    if model is None:
        st.error("Model tidak dapat dimuat. " + (model_error or "Periksa file model atau dependensi."))
        if model_error and "Orange" in model_error:
            st.info("Tambahkan `Orange3` ke `requirements.txt` dan jalankan `pip install -r requirements.txt`.")
    else:
        with st.form("prediction_form"):
            left, right = st.columns(2)
            with left:
                jumlah_spm = st.number_input(
                    "Jumlah SPM",
                    min_value=0,
                    value=int(data["jumlah_spm"].median()),
                    step=1,
                    format="%d",
                )
                revisi_dipa = st.number_input(
                    "Revisi DIPA",
                    min_value=0,
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
            with right:
                skor_ikpa = st.number_input(
                    "Skor IKPA",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(round(data["skor_ikpa"].median(), 2)),
                    step=0.01,
                    format="%.2f",
                )
                tipe_options = sorted(data["tipe_satker"].dropna().unique())
                tipe_satker = st.selectbox("Tipe Satker", tipe_options)

            submit = st.form_submit_button("Hitung Prediksi")

        if submit:
            feature_row = {
                "jumlah_spm": jumlah_spm,
                "revisi_dipa": revisi_dipa,
                "deviasi_rpd_persen": deviasi_rpd,
                "skor_ikpa": skor_ikpa,
            }
            for category in tipe_options:
                feature_row[f"tipe_satker={category}"] = 1 if tipe_satker == category else 0

            input_data = pd.DataFrame([feature_row])

            prediction = None
            error = None
            try:
                if hasattr(model, "predict"):
                    prediction = model.predict(input_data)
                else:
                    prediction = model(input_data)
            except Exception as exc:
                try:
                    import Orange  # noqa: F401
                    if hasattr(model, "domain"):
                        input_table = Orange.data.Table(model.domain, input_data.values)
                        prediction = model(input_table)
                    else:
                        raise
                except Exception as fallback_exc:
                    error = f"{exc} | fallback: {fallback_exc}"

            if error:
                st.error("Prediksi gagal: " + error)
            else:
                st.success(prediction_label(prediction))
                st.markdown("**Input prediksi**")
                st.json(feature_row)
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(input_data)
                        st.markdown("**Probabilitas**")
                        st.write(proba)
                    except Exception:
                        pass

st.caption(
    "Data sumber: `02_realisasi_anggaran_klasifikasi (1).csv`. Model sumber: `Best_model.pkcls`."
)
