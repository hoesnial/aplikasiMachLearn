import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==========================================
# GENERATE DATA SINTETIS (FALLBACK)
# Disesuaikan dengan struktur CSV asli:
# usia, jenis_kelamin, gatal, kerontokan_bulu,
# kerak_pada_kulit, hasil_tes_laboratorium
# ==========================================

def generate_synthetic_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)

    usia             = rng.integers(1, 4, size=n)
    jenis_kelamin    = rng.choice(["Female", "Male"], size=n)
    gatal            = rng.choice(["YA", "TIDAK"], size=n, p=[0.55, 0.45])
    kerontokan_bulu  = rng.choice(["YA", "TIDAK"], size=n, p=[0.50, 0.50])
    kerak_pada_kulit = rng.choice(["YA", "TIDAK"], size=n, p=[0.48, 0.52])

    skor = (
        (gatal            == "YA").astype(int) * 3 +
        (kerontokan_bulu  == "YA").astype(int) * 3 +
        (kerak_pada_kulit == "YA").astype(int) * 3
    )
    prob_positif = 1 / (1 + np.exp(-0.8 * (skor - 4)))
    label = np.where(rng.random(n) < prob_positif, "POSITIF", "NEGATIF")

    return pd.DataFrame({
        "usia":                    usia,
        "jenis_kelamin":           jenis_kelamin,
        "gatal":                   gatal,
        "kerontokan_bulu":         kerontokan_bulu,
        "kerak_pada_kulit":        kerak_pada_kulit,
        "hasil_tes_laboratorium":  label,
    })

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    # Cari CSV di beberapa lokasi — kompatibel dengan struktur folder proyek apa pun
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder pages/
    root_dir = os.path.dirname(base_dir)                   # folder root proyek

    kandidat_path = [
        os.path.join(root_dir, "dataset", "scabies-sapi-diperluas.csv"),  # root/dataset/ ← struktur kamu
        os.path.join(base_dir, "dataset", "scabies-sapi-diperluas.csv"),  # pages/dataset/
        os.path.join(root_dir, "scabies-sapi-diperluas.csv"),             # root/
        os.path.join(base_dir, "scabies-sapi-diperluas.csv"),             # pages/
        "scabies-sapi-diperluas.csv",                                      # working directory
    ]

    for filepath in kandidat_path:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            return df, False  # False = bukan sintetis

    return generate_synthetic_data(), True  # True = sintetis

# ==========================================
# TRAINING MODEL
# ==========================================

@st.cache_resource
def train_model_v2(_df):
    """
    Urutan yang benar:
      1. Encode target
      2. Encode semua fitur teks di df_processed
      3. Ambil X dari df_processed (sudah numerik semua)
    Gunakan pd.api.types.is_string_dtype agar kompatibel
    dengan Pandas lama (dtype=object) DAN Pandas baru (dtype=str).
    """
    df_processed = _df.copy()
    le_dict      = {}

    nama_kolom_target = df_processed.columns[-1]  # kolom terakhir = target

    # 1. Encode target
    target_le = LabelEncoder()
    df_processed[nama_kolom_target] = target_le.fit_transform(
        df_processed[nama_kolom_target].astype(str).str.upper()
    )

    feature_names = [c for c in df_processed.columns if c != nama_kolom_target]

    # 2. Encode semua kolom teks — kompatibel Pandas lama & baru
    for col in feature_names:
        if pd.api.types.is_string_dtype(df_processed[col]) or df_processed[col].dtype == "object":
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le

    # 3. Ambil X setelah semua encoding selesai
    X = df_processed[feature_names].copy()
    y = df_processed[nama_kolom_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion="gini")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":              accuracy_score(y_test, y_pred),
        "precision":             precision_score(y_test, y_pred, zero_division=0),
        "recall":                recall_score(y_test, y_pred, zero_division=0),
        "f1_score":              f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    return model, le_dict, target_le, metrics, feature_names

# ==========================================
# ANTARMUKA STREAMLIT
# ==========================================

def show():
    st.markdown("# 🐄 Prediksi Penyakit Scabies pada Sapi")
    st.markdown(
        "Aplikasi prediksi probabilitas diagnosis penyakit **Scabies** pada sapi "
        "berdasarkan gejala klinis menggunakan algoritma **Decision Tree**."
    )
    st.markdown("---")

    df, is_synthetic = load_data()

    if is_synthetic:
        st.warning(
            "⚠️ File **scabies-sapi-diperluas.csv** tidak ditemukan. "
            "Aplikasi berjalan menggunakan **data sintetis** sebagai demonstrasi. "
            "Letakkan file CSV di folder yang sama dengan app.py untuk menggunakan data asli."
        )
    else:
        st.success(f"📂 Dataset berhasil dimuat — **{len(df)} baris data**.")

    with st.spinner("Membangun model Decision Tree..."):
        try:
            model, le_dict, target_le, metrics, feature_names = train_model_v2(df)
            model_trained = True
        except Exception as e:
            st.error(f"Gagal melatih model: {e}")
            model_trained = False

    if not model_trained:
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediksi Gejala",
        "📊 Evaluasi Model",
        "📈 Eksplorasi Data",
        "🔬 Analisis Decision Tree",
    ])

    # ==================== TAB 1: PREDIKSI ====================
    with tab1:
        st.markdown("### 📝 Input Gejala Klinis Sapi")
        st.markdown("Masukkan data sapi di bawah ini untuk mendapatkan hasil prediksi.")

        col1, col2 = st.columns(2)

        with col1:
            usia           = st.number_input(
                "Usia Sapi (Kategori 1=Muda / 2=Dewasa / 3=Tua)",
                min_value=1, max_value=3, value=2, step=1,
                key="input_usia_sapi_unik"
            )
            jenis_kelamin  = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            gatal          = st.selectbox("Mengalami Gatal-Gatal?", ["TIDAK", "YA"])

        with col2:
            kerontokan_bulu  = st.selectbox("Kerontokan Bulu?",    ["TIDAK", "YA"])
            kerak_pada_kulit = st.selectbox("Kerak pada Kulit?",   ["TIDAK", "YA"])

        st.markdown("---")

        if st.button("🔍 Prediksi Scabies", type="primary", use_container_width=True):

            input_raw = {
                "usia":            usia,
                "jenis_kelamin":   jenis_kelamin,
                "gatal":           gatal,
                "kerontokan_bulu": kerontokan_bulu,
                "kerak_pada_kulit":kerak_pada_kulit,
            }
            input_data = pd.DataFrame([{k: input_raw[k] for k in feature_names}])

            # Encode input sesuai LabelEncoder training
            for col in input_data.columns:
                if col in le_dict:
                    le  = le_dict[col]
                    val = str(input_data.at[0, col])
                    input_data[col] = le.transform([val])[0] if val in le.classes_ else 0

            try:
                prediction_encoded = model.predict(input_data)[0]
                proba              = model.predict_proba(input_data)[0]
                prediction_label   = target_le.inverse_transform([prediction_encoded])[0]

                st.markdown("### 📋 Hasil Diagnosis")

                if prediction_label.upper() == "POSITIF":
                    st.error("⚠️ **SAPI DIPREDIKSI POSITIF SCABIES**")
                    st.markdown(
                        "Segera konsultasikan ke dokter hewan dan isolasi sapi "
                        "dari kawanannya untuk mencegah penularan."
                    )
                else:
                    st.success("✅ **SAPI DIPREDIKSI NEGATIF SCABIES**")
                    st.markdown("Sapi diprediksi aman. Tetap jaga kebersihan kandang secara rutin!")

                classes_upper = [c.upper() for c in target_le.classes_]
                idx_negatif = classes_upper.index("NEGATIF") if "NEGATIF" in classes_upper else 0
                idx_positif = classes_upper.index("POSITIF") if "POSITIF" in classes_upper else 1

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Probabilitas Negatif", f"{proba[idx_negatif]*100:.2f}%")
                with col_b:
                    st.metric("Probabilitas Positif", f"{proba[idx_positif]*100:.2f}%")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

    # ==================== TAB 2: EVALUASI ====================
    with tab2:
        st.markdown("### 📊 Evaluasi Model: Decision Tree")

        if is_synthetic:
            st.info("ℹ️ Metrik di bawah dihitung dari **data sintetis**, bukan data asli.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall",    f"{metrics['recall']*100:.2f}%")
        col4.metric("F1-Score",  f"{metrics['f1_score']*100:.2f}%")

        st.markdown("#### 🔢 Confusion Matrix")
        cm_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"Actual: {c}" for c in target_le.classes_],
            columns=[f"Predicted: {c}" for c in target_le.classes_],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### 📑 Classification Report")
        report_rows = []
        for label, vals in metrics["classification_report"].items():
            if isinstance(vals, dict):
                report_rows.append({
                    "Label":     label,
                    "Precision": f"{vals['precision']:.2f}",
                    "Recall":    f"{vals['recall']:.2f}",
                    "F1-Score":  f"{vals['f1-score']:.2f}",
                    "Support":   int(vals["support"]),
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    # ==================== TAB 3: EKSPLORASI ====================
    with tab3:
        st.markdown("### 📈 Eksplorasi Dataset Scabies Sapi")

        if is_synthetic:
            st.info("ℹ️ Dataset ini adalah **data sintetis** yang digenerate otomatis.")

        kolom_target = df.columns[-1]
        positif_count = len(df[df[kolom_target].astype(str).str.upper() == "POSITIF"])
        negatif_count = len(df) - positif_count

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Data",        len(df))
        col2.metric("Jumlah Fitur",       df.shape[1] - 1)
        col3.metric("Kasus Positif",      positif_count)
        col4.metric("Persentase Positif", f"{(positif_count / len(df)) * 100:.1f}%")

        st.markdown("#### 📊 Distribusi Label")
        dist_df = pd.DataFrame({
            "Diagnosis": ["POSITIF", "NEGATIF"],
            "Jumlah":    [positif_count, negatif_count],
        })
        st.bar_chart(dist_df.set_index("Diagnosis"))

        st.markdown("#### 👁️ Preview 20 Data Pertama")
        st.dataframe(df.head(20), use_container_width=True)

    # ==================== TAB 4: ANALISIS ====================
    with tab4:
        st.markdown("### 🔬 Mengapa Decision Tree?")
        st.markdown(
            "Algoritma **Decision Tree** sangat ideal untuk dataset Scabies ini karena:\n\n"
            "- Menangani data kategorikal (YA/TIDAK) tanpa perlu *feature scaling*\n"
            "- Meniru logika diagnosis dokter hewan secara transparan\n"
            "- Hasil mudah dijelaskan kepada peternak tanpa latar belakang teknis\n\n"
            "> *Contoh: Jika sapi **gatal** DAN **bulu rontok** DAN ada **kerak kulit** "
            "→ probabilitas Scabies sangat tinggi.*"
        )

        st.markdown("#### 🌳 Tingkat Kepentingan Fitur")
        st.markdown(
            "Grafik ini menunjukkan seberapa besar kontribusi tiap fitur "
            "terhadap keputusan model."
        )
        importance_df = pd.DataFrame({
            "Fitur":      feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Fitur"))

        st.markdown("#### ℹ️ Keterangan Fitur")
        keterangan = {
            "usia":            "Kategori usia sapi: 1=Muda, 2=Dewasa, 3=Tua",
            "jenis_kelamin":   "Jenis kelamin sapi (Female / Male)",
            "gatal":           "Apakah sapi menunjukkan perilaku gatal-gatal (YA/TIDAK)",
            "kerontokan_bulu": "Apakah terdapat kerontokan bulu (YA/TIDAK)",
            "kerak_pada_kulit":"Apakah terdapat kerak atau sisik pada kulit (YA/TIDAK)",
        }
        ket_rows = [{"Fitur": k, "Keterangan": v} for k, v in keterangan.items() if k in feature_names]
        st.dataframe(pd.DataFrame(ket_rows), use_container_width=True, hide_index=True)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    show()