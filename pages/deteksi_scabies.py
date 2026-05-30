import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# Import Model-Model Machine Learning
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Prediksi Scabies Sapi", layout="wide", page_icon="🐄")

# ==========================================
# GENERATE DATA SINTETIS (FALLBACK)
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    kandidat_path = [
        os.path.join(root_dir, "dataset", "scabies-sapi-diperluas.csv"),
        os.path.join(base_dir, "dataset", "scabies-sapi-diperluas.csv"),
        os.path.join(root_dir, "scabies-sapi-diperluas.csv"),
        os.path.join(base_dir, "scabies-sapi-diperluas.csv"),
        "scabies-sapi-diperluas.csv",
    ]

    for filepath in kandidat_path:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            return df, False

    return generate_synthetic_data(), True

# ==========================================
# TRAINING MODEL UTAMA & SPLIT DATA
# ==========================================
@st.cache_resource
def prepare_and_train(_df):
    df_processed = _df.copy()
    le_dict      = {}
    nama_kolom_target = df_processed.columns[-1]

    # 1. Encode target
    target_le = LabelEncoder()
    df_processed[nama_kolom_target] = target_le.fit_transform(
        df_processed[nama_kolom_target].astype(str).str.upper()
    )

    feature_names = [c for c in df_processed.columns if c != nama_kolom_target]

    # 2. Encode fitur
    for col in feature_names:
        if pd.api.types.is_string_dtype(df_processed[col]) or df_processed[col].dtype == "object":
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le

    # 3. Split Data
    X = df_processed[feature_names].copy()
    y = df_processed[nama_kolom_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model Utama: Decision Tree
    model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion="gini")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":              accuracy_score(y_test, y_pred),
        "precision":             precision_score(y_test, y_pred, zero_division=0),
        "recall":                recall_score(y_test, y_pred, zero_division=0),
        "f1_score":              f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    return model, le_dict, target_le, metrics, feature_names, X_train, X_test, y_train, y_test

# ==========================================
# TRAINING UNTUK KOMPARASI (7 MODEL)
# ==========================================
@st.cache_resource
def train_comparison_models(_X_train, _X_test, _y_train, _y_test):
    models = {
        "Decision Tree 🌟": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Extra Trees": ExtraTreesClassifier(random_state=42, max_depth=5, n_estimators=50),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, max_depth=3, n_estimators=50),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    results = []
    for name, m in models.items():
        m.fit(_X_train, _y_train)
        y_pred = m.predict(_X_test)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(_y_test, y_pred) * 100,
            "Precision": precision_score(_y_test, y_pred, zero_division=0) * 100,
            "Recall": recall_score(_y_test, y_pred, zero_division=0) * 100,
            "F1-Score": f1_score(_y_test, y_pred, zero_division=0) * 100,
        })
    
    df_results = pd.DataFrame(results).sort_values(by=["Accuracy", "F1-Score"], ascending=False)
    return df_results

# ==========================================
# ANTARMUKA STREAMLIT
# ==========================================
def show():
    # CUSTOM CSS
    st.markdown("""
    <style>
        .metric-label { font-weight: bold; color: #555; }
        .insight-box {
            background-color: #e6f3ff;
            border-left: 4px solid #1f77b4;
            padding: 10px 15px;
            margin: 15px 0;
            border-radius: 4px;
            color: #003366;
        }
        .summary-success {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            border-radius: 5px;
            color: #155724;
            margin-bottom: 10px;
        }
        .summary-info {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            border-radius: 5px;
            color: #0c5460;
        }
        .code-box {
            background-color: #272822;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🐄 Prediksi Penyakit Scabies pada Sapi")
    st.markdown(
        "Aplikasi prediksi probabilitas diagnosis penyakit **Scabies** pada sapi "
        "berdasarkan data klinis menggunakan algoritma **Decision Tree**."
    )
    st.markdown("---")

    df, is_synthetic = load_data()

    if is_synthetic:
        st.warning("⚠️ File CSV tidak ditemukan. Menggunakan **data sintetis** sebagai demonstrasi.")
    else:
        st.success(f"📂 Dataset berhasil dimuat — **{len(df)} baris data**.")

    with st.spinner("Membangun model..."):
        try:
            model, le_dict, target_le, metrics, feature_names, X_train, X_test, y_train, y_test = prepare_and_train(df)
            model_trained = True
        except Exception as e:
            st.error(f"Gagal melatih model: {e}")
            model_trained = False

    if not model_trained:
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Prediksi",
        "📊 Evaluasi Model",
        "📈 Eksplorasi Data",
        "🔬 Detail Decision Tree",
        "🧠 Analisis Pemilihan Model"
    ])

    # ==================== TAB 1: PREDIKSI ====================
    with tab1:
        st.markdown("### 📝 Input Gejala Klinis Sapi")
        col1, col2 = st.columns(2)
        with col1:
            usia = st.number_input("Usia Sapi (Kategori 1=Muda / 2=Dewasa / 3=Tua)", 1, 3, 2, 1)
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            gatal = st.selectbox("Mengalami Gatal-Gatal?", ["TIDAK", "YA"])
        with col2:
            kerontokan_bulu = st.selectbox("Kerontokan Bulu?", ["TIDAK", "YA"])
            kerak_pada_kulit = st.selectbox("Kerak pada Kulit?", ["TIDAK", "YA"])

        st.markdown("---")
        if st.button("🔍 Prediksi Scabies", type="primary", use_container_width=True):
            input_raw = {
                "usia": usia, "jenis_kelamin": jenis_kelamin, "gatal": gatal,
                "kerontokan_bulu": kerontokan_bulu, "kerak_pada_kulit": kerak_pada_kulit,
            }
            input_data = pd.DataFrame([{k: input_raw[k] for k in feature_names}])

            for col in input_data.columns:
                if col in le_dict:
                    le = le_dict[col]
                    val = str(input_data.at[0, col])
                    input_data[col] = le.transform([val])[0] if val in le.classes_ else 0

            prediction_encoded = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            prediction_label = target_le.inverse_transform([prediction_encoded])[0]

            st.markdown("### 📋 Hasil Diagnosis")
            if prediction_label.upper() == "POSITIF":
                st.error("⚠️ **SAPI DIPREDIKSI POSITIF SCABIES**")
            else:
                st.success("✅ **SAPI DIPREDIKSI NEGATIF SCABIES**")

            classes_upper = [c.upper() for c in target_le.classes_]
            idx_negatif = classes_upper.index("NEGATIF") if "NEGATIF" in classes_upper else 0
            idx_positif = classes_upper.index("POSITIF") if "POSITIF" in classes_upper else 1

            col_a, col_b = st.columns(2)
            col_a.metric("Probabilitas Negatif", f"{proba[idx_negatif]*100:.2f}%")
            col_b.metric("Probabilitas Positif", f"{proba[idx_positif]*100:.2f}%")

    # ==================== TAB 2: EVALUASI ====================
    with tab2:
        st.markdown("## 📊 Evaluasi Model: Decision Tree")
        st.markdown("Binary classification: 0 = NEGATIF Scabies, 1 = POSITIF Scabies.")
        
        # 1. Metrik Utama
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall",    f"{metrics['recall']*100:.2f}%")
        col4.metric("F1-Score",  f"{metrics['f1_score']*100:.2f}%")

        st.markdown("---")
        
        # 2. Confusion Matrix
        st.markdown("### 🔢 Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual: {c.upper()}" for c in target_le.classes_],
            columns=[f"Predicted: {c.upper()}" for c in target_le.classes_]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Classification Report
        st.markdown("### 📄 Classification Report")
        cr = metrics["classification_report"]
        
        cr_df = pd.DataFrame(cr).transpose()
        for col_name in ['precision', 'recall', 'f1-score']:
            cr_df[col_name] = cr_df[col_name].apply(lambda x: f"{x:.4f}")
        cr_df['support'] = cr_df['support'].astype(int)
        
        st.dataframe(cr_df, use_container_width=True)

    # ==================== TAB 3: EKSPLORASI DATA ====================
    with tab3:
        st.markdown("## 📈 Eksplorasi Dataset Scabies Sapi")
        st.markdown("Bagian ini menyajikan analisis deskriptif dan visualisasi distribusi data untuk memahami karakteristik dataset sebelum diproses oleh model Machine Learning.")
        
        # Row 1: KPI Metrics
        col_met1, col_met2, col_met3 = st.columns(3)
        col_met1.metric("Total Sampel Data", f"{df.shape[0]} Baris")
        col_met2.metric("Jumlah Fitur (Gejala)", f"{df.shape[1] - 1} Kolom")
        col_met3.metric("Kelas Target", len(df['hasil_tes_laboratorium'].unique()))

        st.markdown("---")
        
        # Row 2: Raw Data & Descriptive Stats
        st.markdown("### 1. Cuplikan Data & Statistik Deskriptif")
        st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("Lihat Statistik Deskriptif Dataset"):
            st.dataframe(df.describe(include='all').fillna('-'), use_container_width=True)

        st.markdown("---")

        # Row 3: Visualizations
        st.markdown("### 2. Analisis Visual Distribusi Data")
        
        col_vis1, col_vis2 = st.columns([1, 1.2])
        
        with col_vis1:
            st.markdown("**Proporsi Kelas Target (Diagnosis)**")
            target_counts = df['hasil_tes_laboratorium'].value_counts().reset_index()
            target_counts.columns = ['Diagnosis', 'Jumlah']
            
            fig_pie = px.pie(
                target_counts, names='Diagnosis', values='Jumlah',
                color='Diagnosis', 
                color_discrete_map={'POSITIF': '#EF553B', 'NEGATIF': '#00CC96'},
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.info("💡 **Insight:** Proporsi data yang seimbang (balance) mencegah model dari bias mayoritas kelas saat *training*.")

        with col_vis2:
            st.markdown("**Distribusi Gejala terhadap Hasil Diagnosis**")
            categorical_cols = [col for col in feature_names if df[col].nunique() <= 5]
            if not categorical_cols:
                categorical_cols = feature_names 
                
            selected_feature = st.selectbox("Pilih fitur/gejala untuk dianalisis:", categorical_cols)
            
            fig_bar = px.histogram(
                df, x=selected_feature, color='hasil_tes_laboratorium',
                barmode='group',
                color_discrete_map={'POSITIF': '#EF553B', 'NEGATIF': '#00CC96'}
            )
            fig_bar.update_layout(
                xaxis_title=selected_feature.replace("_", " ").title(),
                yaxis_title="Jumlah Sapi",
                margin=dict(t=20, b=20, l=0, r=0), height=350,
                legend_title="Diagnosis"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ==================== TAB 4: DETAIL DECISION TREE ====================
    with tab4:
        st.markdown("## 🔬 Analisis Mendalam Decision Tree")
        
        col_feat, col_rules = st.columns([1, 1.2])
        
        with col_feat:
            st.markdown("### 🌳 Feature Importance")
            st.markdown(
                "Grafik ini menunjukkan seberapa besar bobot masing-masing gejala "
                "terhadap keputusan akhir model."
            )
            
            importance_df = pd.DataFrame({
                "Fitur": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            
            st.bar_chart(importance_df.set_index("Fitur"))
            
            st.info(
                "💡 **Interpretasi:** Fitur dengan batang tertinggi bertindak sebagai **Root Node** atau pemisah utama, "
                "yaitu gejala yang paling valid membedakan sapi sehat dan sakit."
            )

        with col_rules:
            st.markdown("### 📜 Ekstraksi Aturan (Rules) Medis")
            st.markdown(
                "Bukti bahwa model ini adalah **White-Box**. Kita bisa mengekstrak pola logika "
                "(*If-Else*) yang dipelajari model dari dataset untuk divalidasi oleh dokter hewan:"
            )
            tree_rules = export_text(model, feature_names=feature_names, max_depth=3)
            st.code(tree_rules, language="text")

        st.markdown("---")
        
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.markdown("### 📋 Kamus Fitur (Klinis)")
            
            keterangan_fitur = {
                "usia": "Kategori usia (1=Muda, 2=Dewasa, 3=Tua).",
                "jenis_kelamin": "Seks sapi (Male/Female).",
                "gatal": "Pruritus. Sapi sering menggosokkan badan (YA/TIDAK).",
                "kerontokan_bulu": "Alopecia akibat garukan hebat (YA/TIDAK).",
                "kerak_pada_kulit": "Penebalan kulit di area infeksi (YA/TIDAK)."
            }
            
            df_ket = pd.DataFrame([
                {"Fitur": k, "Keterangan": v} 
                for k, v in keterangan_fitur.items() if k in feature_names
            ])
            st.dataframe(df_ket, use_container_width=True, hide_index=True)
            
        with col_detail2:
            st.markdown("### ⚙️ Pipeline yang Diimplementasi")
            st.markdown("""
            1. **Data Collection** — Memuat dataset CSV.
            2. **Label Encoding** — Mengubah target `POSITIF`/`NEGATIF` menjadi biner `1`/`0`.
            3. **Feature Encoding** — Mengubah data teks gejala menjadi angka.
            4. **Data Splitting** — **80% Data Latih** dan **20% Data Uji**.
            5. **Model Training** — Membangun `DecisionTreeClassifier(max_depth=5, criterion='gini')`.
            6. **Evaluasi** — Ekstraksi *Rules* dan prediksi data uji.
            """)

    # ==================== TAB 5: ANALISIS PEMILIHAN MODEL TERBAIK ====================
    with tab5:
        st.markdown("## 🔬 Analisis Pemilihan Model Terbaik")
        
        st.markdown("### 📐 Metodologi Eksperimen (7 Model Komparasi)")
        st.markdown(
            "Eksperimen diperluas dengan membandingkan **7 algoritma Machine Learning** "
            "(Decision Tree, Extra Trees, Gradient Boosting, Logistic Regression, SVM, KNN, Naive Bayes). "
            "Evaluasi ditekankan pada keseimbangan **Akurasi Tinggi** dan **Interpretabilitas Medis**."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🏆 Hasil Eksperimen Empiris")
        df_comp = train_comparison_models(X_train, X_test, y_train, y_test)
        
        df_comp_styled = df_comp.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            df_comp_styled[col] = df_comp_styled[col].apply(lambda x: f"{x:.2f}%")
            
        st.dataframe(df_comp_styled, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📍 Visualisasi Performa Metrik")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**2D Performance Map: Accuracy vs F1-Score**")
            fig1 = px.scatter(
                df_comp, x="Accuracy", y="F1-Score", text="Model",
                color="Model", size=[15]*len(df_comp),
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig1.update_traces(textposition='top center', textfont_size=12)
            fig1.update_layout(showlegend=False, height=450, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            st.markdown("**Perbandingan Metrik Utama per Metode**")
            df_melted = df_comp.melt(id_vars="Model", var_name="Metrik", value_name="Skor (%)")
            fig2 = px.line(
                df_melted, x="Model", y="Skor (%)", color="Metrik", markers=True,
                color_discrete_sequence=["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
            )
            fig2.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 📌 Catatan Head-to-Head: Kenapa Bertahan dengan Decision Tree?")

        # 5. Expanders (Head to Head Analysis)
        with st.expander("🧠 Decision Tree vs Logistic Regression"):
            st.markdown("- **Logistic Regression** bekerja linear sehingga kurang optimal menangkap interaksi multi-gejala (misal: gabungan gatal dan kerak kulit yang memicu scabies akut).")
            st.markdown("- **Decision Tree** mampu menangani hubungan non-linear dan persilangan gejala dengan mudah.")
            st.markdown('<div class="insight-box"><b>Insight:</b> Dataset scabies lebih condong pada logika *rule-based* klinis sehingga pohon keputusan (Decision Tree) jauh lebih alami dibanding garis pemisah regresi linier.</div>', unsafe_allow_html=True)
            st.markdown("**Penjelasan:** Logistic regression mengukur bobot setiap fitur secara independen, sedangkan Decision Tree mengevaluasi fitur secara kondisional (tergantung hasil fitur sebelumnya).")

        with st.expander("🧠 Decision Tree vs Extra Trees & Ensemble (XGBoost)"):
            st.markdown("- **Extra Trees** (Extremely Randomized Trees) membangun puluhan/ratusan pohon secara paralel dengan split acak (*ensemble*), membuat akurasinya seringkali naik 1-3%.")
            st.markdown("- Namun, hasil prediksinya menjadi **Black-Box** (sulit dilacak alur logikanya).")
            st.markdown("- **Decision Tree** (Single Tree) mengorbankan margin akurasi kecil demi 100% transparansi alur diagnosis medis.")
            st.markdown('<div class="insight-box"><b>Insight:</b> Untuk deployment di klinik peternakan, kepercayaan dokter (berasal dari <i>interpretabilitas</i>) jauh lebih penting dari akurasi yang sekadar beda persentase desimal. Overkill menggunakan Extra Trees pada dataset ini.</div>', unsafe_allow_html=True)
            st.markdown("**Penjelasan:** Extra Trees melakukan agregat (voting) dari banyak pohon dengan split acak. Walaupun sangat robust, *trade-off*-nya adalah pengguna tidak bisa lagi menggambar satu pohon logika yang dapat divalidasi secara medis.")

        with st.expander("🧠 Decision Tree vs KNN"):
            st.markdown("- **KNN** sensitif terhadap fungsi jarak Euclidean. Fitur kategorial (YA=1, TIDAK=0) seringkali menjadi bias jika tidak diatur metrik jaraknya.")
            st.markdown("- Waktu eksekusi prediksi KNN bergantung pada jumlah data latih (*Lazy Learner*).")
            st.markdown("- **Decision Tree** sama sekali kebal terhadap perbedaan skala (scaling).")
            st.markdown('<div class="insight-box"><b>Insight:</b> KNN kurang cocok untuk diagnosis medis berbasis checkbox (kategorial), sementara Decision Tree menangani kategorisasi biner dengan sempurna.</div>', unsafe_allow_html=True)

        with st.expander("🧠 Decision Tree vs SVM"):
            st.markdown("- **SVM** sangat kuat pada ruang dimensi tinggi (ratusan fitur).")
            st.markdown("- Pada dataset sederhana dengan < 10 fitur klinis, kalkulasi *hyperplane* SVM menjadi tidak efisien.")
            st.markdown('<div class="insight-box"><b>Insight:</b> Menggunakan SVM pada data diagnosis gejala simpel ibarat menggunakan mesin jet untuk menyeberang jalan. Decision Tree lebih efisien dan <i>to the point</i>.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📝 Kesimpulan Analisis")
        
        st.markdown("""
        <div class="summary-success">
            <b>Kesimpulan Final:</b> Dengan membandingkan 7 model klasifikasi, <b>Decision Tree</b> membuktikan dirinya sebagai algoritma yang paling <i>Fit-for-Purpose</i> (tepat guna). Ia memberikan akurasi klasifikasi yang bersaing ketat dengan model level lanjut (seperti Gradient Boosting), namun tetap memiliki kemampuan eksklusif untuk menerjemahkan matriks datanya menjadi <b>Aturan Medis (Clinical Rules)</b> yang valid dan dapat dipertanggungjawabkan di lapangan.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()