"""
Halaman Deteksi Pasien Treatment
Dataset: patient_treatment.csv
Sistem Klasifikasi Cerdas Rujukan Perawatan Pasien
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset/patient_treatment.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("patient_treatment.csv")
        except FileNotFoundError:
            st.error("❌ File 'patient_treatment.csv' tidak ditemukan. Pastikan file ada di folder 'dataset/'")
            st.stop()
    return df

@st.cache_data
def preprocess(df):
    df2 = df.copy()

    # 1. Penanganan Missing Values menggunakan Median
    num_cols = df2.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df2[c].fillna(df2[c].median(), inplace=True)

    # 2. Encoding Variabel Kategorikal
    df2["SEX_M"] = (df2["SEX"] == "M").astype(int)
    df2.drop("SEX", axis=1, inplace=True)

    # 3. Handling Outlier dengan IQR Capping (1% - 99%)
    feat_cols = [c for c in df2.columns if c != "SOURCE"]
    for c in feat_cols:
        Q1, Q3 = df2[c].quantile(0.01), df2[c].quantile(0.99)
        df2[c] = df2[c].clip(Q1, Q3)

    X = df2.drop("SOURCE", axis=1)
    y = df2["SOURCE"]

    return X, y, df2

@st.cache_resource
def preprocess_for_training(df):
    """
    Preprocessing khusus untuk training dengan scaling dan SMOTE balancing.
    Tahap 1 Improvement:
    - StandardScaler: Normalisasi fitur untuk konsistensi
    - SMOTE: Balance kelas untuk mengatasi imbalance Outpatient vs Inpatient
    """
    X, y, _ = preprocess(df)
    
    # 4. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Class Balancing dengan SMOTE (Synthetic Minority Oversampling Technique)
    # Ini penting karena dataset mungkin imbalanced antara Rawat Jalan vs Rawat Inap
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    
    st.sidebar.info(f"📊 Data setelah SMOTE:\n• Original: {len(X)} samples\n• Balanced: {len(X_balanced)} samples")
    
    return X_balanced, y_balanced, scaler, X

@st.cache_resource
def build_models(df):
    # Gunakan data yang sudah di-scale dan di-balance dengan SMOTE
    X_balanced, y_balanced, scaler, X_original = preprocess_for_training(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"),
            "use_scaler": False  # ✅ Data sudah di-scale di preprocessing
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=10, min_samples_split=5, min_samples_leaf=2,
                class_weight="balanced", random_state=42
            ),
            "use_scaler": False
        },
        "Weighted KNN": {
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski"),
            "use_scaler": False  # ✅ Data sudah di-scale di preprocessing
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "use_scaler": False
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
            ),
            "use_scaler": False
        },
    }

    results = {}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in models_config.items():
        mdl = config["model"]
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_te = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        mdl.fit(X_tr, y_train)
        y_pred = mdl.predict(X_te)
        y_prob = mdl.predict_proba(X_te)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc    = auc(fpr, tpr)
        cm         = confusion_matrix(y_test, y_pred)
        report     = classification_report(y_test, y_pred, output_dict=True)
        cv_scores  = cross_val_score(mdl, X_tr, y_train, cv=cv_strategy, scoring="accuracy")

        results[name] = {
            "model":    mdl,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "y_test":   y_test,
            "fpr":      fpr,
            "tpr":      tpr,
            "auc":      roc_auc,
            "cm":       cm,
            "report":   report,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision":precision_score(y_test, y_pred),
            "recall":   recall_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred),
            "use_scaler": False
        }

    return results, scaler, X_original, X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────────
# HELPER PLOTS
# ─────────────────────────────────────────────────
COLORS = {
    "Logistic Regression": "#0d6efd",
    "Decision Tree":       "#6610f2",
    "Weighted KNN":        "#0dcaf0",
    "Extra Trees":         "#20c997",
    "XGBoost":             "#fd7e14",
}

def plot_confusion_matrix(cm, model_name):
    labels = ["Outpatient (0)", "Inpatient (1)"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=labels, y=labels,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(height=350, font=dict(size=12), coloraxis_showscale=False)
    return fig

def plot_roc_all(results):
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"],
            name=f'{name} (AUC={r["auc"]:.4f})',
            line=dict(width=2.5, color=COLORS[name])
        ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="gray", width=1.5),
        showlegend=False
    ))
    fig.update_layout(
        title="Kurva ROC-AUC Komparatif Seluruh Model",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400, legend=dict(x=.6, y=.08),
        margin=dict(t=50, b=40, l=50, r=20)
    )
    return fig

def plot_metrics_bar(results):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=[r[m] for m in metrics],
            marker_color=COLORS[name],
            text=[f'{r[m]*100:.1f}%' for m in metrics],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", title="Perbandingan Metrik Evaluasi Pengujian",
        yaxis=dict(range=[0, 1.2], tickformat=".0%"),
        height=400, legend=dict(orientation="h", y=-0.18),
        margin=dict(t=50, b=80, l=50, r=20)
    )
    return fig

def show():
    st.markdown("# 💊 Deteksi Pasien Treatment")
    st.markdown(
        "Sistem klasifikasi untuk memprediksi kebutuhan rujukan perawatan pasien "
        "(Rawat Jalan/Inpatient) berdasarkan parameter laboratorium klinis menggunakan "
        "algoritma Machine Learning."
    )
    st.markdown("---")

    # Tabs untuk navigasi
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "📈 Eksplorasi Data", "🎯 Pemilihan Model"])

    # Sidebar untuk pemilihan model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pengaturan Model")

    # Load data dan build models
    df = load_data()
    results, scaler, X, X_train, X_test, y_train, y_test = build_models(df)

    # Load data dan build models
    df = load_data()
    results, scaler, X, X_train, X_test, y_train, y_test = build_models(df)

    # Model terbaik berdasarkan akurasi
    best_model_name = max(results.items(), key=lambda item: item[1]["accuracy"])[0]
    st.sidebar.success(f"🏆 Model Terbaik: **{best_model_name}**")

    # Pemilihan model
    selected_model_name = st.sidebar.selectbox(
        "Pilih Model Klasifikasi:",
        list(results.keys()),
        help="Pilih algoritma klasifikasi yang ingin digunakan untuk prediksi."
    )

    # ==================== TAB 1: PREDIKSI ====================
    with tab1:
        st.markdown("### 📝 Input Parameter Laboratorium Klinis")
        st.markdown("Masukkan parameter lab pasien di bawah ini untuk prediksi rujukan perawatan.")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            haematocrit  = st.number_input("🩸 Haematocrit (%)",     min_value=0.0, max_value=70.0,  value=38.0, step=0.1)
            haemoglobins = st.number_input("🩸 Haemoglobins (g/dL)", min_value=0.0, max_value=20.0,  value=12.5, step=0.1)
            erythrocyte  = st.number_input("🔴 Erythrocyte (M/µL)",  min_value=0.0, max_value=10.0,  value=4.5,  step=0.01)
        
        with col2:
            leucocyte    = st.number_input("⚪ Leucocyte (K/µL)",    min_value=0.0, max_value=50.0,  value=8.0,  step=0.1)
            thrombocyte  = st.number_input("🟡 Thrombocyte (K/µL)",  min_value=0,   max_value=800,   value=250,  step=1)
            mch          = st.number_input("🔬 MCH (pg)",            min_value=0.0, max_value=50.0,  value=28.0, step=0.1)
        
        with col3:
            mchc         = st.number_input("🔬 MCHC (g/dL)",         min_value=0.0, max_value=40.0,  value=33.0, step=0.1)
            mcv          = st.number_input("🔬 MCV (fL)",            min_value=0.0, max_value=150.0, value=85.0, step=0.1)
            age          = st.number_input("👤 Usia (Tahun)",        min_value=1,   max_value=120,   value=45,   step=1)

        sex = st.selectbox("⚧ Jenis Kelamin", ["M", "F"])

        st.markdown("---")

        # Tombol prediksi
        if st.button("🔍 Prediksi Rujukan Perawatan", type="primary", use_container_width=True):
            sex_m = 1 if sex == "M" else 0
            inp = np.array([[haematocrit, haemoglobins, erythrocyte, leucocyte,
                             thrombocyte, mch, mchc, mcv, age, sex_m]])
            
            inp_final = scaler.transform(inp) if results[selected_model_name]["use_scaler"] else inp

            mdl = results[selected_model_name]["model"]
            pred = mdl.predict(inp_final)[0]
            prob = mdl.predict_proba(inp_final)[0]

            st.markdown("### 📋 Hasil Prediksi")

            if pred == 1:
                st.error("🏨 **KEPUTUSAN: RAWAT INAP (Inpatient)**")
                st.markdown(
                    "Berdasarkan parameter laboratorium yang dimasukkan, model memprediksi "
                    "bahwa pasien **memerlukan perawatan intensif di rumah sakit**. "
                    "Segera konsultasikan dengan dokter."
                )
            else:
                st.success("🏃 **KEPUTUSAN: RAWAT JALAN (Outpatient)**")
                st.markdown(
                    "Berdasarkan parameter laboratorium yang dimasukkan, model memprediksi "
                    "bahwa pasien **dapat melakukan perawatan mandiri di rumah**. "
                    "Tetap lakukan pemantauan berkala."
                )

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Probabilitas Rawat Jalan (0)", f"{prob[0]*100:.2f}%")
            with col_b:
                st.metric("Probabilitas Rawat Inap (1)", f"{prob[1]*100:.2f}%")

            st.info(f"🤖 Model yang digunakan: **{selected_model_name}** | Akurasi: **{results[selected_model_name]['accuracy']*100:.2f}%**")

    # ==================== TAB 2: EVALUASI MODEL ====================
    with tab2:
        st.markdown(f"### 📊 Evaluasi Model: {selected_model_name}")

        # Metrik utama
        col1, col2, col3, col4, col5 = st.columns(5)
        r_sel = results[selected_model_name]

        with col1:
            st.metric("Accuracy", f"{r_sel['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{r_sel['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{r_sel['recall']*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{r_sel['f1']*100:.2f}%")
        with col5:
            st.metric("ROC AUC", f"{r_sel['auc']:.4f}")

        st.markdown("---")

        # Confusion Matrix
        st.markdown("#### 🔢 Confusion Matrix")
        cm = r_sel['cm']
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Outpatient (0)", "Actual: Inpatient (1)"],
            columns=["Predicted: Outpatient (0)", "Predicted: Inpatient (1)"]
        )
        st.dataframe(cm_df, use_container_width=True)

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.plotly_chart(plot_confusion_matrix(cm, selected_model_name), use_container_width=True, key="tab2_cm")
        with col_roc:
            st.plotly_chart(plot_roc_all(results), use_container_width=True, key="tab2_roc")

        st.markdown("---")

        # Classification Report
        st.markdown("#### 📄 Classification Report")
        rep_df = pd.DataFrame(r_sel["report"]).T[["precision","recall","f1-score","support"]].round(4)
        st.dataframe(rep_df, use_container_width=True)

        st.markdown("---")

        # Feature Importance
        st.markdown("#### 🎯 Analisis Fitur")
        mdl_obj = r_sel["model"]
        
        if selected_model_name == "Logistic Regression":
            odds_ratio = np.exp(mdl_obj.coef_[0])
            feat_imp = pd.DataFrame({
                "Feature": X.columns, 
                "Importance": np.abs(mdl_obj.coef_[0]), 
                "Odds Ratio": odds_ratio
            }).sort_values("Importance", ascending=True)
            
            fig_fi = px.bar(feat_imp, x="Importance", y="Feature", orientation="h", 
                           title="Bobot Kontribusi Absolut Fitur", color="Importance", 
                           color_continuous_scale="Blues")
            st.plotly_chart(fig_fi, use_container_width=True, key="tab2_lr_feat")
            st.dataframe(feat_imp.sort_values("Importance", ascending=False).set_index("Feature"), use_container_width=True)
            
        elif selected_model_name in ["Decision Tree", "Extra Trees", "XGBoost"]:
            feat_imp = pd.DataFrame({
                "Feature": X.columns, 
                "Importance": mdl_obj.feature_importances_
            }).sort_values("Importance", ascending=True)
            
            if selected_model_name == "Decision Tree":
                color_scale = "Purples"
                title_text = "Decision Tree Feature Importance (Gini)"
            elif selected_model_name == "Extra Trees":
                color_scale = "Greens"
                title_text = "Extra Trees Feature Importance"
            else:
                color_scale = "Oranges"
                title_text = "XGBoost Feature Importance (Gini)"
            
            fig_fi = px.bar(feat_imp, x="Importance", y="Feature", orientation="h", 
                           title=title_text, color="Importance", 
                           color_continuous_scale=color_scale)
            st.plotly_chart(fig_fi, use_container_width=True, key="tab2_tree_feat")
            st.dataframe(feat_imp.sort_values("Importance", ascending=False).set_index("Feature"), use_container_width=True)

        st.markdown("---")

        # Perbandingan semua model
        st.markdown("### 🏆 Perbandingan Semua Model")
        
        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']*100:.2f}%",
                "Precision": f"{r['precision']*100:.2f}%",
                "Recall": f"{r['recall']*100:.2f}%",
                "F1-Score": f"{r['f1']*100:.2f}%",
                "AUC-ROC": f"{r['auc']:.4f}",
                "CV Mean": f"{r['cv_mean']*100:.2f}%"
            })
        
        comparison_df = pd.DataFrame(rows)
        st.dataframe(comparison_df.set_index("Model"), use_container_width=True)
        
        st.plotly_chart(plot_metrics_bar(results), use_container_width=True, key="tab2_metrics_bar")

    # ==================== TAB 3: EKSPLORASI DATA ====================
    with tab3:
        st.markdown("### 📈 Eksplorasi Dataset Pasien Treatment")

        # Info dataset
        st.markdown("#### 📋 Informasi Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", f"{len(df):,}")
        with col2:
            st.metric("Total Fitur", f"{X.shape[1]}")
        with col3:
            st.metric("Outpatient", f"{(df.SOURCE==0).sum():,}")
        with col4:
            st.metric("Inpatient", f"{(df.SOURCE==1).sum():,}")

        st.markdown("---")

        st.markdown("#### 🧾 Penjelasan Singkat Fitur")
        feature_descriptions = {
            "HAEMATOCRIT": "Persentase volume sel darah merah terhadap seluruh volume darah.",
            "HAEMOGLOBINS": "Kadar hemoglobin dalam darah, indikator oksigenasi dan anemia.",
            "ERYTHROCYTE": "Jumlah sel darah merah per volume darah.",
            "LEUCOCYTE": "Jumlah sel darah putih yang menunjukkan respons imun dan infeksi.",
            "THROMBOCYTE": "Jumlah trombosit dalam darah, penting untuk pembekuan.",
            "MCH": "Rata-rata massa hemoglobin per sel darah merah (picogram).",
            "MCHC": "Konsentrasi hemoglobin rata-rata dalam sel darah merah.",
            "MCV": "Volume rata-rata sel darah merah, menunjukkan ukuran sel.",
            "AGE": "Usia pasien dalam tahun.",
            "SEX": "Jenis kelamin pasien (M/F).",
            "SOURCE": "Target klasifikasi: 0=Rawat Jalan, 1=Rawat Inap."
        }
        desc_df = pd.DataFrame(
            [(k, v) for k, v in feature_descriptions.items()],
            columns=["Fitur", "Deskripsi Singkat"]
        )
        st.dataframe(desc_df, use_container_width=True)

        st.markdown("---")

        # Preview data
        st.markdown("#### 👀 Preview Data")
        num_rows_preview = st.slider("Jumlah baris untuk ditampilkan:", min_value=5, max_value=50, value=10, step=5)
        
        df_preview = df.sample(num_rows_preview, random_state=42).copy()
        df_preview.insert(0, "NO", range(1, len(df_preview) + 1))
        
        st.dataframe(df_preview, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Metadata
        with st.expander("ℹ️ Metadata Skema & Info Kolom"):
            meta_cols = []
            for col in df.columns:
                meta_cols.append({
                    "Nama Fitur": col,
                    "Tipe Data": str(df[col].dtype),
                    "Non-Null": df[col].notnull().sum(),
                    "Missing": df[col].isnull().sum()
                })
            st.dataframe(pd.DataFrame(meta_cols), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Distribusi target
        st.markdown("#### 🎯 Distribusi Target (Rujukan Perawatan)")
        col1, col2 = st.columns(2)
        
        with col1:
            vc = df["SOURCE"].value_counts().reset_index()
            vc.columns = ["Kelas","Count"]
            vc["Kelas"] = vc["Kelas"].map({0:"Rawat Jalan (0)", 1:"Rawat Inap (1)"})
            fig_pie = px.pie(vc, values="Count", names="Kelas", 
                            color_discrete_sequence=["#198754","#dc3545"], 
                            hole=0.3, title="Rasio Keseimbangan Kelas Target")
            st.plotly_chart(fig_pie, use_container_width=True, key="tab3_pie")
        
        with col2:
            sc = df.groupby(["SEX","SOURCE"]).size().reset_index(name="Count")
            sc["SOURCE"] = sc["SOURCE"].map({0:"Rawat Jalan", 1:"Rawat Inap"})
            fig_bar = px.bar(sc, x="SEX", y="Count", color="SOURCE", barmode="group",
                            color_discrete_sequence=["#198754","#dc3545"], 
                            title="Distribusi Kebutuhan Rawat per Gender")
            st.plotly_chart(fig_bar, use_container_width=True, key="tab3_bar")

        st.markdown("---")

        # Statistik deskriptif
        st.markdown("#### 📊 Statistik Deskriptif")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

        st.markdown("---")

        # Outlier detection
        st.markdown("#### 📦 Deteksi Outliers & Sebaran Fitur")
        num_feat = ["HAEMATOCRIT","HAEMOGLOBINS","ERYTHROCYTE","LEUCOCYTE","THROMBOCYTE","MCH","MCHC","MCV","AGE"]
        sel_feat_eda = st.selectbox("Pilih Parameter Fitur Lab:", num_feat)
        
        df_box = df.copy()
        df_box["Kategori Rujukan"] = df_box["SOURCE"].map({0:"Rawat Jalan", 1:"Rawat Inap"})
        
        fig_hist = px.histogram(df_box, x=sel_feat_eda, color="Kategori Rujukan", 
                                color_discrete_map={"Rawat Jalan":"#198754","Rawat Inap":"#dc3545"},
                                barmode="overlay", marginal="box", opacity=0.6,
                                title=f"Distribusi & Outliers: {sel_feat_eda}")
        st.plotly_chart(fig_hist, use_container_width=True, key="tab3_hist")

        st.markdown("---")

        # Multikolinieritas
        st.markdown("#### 🔗 Analisis Multikolinieritas (VIF)")
        
        X_vif, _, _ = preprocess(df)
        vif_df = pd.DataFrame({
            "Fitur": X_vif.columns,
            "VIF": [variance_inflation_factor(X_vif.values.astype(float), i) for i in range(X_vif.shape[1])]
        }).sort_values("VIF", ascending=False).reset_index(drop=True)

        def color_vif(val):
            if val >= 10:
                return "background-color: #FF1F1F; color: black; font-weight: bold;"
            elif val >= 5:
                return "background-color: #F1FF33; color: black; font-weight: bold;"
            return "background-color: #33FF58; color: black; font-weight: bold;"

        st.dataframe(vif_df.style.map(color_vif, subset=["VIF"]).format({"VIF": "{:.2f}"}), use_container_width=True)
        
        fig_vif = px.bar(vif_df, x="VIF", y="Fitur", orientation="h", color="VIF", 
                        color_continuous_scale="YlOrRd", title="Nilai VIF per Indikator")
        fig_vif.add_vline(x=10, line_dash="dash", line_color="red", 
                         annotation_text="Batas Multikolinieritas (VIF=10)")
        fig_vif.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_vif, use_container_width=True, key="tab3_vif")
        
        st.warning("⚠️ **Catatan:** Nilai VIF tinggi (>10) pada HAEMATOCRIT & HAEMOGLOBINS menunjukkan redundansi linear. Preprocessing menggunakan StandardScaler mengatasi issue ini.")

    # ==================== TAB 4: PEMILIHAN MODEL ====================
    with tab4:
        st.markdown("### 🎯 Analisis Pemilihan Model")
        st.markdown(
            "Tab ini menampilkan analisis perbandingan mendalam tentang kinerja semua model "
            "berdasarkan berbagai metrik untuk membantu Anda memilih model terbaik sesuai kebutuhan."
        )
        st.markdown("---")

        # Perbandingan lengkap semua model
        st.markdown("#### 📊 Tabel Perbandingan Lengkap")
        
        rows = []
        for name, r in results.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']*100:.2f}%",
                "Precision": f"{r['precision']*100:.2f}%",
                "Recall": f"{r['recall']*100:.2f}%",
                "F1-Score": f"{r['f1']*100:.2f}%",
                "AUC-ROC": f"{r['auc']:.4f}",
                "CV Mean": f"{r['cv_mean']*100:.2f}%",
                "CV Std": f"{r['cv_std']*100:.2f}%"
            })
        
        comparison_df = pd.DataFrame(rows)
        st.dataframe(comparison_df.set_index("Model"), use_container_width=True)

        st.markdown("---")

        # Visualisasi perbandingan metrik
        st.markdown("#### 📈 Visualisasi Perbandingan Metrik")
        st.plotly_chart(plot_metrics_bar(results), use_container_width=True, key="tab4_metrics_bar")

        st.markdown("---")

        # Ranking berdasarkan berbagai kriteria
        st.markdown("#### 🏆 Ranking Model Berdasarkan Berbagai Kriteria")
        
        accuracy_rank = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        f1_rank = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
        auc_rank = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)
        cv_rank = sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🥇 Top 3 Model berdasarkan Accuracy:**")
            for i, (name, r) in enumerate(accuracy_rank[:3], 1):
                st.write(f"{i}. **{name}** — {r['accuracy']*100:.2f}%")
            
            st.markdown("**🥇 Top 3 Model berdasarkan F1-Score:**")
            for i, (name, r) in enumerate(f1_rank[:3], 1):
                st.write(f"{i}. **{name}** — {r['f1']*100:.2f}%")
        
        with col2:
            st.markdown("**🥇 Top 3 Model berdasarkan AUC-ROC:**")
            for i, (name, r) in enumerate(auc_rank[:3], 1):
                st.write(f"{i}. **{name}** — {r['auc']:.4f}")
            
            st.markdown("**🥇 Top 3 Model berdasarkan CV Mean (Stabilitas):**")
            for i, (name, r) in enumerate(cv_rank[:3], 1):
                st.write(f"{i}. **{name}** — {r['cv_mean']*100:.2f}%")

        st.markdown("---")

        # Rekomendasi model
        st.markdown("#### 💡 Rekomendasi Model")
        
        best_accuracy = accuracy_rank[0]
        best_f1 = f1_rank[0]
        best_auc = auc_rank[0]
        best_cv = cv_rank[0]
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.success(f"**✅ Model Terbaik Keseluruhan:** {best_model_name}\n\n"
                      f"Akurasi: {results[best_model_name]['accuracy']*100:.2f}%\n\n"
                      f"AUC-ROC: {results[best_model_name]['auc']:.4f}\n\n"
                      f"F1-Score: {results[best_model_name]['f1']*100:.2f}%")
        
        with col_rec2:
            st.info(f"**ℹ️ Detail Rekomendasi:**\n\n"
                   f"• **Akurasi:** {best_accuracy[0]}\n"
                   f"• **F1-Score:** {best_f1[0]}\n"
                   f"• **AUC-ROC:** {best_auc[0]}\n"
                   f"• **Stabilitas:** {best_cv[0]}")

        st.markdown("---")

        # Panduan pemilihan model
        st.markdown("#### 📖 Panduan Pemilihan Model")
        
        with st.expander("🔍 Kapan menggunakan masing-masing model?", expanded=True):
            st.markdown("""
            **Logistic Regression:**
            - Cocok untuk interpretabilitas tinggi (mengerti pengaruh setiap fitur)
            - Model sederhana dan cepat
            - Optimal untuk dataset dengan hubungan linear dengan target
            
            **Decision Tree:**
            - Mudah diinterpretasi dan divisualisasikan
            - Cocok untuk data dengan hubungan non-linear
            - Risiko overfitting lebih tinggi tanpa pruning
            
            **Weighted KNN:**
            - Baik untuk dataset kecil hingga medium
            - Non-parametrik, bisa menangkap pola kompleks
            - Lambat saat prediksi pada dataset besar
            
            **Extra Trees:**
            - Ensemble yang robust dan stabil
            - Menggunakan banyak pohon acak untuk memperkuat generalisasi
            - Biasanya lebih cepat dan tetap memberikan feature importance yang kuat
            
            **XGBoost:**
            - Performa tinggi untuk kompetisi dan production
            - Gradient boosting yang powerful dan teroptimasi
            - Perlu tuning parameter lebih hati-hati
            """)

        st.markdown("---")

        # Metrik-metrik penting
        st.markdown("#### 📚 Penjelasan Metrik Evaluasi")
        
        with st.expander("📖 Definisi dan interpretasi metrik", expanded=False):
            st.markdown("""
            **Accuracy:**
            Persentase prediksi yang benar dari total prediksi. Baik untuk dataset balanced.
            
            **Precision:**
            Dari semua prediksi positif (Rawat Inap), berapa yang benar-benar positif.
            Penting ketika false positive costly.
            
            **Recall:**
            Dari semua kasus positif aktual, berapa yang berhasil dideteksi.
            Penting ketika false negative costly.
            
            **F1-Score:**
            Harmonic mean antara precision dan recall. Metrik balanced untuk imbalanced data.
            
            **AUC-ROC:**
            Area Under ROC Curve. Mengukur discrimination ability model pada berbagai threshold.
            Ideal untuk evaluasi probabilistic classifier.
            
            **CV Mean & CV Std:**
            Cross-validation mean dan standard deviation. Mengukur stabilitas model.
            CV Std rendah = model konsisten, CV Std tinggi = model tidak stabil.
            """)

        st.markdown("---")

        # Tabel ringkas untuk pengambilan keputusan
        st.markdown("#### 🎯 Matriks Keputusan")
        
        decision_matrix = []
        for name, r in results.items():
            decision_matrix.append({
                "Model": name,
                "Akurasi Tinggi": "✅" if r["accuracy"] >= 0.85 else "❌",
                "F1 Tinggi": "✅" if r["f1"] >= 0.85 else "❌",
                "AUC Tinggi": "✅" if r["auc"] >= 0.85 else "❌",
                "Stabil (CV)": "✅" if r["cv_std"] <= 0.05 else "❌",
                "Skor Total": sum([
                    r["accuracy"] >= 0.85,
                    r["f1"] >= 0.85,
                    r["auc"] >= 0.85,
                    r["cv_std"] <= 0.05
                ])
            })
        
        decision_df = pd.DataFrame(decision_matrix).sort_values("Skor Total", ascending=False)
        st.dataframe(decision_df.set_index("Model"), use_container_width=True)
