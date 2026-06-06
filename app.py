import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fertilizer Recommendation",
    page_icon="🌱",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/MutiaraCR/Dataset/refs/heads/main/Crop%20and%20fertilizer%20dataset.csv"

DISTRICT_OPTIONS     = ["Kolhapur", "Pune", "Sangli", "Satara", "Solapur"]
SOIL_OPTIONS         = ["Black", "Dark Brown", "Light Brown", "Medium Brown", "Red", "Red ", "Reddish Brown"]
CROP_OPTIONS         = ["Cotton", "Ginger", "Gram", "Grapes", "Groundnut", "Jowar",
                        "Maize", "Masoor", "Moong", "Rice", "Soybean", "Sugarcane",
                        "Tur", "Turmeric", "Urad", "Wheat"]
NUMERICAL_FEATURES   = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
CATEGORICAL_FEATURES = ["District_Name", "Soil_color", "Crop"]

COLOR_RF  = "#2E7D32"
COLOR_XGB = "#1565C0"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1B5E20; margin-bottom:0; }
    .sub-title   { font-size:1rem; color:#555; margin-top:0.2rem; margin-bottom:1.5rem; }
    .sec-header  { font-size:1.15rem; font-weight:600; color:#2E7D32;
                   border-bottom:2px solid #A5D6A7; padding-bottom:4px; margin-bottom:1rem; }
    .pred-card   { border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:0.5rem; }
    .pred-rf     { background:#E8F5E9; border-left:6px solid #2E7D32; }
    .pred-xgb    { background:#E3F2FD; border-left:6px solid #1565C0; }
    .pred-name   { font-size:1.5rem; font-weight:700; }
    .pred-model  { font-size:0.8rem; color:#555; margin-bottom:4px; }
    .stButton button { background-color:#2E7D32; color:white; border-radius:8px;
                       font-weight:600; height:3em; width:100%; }
    .stButton button:hover { background-color:#1B5E20; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    rf_model  = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    le        = joblib.load("label_encoder.pkl")
    return rf_model, xgb_model, le


# ─── Load Dataset ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.drop("Link", axis=1)
    return df


# ─── Load Metrics ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Menghitung metrik evaluasi...")
def compute_metrics(_rf_model, _xgb_model, _le):
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import f1_score, roc_auc_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    df = load_data()
    X  = df.drop("Fertilizer", axis=1)
    y  = df["Fertilizer"]

    # Encode ulang pakai le yang sudah di-load agar konsisten
    y_enc = _le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    rf_pred   = _rf_model.predict(X_test)
    xgb_pred  = _xgb_model.predict(X_test)
    rf_proba  = _rf_model.predict_proba(X_test)
    xgb_proba = _xgb_model.predict_proba(X_test)

    rf_f1   = float(f1_score(y_test, rf_pred,  average="macro"))
    xgb_f1  = float(f1_score(y_test, xgb_pred, average="macro"))
    rf_auc  = float(roc_auc_score(y_test, rf_proba,  multi_class="ovr", average="macro"))
    xgb_auc = float(roc_auc_score(y_test, xgb_proba, multi_class="ovr", average="macro"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv  = cross_val_score(_rf_model,  X, y_enc, cv=cv, scoring="f1_macro", n_jobs=-1)
    xgb_cv = cross_val_score(_xgb_model, X, y_enc, cv=cv, scoring="f1_macro", n_jobs=-1)

    classes    = _le.classes_
    rf_report  = classification_report(y_test, rf_pred,  target_names=classes, output_dict=True, zero_division=0)
    xgb_report = classification_report(y_test, xgb_pred, target_names=classes, output_dict=True, zero_division=0)

    return {
        "rf":  {"f1": rf_f1,  "auc": rf_auc,  "cv": rf_cv,  "report": rf_report},
        "xgb": {"f1": xgb_f1, "auc": xgb_auc, "cv": xgb_cv, "report": xgb_report},
    }


# ─── Init ─────────────────────────────────────────────────────────────────────
try:
    rf_model, xgb_model, le = load_models()
except FileNotFoundError:
    st.error("❌ File model tidak ditemukan. Pastikan `rf_model.pkl`, `xgb_model.pkl`, dan `label_encoder.pkl` ada di direktori yang sama dengan `app.py`.")
    st.stop()

df      = load_data()
metrics = compute_metrics(rf_model, xgb_model, le)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌱 Fertilizer Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem rekomendasi pupuk berbasis Machine Learning — Random Forest & XGBoost</p>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Pupuk", "📊 Perbandingan Model", "📈 Analisis Data"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="sec-header">Input Data Pertanian</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**📍 Lokasi & Tanah**")
        district  = st.selectbox("Distrik",        DISTRICT_OPTIONS)
        soil      = st.selectbox("Warna Tanah",    SOIL_OPTIONS)
        crop      = st.selectbox("Jenis Tanaman",  CROP_OPTIONS)

        st.markdown("**🌡️ Kondisi Lingkungan**")
        rainfall    = st.slider("Curah Hujan (mm)", 300,  1700, 800, step=50)
        temperature = st.slider("Suhu (°C)",         10,    40,  25)

    with col_r:
        st.markdown("**🧪 Kandungan Tanah**")
        nitrogen   = st.slider("Nitrogen (N)",  20, 150,  90)
        phosphorus = st.slider("Fosfor (P)",    10,  90,  50)
        potassium  = st.slider("Kalium (K)",     5, 150,  60)
        ph         = st.slider("pH Tanah",     5.5, 8.5, 6.5, step=0.1)

    st.markdown("---")
    predict_btn = st.button("🌿 Rekomendasikan Pupuk", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "District_Name": district,
            "Soil_color":    soil,
            "Nitrogen":      nitrogen,
            "Phosphorus":    phosphorus,
            "Potassium":     potassium,
            "pH":            ph,
            "Rainfall":      rainfall,
            "Temperature":   temperature,
            "Crop":          crop,
        }])

        rf_enc   = rf_model.predict(input_df)[0]
        xgb_enc  = xgb_model.predict(input_df)[0]
        rf_proba = rf_model.predict_proba(input_df)[0]
        xgb_proba= xgb_model.predict_proba(input_df)[0]

        rf_label  = le.inverse_transform([rf_enc])[0]
        xgb_label = le.inverse_transform([xgb_enc])[0]
        rf_conf   = rf_proba[rf_enc]  * 100
        xgb_conf  = xgb_proba[xgb_enc] * 100

        st.markdown('<p class="sec-header">Hasil Rekomendasi</p>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        with r1:
            st.markdown(f"""
            <div class="pred-card pred-rf">
                <div class="pred-model">🌲 Random Forest</div>
                <div class="pred-name">{rf_label}</div>
                <div>Confidence: <strong>{rf_conf:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="pred-card pred-xgb">
                <div class="pred-model">⚡ XGBoost</div>
                <div class="pred-name">{xgb_label}</div>
                <div>Confidence: <strong>{xgb_conf:.1f}%</strong></div>
            </div>""", unsafe_allow_html=True)

        if rf_label == xgb_label:
            st.success(f"✅ Kedua model sepakat: **{rf_label}** adalah pupuk yang direkomendasikan.")
        else:
            st.warning("⚠️ Kedua model memberikan rekomendasi berbeda. Gunakan model dengan confidence lebih tinggi sebagai acuan.")

        # Top-5 Probabilitas
        st.markdown('<p class="sec-header">Top 5 Probabilitas Prediksi</p>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)

        def plot_top5(proba, color, title):
            idx    = np.argsort(proba)[-5:][::-1]
            labels = le.inverse_transform(idx)
            vals   = proba[idx] * 100
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(range(5), vals[::-1], color=color, alpha=0.85)
            ax.set_yticks(range(5))
            ax.set_yticklabels(labels[::-1], fontsize=9)
            ax.set_xlabel("Probabilitas (%)", fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlim(0, 105)
            for bar, v in zip(bars, vals[::-1]):
                ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}%", va="center", fontsize=8)
            fig.tight_layout()
            return fig

        with p1:
            st.pyplot(plot_top5(rf_proba,  COLOR_RF,  "Random Forest"))
        with p2:
            st.pyplot(plot_top5(xgb_proba, COLOR_XGB, "XGBoost"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERBANDINGAN MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="sec-header">Ringkasan Performa Model</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RF — Macro F1",  f"{metrics['rf']['f1']:.4f}")
    c2.metric("RF — ROC AUC",   f"{metrics['rf']['auc']:.4f}")
    c3.metric("XGB — Macro F1", f"{metrics['xgb']['f1']:.4f}",
              delta=f"{metrics['xgb']['f1'] - metrics['rf']['f1']:+.4f} vs RF")
    c4.metric("XGB — ROC AUC",  f"{metrics['xgb']['auc']:.4f}",
              delta=f"{metrics['xgb']['auc'] - metrics['rf']['auc']:+.4f} vs RF")

    # ── Bar: F1 & AUC ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">F1-Score & ROC-AUC</p>', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    labels = ["Macro F1", "ROC AUC"]
    rf_vals  = [metrics["rf"]["f1"],  metrics["rf"]["auc"]]
    xgb_vals = [metrics["xgb"]["f1"], metrics["xgb"]["auc"]]
    x = np.arange(2)
    w = 0.3
    b1 = ax1.bar(x - w/2, rf_vals,  w, label="Random Forest", color=COLOR_RF,  alpha=0.88)
    b2 = ax1.bar(x + w/2, xgb_vals, w, label="XGBoost",       color=COLOR_XGB, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score")
    ax1.set_title("Perbandingan Metrik Evaluasi (Test Set)", fontweight="bold")
    ax1.legend()
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.4f}", ha="center", fontsize=9)
    fig1.tight_layout()
    st.pyplot(fig1)

    # ── Cross-Validation per Fold ──────────────────────────────────────────────
    st.markdown('<p class="sec-header">Cross-Validation F1-Macro (5-Fold)</p>', unsafe_allow_html=True)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, color, name in zip(
        axes2, ["rf", "xgb"], [COLOR_RF, COLOR_XGB], ["Random Forest", "XGBoost"]
    ):
        cv_scores = metrics[key]["cv"]
        folds = range(1, len(cv_scores) + 1)
        ax.bar(folds, cv_scores, color=color, alpha=0.82, width=0.5)
        ax.axhline(cv_scores.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {cv_scores.mean():.4f}")
        ax.fill_between(
            [0.5, len(cv_scores) + 0.5],
            cv_scores.mean() - cv_scores.std(),
            cv_scores.mean() + cv_scores.std(),
            alpha=0.12, color="red", label=f"±Std: {cv_scores.std():.4f}"
        )
        ax.set_xticks(list(folds))
        ax.set_xticklabels([f"Fold {i}" for i in folds], fontsize=9)
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel("F1-Macro")
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=8)

    fig2.suptitle("Cross-Validation per Fold", fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2)

    # ── F1 per Kelas ──────────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">F1-Score per Kelas Pupuk</p>', unsafe_allow_html=True)

    classes        = le.classes_
    rf_f1_class    = [metrics["rf"]["report"][c]["f1-score"]  for c in classes]
    xgb_f1_class   = [metrics["xgb"]["report"][c]["f1-score"] for c in classes]

    fig3, ax3 = plt.subplots(figsize=(11, 4))
    x3 = np.arange(len(classes))
    w3 = 0.35
    ax3.bar(x3 - w3/2, rf_f1_class,  w3, label="Random Forest", color=COLOR_RF,  alpha=0.85)
    ax3.bar(x3 + w3/2, xgb_f1_class, w3, label="XGBoost",       color=COLOR_XGB, alpha=0.85)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("F1-Score")
    ax3.set_ylim(0, 1.12)
    ax3.axhline(0.9, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax3.set_title("F1-Score per Kelas Pupuk", fontweight="bold")
    ax3.legend()
    fig3.tight_layout()
    st.pyplot(fig3)

    # ── Tabel Detail ──────────────────────────────────────────────────────────
    with st.expander("📋 Lihat tabel detail precision / recall / F1 per kelas"):
        rows = []
        for c in classes:
            rows.append({
                "Fertilizer":    c,
                "RF F1":         round(metrics["rf"]["report"][c]["f1-score"],   4),
                "RF Precision":  round(metrics["rf"]["report"][c]["precision"],  4),
                "RF Recall":     round(metrics["rf"]["report"][c]["recall"],     4),
                "XGB F1":        round(metrics["xgb"]["report"][c]["f1-score"],  4),
                "XGB Precision": round(metrics["xgb"]["report"][c]["precision"], 4),
                "XGB Recall":    round(metrics["xgb"]["report"][c]["recall"],    4),
            })
        tbl = pd.DataFrame(rows).set_index("Fertilizer")
        st.dataframe(
            tbl.style.background_gradient(cmap="Greens", subset=["RF F1", "XGB F1"]),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALISIS DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="sec-header">Distribusi Kelas Pupuk</p>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fert_counts = df["Fertilizer"].value_counts()
    bars4 = ax4.barh(fert_counts.index, fert_counts.values, color="#4CAF50", alpha=0.85)
    ax4.set_xlabel("Jumlah Data")
    ax4.set_title("Distribusi Kelas Pupuk", fontweight="bold")
    for bar, val in zip(bars4, fert_counts.values):
        ax4.text(val + 3, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=8)
    fig4.tight_layout()
    st.pyplot(fig4)

    st.markdown('<p class="sec-header">Distribusi Fitur Numerik</p>', unsafe_allow_html=True)
    fig5, axes5 = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes5.flatten(), NUMERICAL_FEATURES):
        sns.histplot(df[col], kde=True, ax=ax, color="#66BB6A", alpha=0.7)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
    fig5.suptitle("Distribusi Fitur Numerik", fontweight="bold")
    fig5.tight_layout()
    st.pyplot(fig5)

    st.markdown('<p class="sec-header">Matriks Korelasi</p>', unsafe_allow_html=True)
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    corr = df[NUMERICAL_FEATURES].corr()
    sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".2f",
                linewidths=0.5, ax=ax6, vmin=-1, vmax=1)
    ax6.set_title("Korelasi Fitur Numerik", fontweight="bold")
    fig6.tight_layout()
    st.pyplot(fig6)

    st.markdown('<p class="sec-header">Distribusi Jenis Tanaman</p>', unsafe_allow_html=True)
    fig7, ax7 = plt.subplots(figsize=(10, 4))
    crop_counts = df["Crop"].value_counts()
    ax7.bar(crop_counts.index, crop_counts.values, color="#81C784", alpha=0.85)
    ax7.set_ylabel("Jumlah Data")
    ax7.set_title("Distribusi Jenis Tanaman", fontweight="bold")
    ax7.set_xticklabels(crop_counts.index, rotation=45, ha="right")
    fig7.tight_layout()
    st.pyplot(fig7)

    with st.expander("🔎 Lihat sampel data (20 baris acak)"):
        st.dataframe(df.sample(20, random_state=42).reset_index(drop=True),
                     use_container_width=True)