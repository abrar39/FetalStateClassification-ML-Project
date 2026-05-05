import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
import os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Health Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'DM Serif Display', serif;
    color: #58a6ff;
    font-size: 1.3rem;
    letter-spacing: -0.02em;
}

/* Header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #ff7b72 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff44; }

.metric-label {
    color: #8b949e;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #e6edf3;
}

/* Result boxes */
.result-normal {
    background: linear-gradient(135deg, #0d2818, #0d3321);
    border: 1px solid #2ea043;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-suspect {
    background: linear-gradient(135deg, #2d2200, #3d2f00);
    border: 1px solid #d29922;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-pathological {
    background: linear-gradient(135deg, #2d0f0f, #3d1414);
    border: 1px solid #f85149;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.result-desc {
    color: #8b949e;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #e6edf3;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Sliders & inputs override */
.stSlider [data-testid="stTickBar"] { display: none; }

/* Probability bar */
.prob-bar-container {
    margin: 0.5rem 0;
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #8b949e;
    margin-bottom: 3px;
}
.prob-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Try to load a saved model; fall back to a demo RandomForest."""
    model_dir = Path("models")
    # Check common model pickle file exists in the "models" subdirectory
    model = model_dir / "final_lr_pipeline.pkl"
    print(f"Looking for model and scaler files in {model_dir}...")
    if model:
        print(f"Loading model from  {model}...")
        with open(model, "rb") as f:
            model = pickle.load(f)
        return model, True
    print("No model file found. Using demo model.")
    return None, False
    
    # ── Demo model (trains on the bundled UCI dataset via sklearn) ──
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        st.warning(
            "⚠️ No saved model found (`model.pkl`). Running a **demo model** trained on "
            "synthetic data. Replace with your trained model for real predictions.",
            icon="⚠️",
        )
        X, y = make_classification(
            n_samples=2126, n_features=21, n_classes=3,
            n_informative=10, random_state=42,
            weights=[0.778, 0.138, 0.084],
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return clf, False
    except Exception as e:
        st.error(f"Could not create demo model: {e}")
        return None, False


# ── Feature definitions ────────────────────────────────────────────────────────
FEATURES = {
"Accelerations": {
    "key": "AC", "min": 0.0, "max": 50.0, "default": 0.0, "step": 1.0,
    "help": "Number of accelerations."
},
"Fetal Movement": {
    "key": "FM", "min": 0.0, "max": 1000.0, "default": 0.0, "step": 10.0,
    "help": "Number of fetal movements."
},
"Uterine Contractions": {
    "key": "UC", "min": 0.0, "max": 50.0, "default": 0.0, "step": 1.0,
    "help": "Number of uterine contractions per second."
},
"Light Decelerations": {
    "key": "DL", "min": 0.0, "max": 50.0, "default": 0.0, "step": 1.0,
    "help": "Number of light decelerations per second."
},
"Severe Decelerations": {
    "key": "DS", "min": 0.0, "max": 1.0, "default": 0.0, "step": 1.0,
    "help": "Number of severe decelerations."
},
"Prolonged Decelerations": {
    "key": "DP", "min": 0.0, "max": 10.0, "default": 0.0, "step": 1.0,
    "help": "Number of prolonged decelerations."
},
"Baseline Fetal Heart Rate (bpm)": {
    "key": "LB", "min": 100.0, "max": 160.0, "default": 133.0, "step": 1.0,
    "help": "Baseline fetal heart rate in beats per minute."
},
"% Time with Abnormal Short Term Variability": {
    "key": "ASTV", "min": 0.0, "max": 100.0, "default": 47.0, "step": 1.0,
    "help": "Percentage of time with abnormal short term variability."
},
"Mean Short Term Variability": {
    "key": "MSTV", "min": 0.0, "max": 1.0, "default": 1.3, "step": 0.1,
    "help": "Mean value of short term variability."
},
"% Time with Abnormal Long Term Variability": {
    "key": "ALTV", "min": 0.0, "max": 100.0, "default": 10.0, "step": 1.0,
    "help": "Percentage of time with abnormal long term variability."
},
"Mean Long Term Variability": {
    "key": "MLTV", "min": 0.0, "max": 100.0, "default": 8.0, "step": 0.1,
    "help": "Mean value of long term variability."
},
"Histogram Width": {
    "key": "Width", "min": 0.0, "max": 200.0, "default": 70.0, "step": 1.0,
    "help": "Width of FHR histogram."
},
"Histogram Max": {
    "key": "Nmax", "min": 0.0, "max": 250.0, "default": 150.0, "step": 1.0,
    "help": "Maximum frequency of the histogram."
},
"Histogram Zeros": {
    "key": "Nzeros", "min": 0.0, "max": 10.0, "default": 0.0, "step": 1.0,
    "help": "Number of zeros in the histogram."
},
"Histogram Mode": {
    "key": "Mode", "min": 0.0, "max": 200.0, "default": 137.0, "step": 1.0,
    "help": "Histogram mode."
},
"Histogram Mean": {
    "key": "Mean", "min": 0.0, "max": 200.0, "default": 134.0, "step": 1.0,
    "help": "Histogram mean."
},
"Histogram Median": {
    "key": "Median", "min": 0.0, "max": 200.0, "default": 138.0, "step": 1.0,
    "help": "Histogram median."
},
"Histogram Variance": {
    "key": "Variance", "min": 0.0, "max": 300.0, "default": 18.0, "step": 1.0,
    "help": "Histogram variance."
},
"Histogram Tendency": {
    "key": "Tendency", "min": -1.0, "max": 1.0, "default": 0.0, "step": 1.0,
    "help": "Histogram tendency (-1=left; 0=center; 1=right)."
},
"Class A": {
    "key": "A", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Calm sleep (Binary)."
},
"Class B": {
    "key": "B", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "REM sleep (Binary)."
},
"Class C": {
    "key": "C", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Calm vigilance (Binary).)"
},
"Class D": {
    "key": "D", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Active vigilance (Binary)."
},
"Class E": {
    "key": "E", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Shifting pattern (Binary)."
},
"Class AD": {
    "key": "AD", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Accelerative/decelerative pattern (Binary)."
},
"Class DE": {
    "key": "DE", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Decelerative pattern (Binary)."
},
"Class LD": {
    "key": "LD", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Largely decelerative pattern (Binary)."
},
"Class FS": {
    "key": "FS", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Flat-sinusoidal pattern (Binary)."
},
"Class SUSP": {
    "key": "SUSP", "min": -1.0, "max": 1.0, "default": -1.0, "step": 2.0,
    "help": "Suspect pattern (Binary)."
},
"Fetal Health Class": {
    "key": "CLASS", "min": 1.0, "max": 10.0, "default": 1.0, "step": 1.0,
    "help": "Fetal health class 1 to 10."
}
}

CLASS_INFO = {
    1: {
        "label": "Normal",
        "emoji": "✅",
        "color": "#2ea043",
        "css": "result-normal",
        "desc": "The fetal health indicators appear within normal range. Routine monitoring is recommended.",
    },
    2: {
        "label": "Suspect",
        "emoji": "⚠️",
        "color": "#d29922",
        "css": "result-suspect",
        "desc": "Some indicators are outside normal range. Further clinical assessment is advised.",
    },
    3: {
        "label": "Pathological",
        "emoji": "🚨",
        "color": "#f85149",
        "css": "result-pathological",
        "desc": "Critical indicators detected. Immediate medical evaluation is strongly recommended.",
    },
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🫀 Input Parameters")
        st.markdown("---")

        values = {}
        tabs = st.tabs(["📈 CTG", "📊 Variability", "📉 Histogram"])

        ctg_features = list(FEATURES.items())[:7]
        var_features  = list(FEATURES.items())[7:11]
        hist_features = list(FEATURES.items())[11:]

        with tabs[0]:
            for label, cfg in ctg_features:
                values[cfg["key"]] = st.slider(
                    label, cfg["min"], cfg["max"], cfg["default"],
                    step=cfg["step"], help=cfg["help"],
                )

        with tabs[1]:
            for label, cfg in var_features:
                values[cfg["key"]] = st.slider(
                    label, cfg["min"], cfg["max"], cfg["default"],
                    step=cfg["step"], help=cfg["help"],
                )

        with tabs[2]:
            for label, cfg in hist_features:
                values[cfg["key"]] = st.slider(
                    label, cfg["min"], cfg["max"], cfg["default"],
                    step=cfg["step"], help=cfg["help"],
                )

        st.markdown("---")
        predict_btn = st.button("🔍 Run Classification", use_container_width=True)
        return values, predict_btn


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model, is_model_loaded = load_model()

    # Hero header
    st.markdown('<div class="hero-title">Fetal Health Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Cardiotocography-based fetal state assessment · '
        'Normal · Suspect · Pathological</div>',
        unsafe_allow_html=True,
    )

    values, predict_btn = render_sidebar()

    # ── Top KPI row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline FHR</div>
            <div class="metric-value">{values['LB']} <span style="font-size:1rem;color:#8b949e">bpm</span></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accelerations</div>
            <div class="metric-value">{values['AC']:.3f} <span style="font-size:1rem;color:#8b949e">/s</span></div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Uterine Contractions</div>
            <div class="metric-value">{values['UC']:.3f} <span style="font-size:1rem;color:#8b949e">/s</span></div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ST Variability</div>
            <div class="metric-value">{values['ASTV']} <span style="font-size:1rem;color:#8b949e">%</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Predict ────────────────────────────────────────────────────────────────
    if predict_btn and model is not None:
        # apply scaling and dummy encoding to relevant features
        #features_to_scale = ["AC", "FM", "UC", "DL", "DS", "DP", "LB", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Nmax", "Nzeros", "Mode", "Mean", "Median", "Variance"]
        features_to_encode = ["Tendency", "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP", "CLASS"]
        
        feature_order = [cfg["key"] for cfg in FEATURES.values()]
        input_array = np.array([[values[k] for k in feature_order]])

        # Create a dataframe of the input features
        input_df = pd.DataFrame(input_array, columns=feature_order)
        #input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])
        #input_df = pd.get_dummies(input_df, columns=features_to_encode, drop_first=True, dtype=np.uint8)
        input_df[features_to_encode] = input_df[features_to_encode].astype(float) # Ensure binary features are integers
        pred = model.predict(input_df)[0]
        info = CLASS_INFO[pred]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]

        # Result + probabilities side by side
        res_col, prob_col = st.columns([1, 1])

        with res_col:
            st.markdown(f"""
            <div class="{info['css']}">
                <div style="font-size:3rem;margin-bottom:0.5rem">{info['emoji']}</div>
                <div class="result-title" style="color:{info['color']}">{info['label']}</div>
                <div class="result-desc">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

        with prob_col:
            st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
            if proba is not None:
                colors = ["#2ea043", "#d29922", "#f85149"]
                labels = ["Normal", "Suspect", "Pathological"]
                for idx, (lbl, prob, clr) in enumerate(zip(labels, proba, colors)):
                    pct = int(prob * 100)
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <div class="prob-label"><span>{lbl}</span><span style="color:{clr};font-weight:600">{pct}%</span></div>
                        <div class="prob-bar-bg">
                            <div style="width:{pct}%;height:100%;background:{clr};border-radius:4px;transition:width 0.5s"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Model does not support probability output.")

        # Input summary table
        st.markdown("---")
        st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame(
            [(lbl, values[cfg["key"]]) for lbl, cfg in FEATURES.items()],
            columns=["Feature", "Value"],
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    elif predict_btn and model is None:
        st.error("Model could not be loaded. Please check your model file.")

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            background:#161b22;
            border:1px dashed #30363d;
            border-radius:16px;
            padding:3rem;
            text-align:center;
            color:#8b949e;
        ">
            <div style="font-size:3rem;margin-bottom:1rem">🫀</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#e6edf3;margin-bottom:0.5rem">
                Adjust parameters and run classification
            </div>
            <div style="font-size:0.9rem;line-height:1.7">
                Use the sidebar sliders to set CTG features,<br>
                then click <strong style="color:#58a6ff">Run Classification</strong> to get a prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#484f58;font-size:0.75rem;padding:1rem 0">
        ⚕️ For clinical decision support only · Not a substitute for professional medical advice<br>
        Dataset: UCI Fetal Health Classification · Cardiotocography (CTG) signals
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
