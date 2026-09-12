"""
Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection
------------------------------------------------------------------------
Interactive Streamlit Research Prototype Dashboard (SIH 2026)

DISCLAIMER:
This application is strictly for algorithmic benchmarking, educational evaluation,
and research prototyping. It is NOT a medical diagnostic tool and is NOT certified
for clinical decision-making or patient evaluation.
"""

import os
import sys

# Set environment before any graphics/matplotlib imports
os.environ["MPLCONFIGDIR"] = "/tmp"

import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import __main__

def balanced_readout_mod4(x):
    return int(x) % 4

__main__.balanced_readout_mod4 = balanced_readout_mod4
# Import preprocessing utility
import preprocess


# -----------------------------------------------------------------------------
# 1. Top-Level Helpers for Unpickling & Feature Extraction
# -----------------------------------------------------------------------------
def parity_mod4(bitstring_int: int) -> int:
    """Named function matching VQC training for unpickling model weights."""
    return sum(int(b) for b in bin(bitstring_int)[2:]) % 4

# Ensure parity_mod4 is registered in __main__ namespace for pickle resolution
import __main__
__main__.parity_mod4 = parity_mod4


def safe_extract_prediction_index(pred_output) -> int:
    """
    Safely converts scalar or array-like prediction output into an integer index.
    Handles 0-D scalar arrays (e.g. Qiskit VQC on single sample), 1-D arrays,
    and Python native numeric types without indexing errors.
    """
    arr = np.asarray(pred_output)
    if arr.ndim == 0:
        return int(arr.item())
    return int(arr.ravel()[0])


@st.cache_resource
def load_all_artifacts():
    """Loads all models, transformers, and metric logs once into memory."""
    artifacts = {}

    # Load metadata
    meta_path = Path("processed_data/metadata.json")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            artifacts["metadata"] = json.load(f)
    else:
        artifacts["metadata"] = None

    # Load fitted transformers (scaler, PCA, quantum_scaler)
    trans_path = Path("processed_data/transformers.pkl")
    if trans_path.exists():
        with open(trans_path, "rb") as f:
            artifacts["transformers"] = pickle.load(f)
    else:
        artifacts["transformers"] = None

    # Load classical SVM model
    svm_path = Path("models/svm_baseline_model.pkl")
    if svm_path.exists():
        with open(svm_path, "rb") as f:
            artifacts["svm_model"] = pickle.load(f)
    else:
        artifacts["svm_model"] = None

    # Load quantum VQC model
    vqc_path = Path("models/vqc_model.pkl")
    if vqc_path.exists():
        with open(vqc_path, "rb") as f:
            artifacts["vqc_model"] = pickle.load(f)
    else:
        artifacts["vqc_model"] = None

    # Load metrics logs
    svm_m_path = Path("results/svm_metrics.json")
    if svm_m_path.exists():
        with open(svm_m_path, "r") as f:
            artifacts["svm_metrics"] = json.load(f)
    else:
        artifacts["svm_metrics"] = None

    vqc_m_path = Path("results/vqc_metrics.json")
    if vqc_m_path.exists():
        with open(vqc_m_path, "r") as f:
            artifacts["vqc_metrics"] = json.load(f)
    else:
        artifacts["vqc_metrics"] = None

    return artifacts


# -----------------------------------------------------------------------------
# 2. UI Styling & Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HQML - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SIH Theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .disclaimer-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 12px 18px;
        border-radius: 4px;
        color: #991b1b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .quantum-card {
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
        border: 1px solid #c4b5fd;
        border-radius: 8px;
        padding: 16px;
    }
    .classical-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
artifacts = load_all_artifacts()

with st.sidebar:
    st.markdown("### 🧠 HQML Platform")
    st.caption("SIH 2026 Research Practice Prototype")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "MRI Classification",
            "Model Comparison",
            "Quantum Circuit",
            "Explainability",
            "About the Prototype"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### System Status")
    if artifacts["svm_model"] and artifacts["vqc_model"]:
        st.success("✅ Models Loaded (SVM & VQC)")
    else:
        st.warning("⚠️ Some model artifacts missing")

    st.caption("Backend: Qiskit StatevectorSampler\nQubits: 4 | Features: 4D PCA")


# -----------------------------------------------------------------------------
# 4. Page: Overview
# -----------------------------------------------------------------------------
if nav_selection == "Overview":
    st.markdown('<div class="main-header">Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Research Prototype for Benchmarking 4-Qubit Variational Quantum Classifiers (VQC) against Classical Kernel Methods</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Non-Clinical Research Disclaimer:</strong> This platform is an experimental academic prototype designed for Smart India Hackathon (SIH 2026) exploration. It evaluates the mathematical mapping of compressed MRI features into quantum state spaces. <strong>It is NOT certified for medical diagnosis or clinical decision-making.</strong>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scans in Benchmark", "120 Scans", "Balanced 4-Class")
    with col2:
        st.metric("Raw Classical Descriptors", "59 Features", "Spatial + Gradient")
    with col3:
        st.metric("Quantum Feature Space", "4 Qubits", "16D Hilbert Space")
    with col4:
        st.metric("Simulation Backend", "Qiskit 2.5", "Noise-Free Local")

    st.markdown("### 🔄 End-to-End Pipeline Architecture")
    st.info("""
    1. **MRI Acquisition & Preprocessing:** 2D Brain MRI slices are standardized to $64 \\times 64$ grayscale resolution with normalized pixel intensities $[0.0, 1.0]$.
    2. **Classical Feature Extraction (59 Descriptors):** Computes global statistical moments, 16-bin intensity histograms, $4 \\times 4$ spatial patch variance, and finite-difference edge gradients.
    3. **Dimensionality Reduction (PCA):** Compresses the 59 descriptors into **4 Principal Components** (capturing $52.4\\%$ cumulative variance) to conform to the 4-qubit capacity of current NISQ quantum devices.
    4. **Dual Model Inference:**
       - **Classical SVM (RBF Kernel):** Infinite-dimensional reproducing kernel Hilbert space baseline.
       - **4-Qubit Quantum VQC:** Second-order Pauli expansion (`ZZFeatureMap`) paired with parameterized single-qubit rotations and entangling gates (`RealAmplitudes`).
    5. **Comparative Benchmarking & Explainability:** Full evaluation across Accuracy, Precision, Recall, Specificity, and F1-score alongside permutation feature importance.
    """)

    st.markdown("### 🎯 Key Project Objectives")
    st.markdown("""
    - **Algorithmic Parity:** Train and test classical and quantum architectures on the *exact same* data split and 4D feature representation.
    - **Resource-Aware Quantum Design:** Maintain a shallow 4-qubit circuit topology feasible on near-term NISQ simulators without physical quantum hardware.
    - **Scientific Transparency:** Provide rigorous model explainability without claiming unsubstantiated quantum advantage or clinical causation.
    """)


# -----------------------------------------------------------------------------
# 5. Page: MRI Classification
# -----------------------------------------------------------------------------
elif nav_selection == "MRI Classification":
    st.markdown('<div class="main-header">Brain MRI Scan Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a brain MRI scan or choose a benchmark sample for simultaneous Classical and Quantum inference.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Research Alert:</strong> Predictions generated below are experimental machine learning outputs and must <strong>never</strong> be interpreted as a medical diagnosis or clinical opinion.
    </div>
    """, unsafe_allow_html=True)

    col_input, col_display = st.columns([1.1, 1.3])

    target_image = None
    image_source_label = ""

    with col_input:
        st.markdown("#### Step 1: Input MRI Scan")
        input_mode = st.radio("Choose Input Method", ["Upload Image File", "Select Benchmark Demo Sample"], horizontal=True)

        if input_mode == "Upload Image File":
            uploaded_file = st.file_uploader("Upload Brain MRI (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_file is not None:
                try:
                    target_image = Image.open(uploaded_file)
                    image_source_label = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
        else:
            sample_dir = Path("data/BRISC")
            if sample_dir.exists():
                classes = [d.name for d in sample_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
                selected_cls = st.selectbox("Select Pathology Class for Demo", sorted(classes))
                cls_folder = sample_dir / selected_cls
                sample_files = list(cls_folder.glob("*.png")) + list(cls_folder.glob("*.jpg"))
                if sample_files:
                    selected_sample = st.selectbox("Select Sample Scan", [f.name for f in sample_files[:8]])
                    target_image = Image.open(cls_folder / selected_sample)
                    image_source_label = f"{selected_cls}/{selected_sample}"
            else:
                st.warning("Benchmark data directory not found.")

    with col_display:
        st.markdown("#### Step 2: Loaded MRI Preview")
        if target_image is not None:
            st.image(target_image, caption=f"Scan: {image_source_label}", width=250)
        else:
            st.info("Please upload an MRI image or select a benchmark sample to proceed.")

    if target_image is not None and artifacts["svm_model"] is not None and artifacts["vqc_model"] is not None:
        st.markdown("---")
        st.markdown("#### Step 3: Preprocessing & Feature Extraction")

        with st.spinner("Processing MRI slice through 59-descriptor pipeline and 4D PCA..."):
            try:
                # 1. Resize to 64x64 grayscale and normalize [0, 1]
                resized_img = target_image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
                norm_arr = np.asarray(resized_img, dtype=np.float32) / 255.0
                batch_arr = np.expand_dims(norm_arr, axis=0)  # Shape: (1, 64, 64)

                # 2. Extract 59 classical features
                raw_feats = preprocess.extract_classical_features(batch_arr)

                # 3. Transform via fitted Scaler and PCA
                scaler = artifacts["transformers"]["feature_scaler"]
                pca = artifacts["transformers"]["pca"]
                quantum_scaler = artifacts["transformers"]["quantum_scaler"]

                scaled_feats = scaler.transform(raw_feats)
                pca_feats = pca.transform(scaled_feats)
                quantum_feats = quantum_scaler.transform(pca_feats)

                # Display 4 PCA Coordinates
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("PC1 (Luminance Variance)", f"{pca_feats[0, 0]:.3f}", f"θ0 = {quantum_feats[0, 0]:.2f} rad")
                p2.metric("PC2 (Histogram Spread)", f"{pca_feats[0, 1]:.3f}", f"θ1 = {quantum_feats[0, 1]:.2f} rad")
                p3.metric("PC3 (Edge Gradients)", f"{pca_feats[0, 2]:.3f}", f"θ2 = {quantum_feats[0, 2]:.2f} rad")
                p4.metric("PC4 (Spatial Moments)", f"{pca_feats[0, 3]:.3f}", f"θ3 = {quantum_feats[0, 3]:.2f} rad")

            except Exception as ex:
                st.error(f"Incompatible image format or preprocessing failure: {ex}")
                st.stop()

        st.markdown("#### Step 4: Dual Model Inference")
        c_svm, c_vqc = st.columns(2)

        # Mapping dictionary
        meta = artifacts["metadata"]
        class_map = {int(k): v for k, v in meta["classes"].items()} if meta else {0: "glioma", 1: "meningioma", 2: "no_tumor", 3: "pituitary"}

        # Classical Inference
        with c_svm:
            st.markdown('<div class="classical-card">', unsafe_allow_html=True)
            st.markdown("### 🖥️ Classical SVM (RBF)")
            t0 = time.perf_counter()
            svm_pred_raw = artifacts["svm_model"].predict(pca_feats)
            svm_time = (time.perf_counter() - t0) * 1000
            svm_pred_idx = safe_extract_prediction_index(svm_pred_raw)
            svm_label = class_map.get(svm_pred_idx, f"Class {svm_pred_idx}")

            st.markdown(f"**Predicted Category:** `{svm_label.upper()}`")
            st.caption(f"Inference Latency: {svm_time:.4f} ms")
            st.markdown("</div>", unsafe_allow_html=True)

        # Quantum Inference
        with c_vqc:
            st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
            st.markdown("### ⚛️ 4-Qubit Quantum VQC")
            t0 = time.perf_counter()
            vqc_pred_raw = artifacts["vqc_model"].predict(quantum_feats)
            vqc_time = (time.perf_counter() - t0) * 1000
            vqc_pred_idx = safe_extract_prediction_index(vqc_pred_raw)
            vqc_label = class_map.get(vqc_pred_idx, f"Class {vqc_pred_idx}")

            st.markdown(f"**Predicted Category:** `{vqc_label.upper()}`")
            st.caption(f"Inference Latency: {vqc_time:.2f} ms")
            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Page: Model Comparison
# -----------------------------------------------------------------------------
elif nav_selection == "Model Comparison":
    st.markdown('<div class="main-header">Model Performance Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Direct side-by-side comparison of Classical SVM vs. 4-Qubit Quantum VQC on identical held-out test data.</div>', unsafe_allow_html=True)

    svm_m = artifacts["svm_metrics"]
    vqc_m = artifacts["vqc_metrics"]

    if svm_m and vqc_m:
        svm_ov = svm_m["overall_metrics"]
        vqc_ov = vqc_m["overall_metrics"]

        # Comparison Table
        comparison_data = {
            "Metric": ["Accuracy", "Macro Precision", "Macro Sensitivity / Recall", "Macro Specificity", "Macro F1-Score", "Training Duration", "Per-Sample Latency"],
            "Classical SVM (RBF Kernel)": [
                f"{svm_ov['accuracy']*100:.2f}%",
                f"{svm_ov['macro_precision']*100:.2f}%",
                f"{svm_ov['macro_recall_sensitivity']*100:.2f}%",
                f"{svm_ov['macro_specificity']*100:.2f}%",
                f"{svm_ov['macro_f1_score']*100:.2f}%",
                f"{svm_m['training_time_ms']:.2f} ms",
                f"{svm_m['per_sample_inference_latency_ms']:.4f} ms"
            ],
            "Quantum VQC (4 Qubits)": [
                f"{vqc_ov['accuracy']*100:.2f}%",
                f"{vqc_ov['macro_precision']*100:.2f}%",
                f"{vqc_ov['macro_recall_sensitivity']*100:.2f}%",
                f"{vqc_ov['macro_specificity']*100:.2f}%",
                f"{vqc_ov['macro_f1_score']*100:.2f}%",
                f"{vqc_m['training_time_seconds']:.2f} s ({vqc_m['training_time_seconds']*1000:.0f} ms)",
                f"{vqc_m['per_sample_inference_latency_ms']:.2f} ms"
            ]
        }
        df_comp = pd.DataFrame(comparison_data)
        st.table(df_comp)

        # Bar Chart of Metrics
        st.markdown("### 📊 Metric Visualizations")
        chart_metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro Specificity", "Macro F1"]
        chart_df = pd.DataFrame({
            "Metric": chart_metrics,
            "Classical SVM": [
                svm_ov["accuracy"] * 100,
                svm_ov["macro_precision"] * 100,
                svm_ov["macro_recall_sensitivity"] * 100,
                svm_ov["macro_specificity"] * 100,
                svm_ov["macro_f1_score"] * 100
            ],
            "Quantum VQC": [
                vqc_ov["accuracy"] * 100,
                vqc_ov["macro_precision"] * 100,
                vqc_ov["macro_recall_sensitivity"] * 100,
                vqc_ov["macro_specificity"] * 100,
                vqc_ov["macro_f1_score"] * 100
            ]
        }).set_index("Metric")

        st.bar_chart(chart_df)

        # Side-by-Side Confusion Matrices
        st.markdown("### 🧩 Confusion Matrices (Held-Out Test Partition)")
        cm_col1, cm_col2 = st.columns(2)

        svm_cm_img = Path("results/confusion_matrix_svm.png")
        vqc_cm_img = Path("results/confusion_matrix_vqc.png")

        with cm_col1:
            if svm_cm_img.exists():
                st.image(str(svm_cm_img), caption="Classical SVM (RBF Kernel) Confusion Matrix")
        with cm_col2:
            if vqc_cm_img.exists():
                st.image(str(vqc_cm_img), caption="Quantum VQC (4 Qubits) Confusion Matrix")
    else:
        st.warning("Metrics artifacts not found in results/ directory.")


# -----------------------------------------------------------------------------
# 7. Page: Quantum Circuit
# -----------------------------------------------------------------------------
elif nav_selection == "Quantum Circuit":
    st.markdown('<div class="main-header">Quantum Circuit Architecture & Encoding</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Structural decomposition of the 4-Qubit Variational Quantum Classifier (VQC).</div>', unsafe_allow_html=True)

    st.markdown("### 1. Quantum Feature Encoding Pipeline")
    mapping_img = Path("outputs/explainability/quantum_feature_mapping.png")
    if mapping_img.exists():
        st.image(str(mapping_img), caption="Classical 4D PCA Coordinates to 4-Qubit Hilbert State Encoding", use_container_width=True)

    st.markdown("### 2. Full VQC Circuit Diagram")
    circuit_img = Path("outputs/explainability/quantum_circuit_diagram.png")
    if circuit_img.exists():
        st.image(str(circuit_img), caption="4-Qubit Quantum Circuit (ZZFeatureMap + RealAmplitudes)", use_container_width=True)

    circuit_txt_path = Path("outputs/explainability/quantum_circuit_diagram.txt")
    if circuit_txt_path.exists():
        with st.expander("View Full Text/ASCII Circuit Matrix"):
            with open(circuit_txt_path, "r") as f:
                st.code(f.read(), language="text")

    st.markdown("### 3. Plain-English Quantum Stage Breakdown")
    st.markdown("""
    - **Step 1: Superposition via Hadamard Gates ($H$):** Initializes all 4 qubits into equal superpositions of $|0\\rangle$ and $|1\\rangle$.
    - **Step 2: Single-Qubit Phase Encoding ($P(2x_i)$):** Rotates the relative phase of each qubit based directly on its corresponding PCA feature coordinate.
    - **Step 3: Non-Linear Entanglement ($R_{ZZ}$):** Uses CNOT gates to couple adjacent qubits with phase interactions $2(\\pi - x_i)(\\pi - x_j)$, capturing cross-feature correlations in a 16-dimensional quantum state.
    - **Step 4: Trainable Variational Layers (`RealAmplitudes`):** 12 tunable single-qubit $R_y(\\theta)$ rotations optimized by COBYLA to steer measurement probabilities toward the target class.
    - **Step 5: Parity Readout:** Measurement collapse produces a 4-bit integer whose bit-sum modulo 4 determines the final classification index.
    """)


# -----------------------------------------------------------------------------
# 8. Page: Explainability
# -----------------------------------------------------------------------------
elif nav_selection == "Explainability":
    st.markdown('<div class="main-header">Model Explainability & Interpretability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Post-hoc feature importance and decomposition of underlying MRI representations.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Scientific Boundary Note:</strong> Feature importance analyses identify which image properties (luminosity, edge density, regional patches) influenced the model's mathematical output. They <strong>do NOT</strong> prove biological causality or identify histopathological cellular markers.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Top Contributing Descriptors", "PCA Loadings Decomposition", "Explainability Report"])

    with tab1:
        st.markdown("#### Permutation Feature Importance Across 59 Descriptors")
        feat_img = Path("outputs/explainability/classical_feature_importance.png")
        if feat_img.exists():
            st.image(str(feat_img), caption="Top 15 Image Descriptors Ranked by Accuracy Impact Upon Permutation", use_container_width=True)

    with tab2:
        st.markdown("#### How Original Descriptors Form the 4 Qubit Inputs (PCA Loadings)")
        loadings_img = Path("outputs/explainability/pca_loadings_heatmap.png")
        if loadings_img.exists():
            st.image(str(loadings_img), caption="PCA Weight Loadings: Mapping 59 Descriptors into PC1..PC4", use_container_width=True)

    with tab3:
        report_path = Path("outputs/explainability/explainability_report.md")
        if report_path.exists():
            with open(report_path, "r") as f:
                st.markdown(f.read())


# -----------------------------------------------------------------------------
# 9. Page: About the Prototype
# -----------------------------------------------------------------------------
elif nav_selection == "About the Prototype":
    st.markdown('<div class="main-header">About This Research Prototype</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Smart India Hackathon (SIH) 2026 Practice Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🔬 Project Context & Vision
    This software is a prototype implementation exploring the intersection of **Quantum Machine Learning (QML)** and **Biomedical Image Analysis**.

    As quantum computing technologies mature, understanding how variational algorithms interact with compressed medical imaging representations is a foundational research prerequisite.

    ### ⚙️ Technical Highlights
    - **No Physical Quantum Hardware Required:** Executes on Qiskit's high-performance `StatevectorSampler` local simulator.
    - **Direct Classical Benchmark:** Rigorously compared against a classical SVM utilizing an RBF kernel in reproducing kernel Hilbert space.
    - **Modular Clean Codebase:** Preprocessing, classical modeling, quantum circuits, and explainability modules are decoupled for independent testing.

    ### 🛡️ Ethical & Regulatory Notice
    - **Research Status:** Strictly intended for algorithmic exploration and academic benchmarking.
    - **Non-Diagnostic:** Not approved, tested, or certified by any health authority (e.g., CDSCO, FDA, CE).
    - **No Quantum Advantage Claim:** Performance metrics accurately reflect current NISQ simulator boundaries without inflated claims.
    """)

    st.markdown("---")
    st.caption("Built with Streamlit, Qiskit Machine Learning, and Scikit-Learn | SIH 2026 Practice")
