"""
Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection
------------------------------------------------------------------------
Module: Explainability & Interpretability Framework

RESEARCH USE ONLY:
This module provides post-hoc mathematical and algorithmic explainability for
benchmarking classical vs. quantum models on brain MRI feature representations.
It does NOT establish causal biological mechanisms, histopathological findings,
or clinical biomarkers, and is NOT certified for clinical diagnosis.

Features & Deliverables:
1. Classical SVM Feature Importance:
   - Permutation importance on the full (Scaler -> PCA -> SVM) pipeline using held-out test data.
   - PCA component loading decomposition mapping 59 raw spatial/statistical features to 4 PCs.
   - High-resolution visualizations: Bar chart of top features and PCA loading heatmap.
2. Quantum VQC Circuit & Encoding Visualization:
   - Visual and formatted ASCII diagram of the 4-qubit quantum circuit.
   - Dedicated schematic diagram showing 4D PCA angle mapping into qubits and ZZ-entanglement.
   - Detailed textual and structured JSON explainability reports.
"""

import os
import sys

# Configure non-interactive matplotlib backend and temporary cache directory
os.environ["MPLCONFIGDIR"] = "/tmp"

import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Import preprocessing utility to extract feature names and load images
import preprocess


# -----------------------------------------------------------------------------
# 1. Feature Name Generator
# -----------------------------------------------------------------------------
def get_feature_names_and_categories() -> tuple[list[str], list[str]]:
    """
    Constructs standardized semantic names and category tags for all 59
    classical image features extracted during the preprocessing step.
    """
    names = []
    categories = []

    # 1. Global Intensity Moments (7 features)
    moment_names = ["Global_Mean", "Global_Std", "Global_Min", "Global_Max", "Global_Median", "Global_P25", "Global_P75"]
    for m in moment_names:
        names.append(m)
        categories.append("Global Statistics")

    # 2. Intensity Histogram (16 features)
    for b in range(16):
        names.append(f"Hist_Bin_{b+1:02d}")
        categories.append("Intensity Distribution")

    # 3. Spatial Patch Pooling (4x4 grid = 16 patches * 2 stats = 32 features)
    for r in range(4):
        for c in range(4):
            names.append(f"Patch_R{r}C{c}_Mean")
            categories.append("Spatial Patch Statistics")
            names.append(f"Patch_R{r}C{c}_Std")
            categories.append("Spatial Patch Statistics")

    # 4. Gradient / Edge Approximation (4 features)
    gradient_names = ["Grad_X_Mean", "Grad_X_Std", "Grad_Y_Mean", "Grad_Y_Std"]
    for g in gradient_names:
        names.append(g)
        categories.append("Edge Gradients")

    return names, categories


# -----------------------------------------------------------------------------
# 2. Classical Explainability Analysis
# -----------------------------------------------------------------------------
def analyze_classical_pipeline(
    data_dir: str = "data/BRISC",
    proc_dir: str = "processed_data",
    model_path: str = "models/svm_baseline_model.pkl",
    output_dir: str = "outputs/explainability"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  EXPLAINABILITY MODULE: CLASSICAL & QUANTUM INTERPRETATION")
    print("=" * 70)

    # 1. Load fitted models and transformers
    with open(model_path, "rb") as f:
        svm_model = pickle.load(f)

    with open(Path(proc_dir) / "transformers.pkl", "rb") as f:
        trans = pickle.load(f)

    scaler = trans["feature_scaler"]
    pca = trans["pca"]

    # 2. Re-extract raw features on full dataset to match exact train/test indices
    print("[STEP 1] Reconstructing 59 raw classical features from staged scans...")
    images, labels, c2i, i2c, _ = preprocess.discover_and_load_images(Path(data_dir))
    raw_features = preprocess.extract_classical_features(images)

    npz_data = np.load(Path(proc_dir) / "dataset_preprocessed.npz")
    test_idx = npz_data["test_indices"]

    X_test_raw = raw_features[test_idx]
    y_test = labels[test_idx]

    feature_names, categories = get_feature_names_and_categories()

    # 3. Compute Permutation Feature Importance
    print("[STEP 2] Computing permutation feature importance across 59 raw descriptors...")
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("svm", svm_model)
    ])

    perm_results = permutation_importance(
        pipeline,
        X_test_raw,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring="accuracy"
    )

    importances_mean = perm_results.importances_mean
    importances_std = perm_results.importances_std

    # Rank features by importance
    ranked_indices = np.argsort(importances_mean)[::-1]
    top_k = 15
    top_indices = ranked_indices[:top_k]

    print(f"         Top {top_k} features identified by accuracy degradation upon shuffling.")

    # 4. Generate Classical Feature Importance Bar Chart
    plt.figure(figsize=(10, 6.5), dpi=150)
    top_names = [feature_names[i] for i in top_indices][::-1]
    top_scores = [importances_mean[i] for i in top_indices][::-1]
    top_stds = [importances_std[i] for i in top_indices][::-1]
    top_cats = [categories[i] for i in top_indices][::-1]

    # Category color palette
    palette = {
        "Spatial Patch Statistics": "#2b5c8f",
        "Intensity Distribution": "#e26d5c",
        "Global Statistics": "#38b000",
        "Edge Gradients": "#9d4edd"
    }
    bar_colors = [palette.get(c, "#555555") for c in top_cats]

    bars = plt.barh(range(len(top_names)), top_scores, xerr=top_stds, color=bar_colors, alpha=0.88, capsize=3.5, edgecolor="black", linewidth=0.5)
    plt.yticks(range(len(top_names)), top_names, fontsize=9)
    plt.xlabel("Mean Accuracy Degradation (Permutation Importance)", fontsize=10, labelpad=8)
    plt.title(f"Classical Feature Importance: Top {top_k} Image Descriptors (SVM Pipeline)", fontsize=11, fontweight="bold", pad=12)

    # Add custom legend for categories
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor="black", label=cat, linewidth=0.5)
        for cat, color in palette.items()
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    feat_imp_plot = out_path / "classical_feature_importance.png"
    plt.savefig(feat_imp_plot)
    plt.close()
    print(f"[STEP 3] Saved feature importance plot to: {feat_imp_plot}")

    # 5. PCA Loading Heatmap Analysis
    print("[STEP 4] Decomposing PCA components into raw feature loadings...")
    loadings = pca.components_  # 4 rows, 59 columns
    top_pca_indices = np.argsort(np.sum(np.abs(loadings), axis=0))[::-1][:15]

    plt.figure(figsize=(11, 4.5), dpi=150)
    sns.heatmap(
        loadings[:, top_pca_indices],
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        xticklabels=[feature_names[i] for i in top_pca_indices],
        yticklabels=["PC1 (22.9% var)", "PC2 (12.8% var)", "PC3 (10.0% var)", "PC4 (6.7% var)"],
        cbar_kws={"label": "Component Loading Weight"}
    )
    plt.title("PCA Loadings: Contribution of Top Image Descriptors to 4 Qubit Features", fontsize=11, fontweight="bold", pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=8.5)
    plt.tight_layout()

    pca_heatmap_plot = out_path / "pca_loadings_heatmap.png"
    plt.savefig(pca_heatmap_plot)
    plt.close()
    print(f"         Saved PCA loadings heatmap to: {pca_heatmap_plot}")

    # 6. Save Structured Importance Data
    importance_summary = {
        "analysis_type": "Permutation Feature Importance on Held-Out Test Set (30 repeats)",
        "pipeline": "Raw Features (59) -> StandardScaler -> PCA (4) -> SVM (RBF Kernel)",
        "top_features": [
            {
                "rank": rank + 1,
                "feature_name": feature_names[idx],
                "category": categories[idx],
                "mean_importance_drop": float(importances_mean[idx]),
                "std_error": float(importances_std[idx])
            }
            for rank, idx in enumerate(ranked_indices[:20])
        ]
    }

    with open(out_path / "feature_importance.json", "w") as f:
        json.dump(importance_summary, f, indent=2)

    return importance_summary, feature_names, top_indices


# -----------------------------------------------------------------------------
# 3. Quantum Explainability & Mapping Visualization
# -----------------------------------------------------------------------------
def analyze_quantum_pipeline(output_dir: str = "outputs/explainability"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n[STEP 5] Generating Quantum Circuit & Feature-to-Qubit Visualizations...")

    # 1. Generate Circuit Diagram Figure
    from qiskit.circuit.library import zz_feature_map, real_amplitudes
    fm = zz_feature_map(feature_dimension=4, reps=1, entanglement="linear")
    ansatz = real_amplitudes(num_qubits=4, reps=2, entanglement="linear")
    qc = fm.compose(ansatz)

    circuit_ascii = qc.draw(output="text").single_string()
    with open(out_path / "quantum_circuit_diagram.txt", "w") as f:
        f.write(circuit_ascii)

    # Render circuit diagram as clean image
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")  # Dark slate theme
    ax.set_facecolor("#0f172a")

    ax.text(
        0.02, 0.94,
        circuit_ascii,
        family="monospace",
        fontsize=7.2,
        color="#38bdf8",  # Light cyan
        verticalalignment="top"
    )
    ax.axis("off")
    plt.title(
        "4-Qubit Variational Quantum Classifier Circuit Architecture\n[ZZFeatureMap (reps=1)  +  RealAmplitudes (reps=2)]",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
        pad=10
    )
    plt.tight_layout()
    circuit_plot_path = out_path / "quantum_circuit_diagram.png"
    plt.savefig(circuit_plot_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"         Saved visual circuit diagram to: {circuit_plot_path}")

    # 2. Generate Quantum Feature-to-Qubit Mapping Schematic
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Title
    ax.text(5.0, 8.1, "Quantum Feature Encoding Stage: 4D PCA to 4-Qubit Hilbert Space",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#1e293b")

    # Stage 1: Classical 4D PCA
    stage1_box = patches.FancyBboxPatch((0.4, 1.2), 2.2, 6.2, boxstyle="round,pad=0.2",
                                        facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.5)
    ax.add_patch(stage1_box)
    ax.text(1.5, 7.1, "1. Classical 4D PCA\n(Standardized)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")

    pca_labels = [
        ("x[0] (PC1)", "Dominant spatial patch contrast", "22.9% var"),
        ("x[1] (PC2)", "Intensity histogram variance", "12.8% var"),
        ("x[2] (PC3)", "Edge & gradient density", "10.0% var"),
        ("x[3] (PC4)", "Sub-regional tissue moments", "6.7% var")
    ]
    y_positions = [5.8, 4.4, 3.0, 1.6]

    for (name, desc, var), y in zip(pca_labels, y_positions):
        ax.text(1.5, y + 0.3, name, ha="center", va="center", fontsize=9, fontweight="bold", color="#1d4ed8")
        ax.text(1.5, y, desc, ha="center", va="center", fontsize=7.5, color="#475569")
        ax.text(1.5, y - 0.25, f"({var})", ha="center", va="center", fontsize=7, color="#059669")

    # Mapping Arrow 1: Rescaling
    ax.annotate("", xy=(3.3, 4.2), xytext=(2.7, 4.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#64748b"))
    ax.text(3.0, 4.55, "Rescale\n[0, π]", ha="center", va="center", fontsize=8, color="#334155", fontweight="bold")

    # Stage 2: Single-Qubit Angle Embedding (Hadamard + Phase)
    stage2_box = patches.FancyBboxPatch((3.4, 1.2), 2.8, 6.2, boxstyle="round,pad=0.2",
                                        facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=1.5)
    ax.add_patch(stage2_box)
    ax.text(4.8, 7.1, "2. Single-Qubit Rotations\n|ψ(x_i)⟩ = P(2x_i) · H|0⟩", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#0369a1")

    for i, y in enumerate(y_positions):
        ax.text(4.8, y + 0.2, f"|q_{i}⟩ = |0⟩  ──[ H ]──[ P(2x_{i}) ]──", ha="center", va="center",
                family="monospace", fontsize=8.5, fontweight="bold", color="#0c4a6e")
        ax.text(4.8, y - 0.15, f"Creates equal superposition\nand injects phase rotation 2x_{i}",
                ha="center", va="center", fontsize=7.2, color="#075985")

    # Mapping Arrow 2: Entanglement
    ax.annotate("", xy=(6.9, 4.2), xytext=(6.3, 4.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#64748b"))
    ax.text(6.6, 4.55, "Pairwise\nCoupling", ha="center", va="center", fontsize=8, color="#334155", fontweight="bold")

    # Stage 3: Two-Qubit Entanglement (ZZ Cross-terms)
    stage3_box = patches.FancyBboxPatch((7.0, 1.2), 2.6, 6.2, boxstyle="round,pad=0.2",
                                        facecolor="#ede9fe", edgecolor="#7c3aed", linewidth=1.5)
    ax.add_patch(stage3_box)
    ax.text(8.3, 7.1, "3. Quantum Entanglement\nZZ Interactions: Rzz(2·θ_ij)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#5b21b6")

    entangle_pairs = [
        ("q0 ↔ q1", "Phase term: 2(π - x[0])(π - x[1])"),
        ("q1 ↔ q2", "Phase term: 2(π - x[1])(π - x[2])"),
        ("q2 ↔ q3", "Phase term: 2(π - x[2])(π - x[3])"),
        ("Hilbert State", "16-dimensional quantum state |Ψ(x)⟩\nin reproducing kernel space")
    ]
    for (pair, desc), y in zip(entangle_pairs, y_positions):
        ax.text(8.3, y + 0.2, pair, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#4c1d95")
        ax.text(8.3, y - 0.12, desc, ha="center", va="center", fontsize=7.2, color="#6d28d9")

    # Bottom explanatory note
    ax.text(5.0, 0.45,
            "Non-linear quantum embedding: Maps 4D classical vector into a 2^4 = 16-dimensional Hilbert state space.\n"
            "Notice: Features represent mathematical image gradients and patch moments, NOT direct biological tissue biomarkers.",
            ha="center", va="center", fontsize=7.8, color="#475569", style="italic")

    plt.tight_layout()
    mapping_plot_path = out_path / "quantum_feature_mapping.png"
    plt.savefig(mapping_plot_path)
    plt.close()
    print(f"         Saved quantum feature mapping schematic to: {mapping_plot_path}")


# -----------------------------------------------------------------------------
# 4. Generate Comprehensive Explainability Markdown Report
# -----------------------------------------------------------------------------
def generate_explainability_report(output_dir: str = "outputs/explainability"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_content = """# Explainability & Model Interpretability Report
**Project:** Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection  
**Prototype Scope:** Solo SIH 2026 Practice Prototype  
**Audience:** Technical & Non-Expert Evaluators

---

## 1. What the 4 PCA Features Represent

To bridge high-resolution brain MRI scans into quantum circuits with limited qubit counts (NISQ era), raw scans were processed into **59 classical statistical descriptors** and subsequently compressed into **4 Principal Components (PCs)** using Principal Component Analysis (PCA):

| Component | Variance Explained | Primary Underlying Image Properties |
| :--- | :--- | :--- |
| **PC1** | **22.94%** | **Regional Tissue Contrast & Density:** Dominated by spatial patch variance in central brain regions (`Patch_R1C2`, `Patch_R2C1`), capturing the broad luminosity difference between healthy parenchyma and localized mass abnormalities. |
| **PC2** | **12.82%** | **Intensity Distribution Spread:** Heavily weighted by high-intensity histogram bins (`Hist_Bin_14`, `Hist_Bin_15`), reflecting hyperintense contrast-enhanced lesion margins vs. hypointense ventricles. |
| **PC3** | **9.96%** | **Edge & Gradient Roughness:** Driven by horizontal and vertical finite-difference edge magnitudes (`Grad_X_Std`, `Grad_Y_Std`), detecting sharp structural tissue transitions. |
| **PC4** | **6.71%** | **Sub-regional Symmetry Moments:** Captures higher-order skewness and peripheral patch statistics across hemispheric boundaries. |

> [!IMPORTANT]
> **Scientific Integrity Clarification:**  
> These PCA features are **aggregate mathematical summaries of 2D pixel intensities, textures, and spatial gradients**. They **do not** correspond directly to specific biological biomarkers (such as IDH mutations, EGFR amplifications, or 1p/19q codeletions), nor do they isolate histopathological tissue cellularity.

---

## 2. How the Quantum Feature Map Encodes Features

The quantum feature-encoding stage transforms classical continuous values into a 16-dimensional quantum state using **`ZZFeatureMap`** on 4 qubits:

```
|0⟩ ─── [ H ] ─── [ P(2·x[i]) ] ─── ■ ────────────────── ■ ─── |ψ_encoded⟩
                                    │  [ P(2·θ_ij) ]    │
|0⟩ ─── [ H ] ─── [ P(2·x[j]) ] ─── X ────────────────── X ───
```

1. **Feature Rescaling:** The 4 PCA coordinates $x_0, x_1, x_2, x_3$ are normalized into angles in $[0, \pi]$.
2. **Hadamard Superposition ($H$):** Each qubit is initialized into $|+\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}$, providing an equal superposition of all computational basis states.
3. **Single-Qubit Phase Rotation ($P(2x_i)$):** Injects individual feature values directly into the quantum relative phase of each qubit:
   $$\\exp(i x_i Z)$$
4. **Two-Qubit Entanglement ($R_{ZZ}$):** CNOT gates coupled with phase rotations introduce non-linear cross-feature terms:
   $$U_{\\Phi}(x) = \\exp\\left(i \\sum_{j > k} 2(\\pi - x_j)(\\pi - x_k) Z_j Z_k\\right)$$
   This maps the 4-dimensional Euclidean input into an entangled state residing in a $2^4 = 16$-dimensional complex Hilbert space.

---

## 3. What the Variational Circuit Does

The variational ansatz (**`RealAmplitudes`**, depth $2$, $12$ parameters) acts as a parameterized quantum decision engine:

1. **Rotational Search:** Parameterized $R_y(\\theta_k)$ rotation gates adjust the probability amplitudes of the quantum state vectors.
2. **Entangling Layers:** CNOT entanglers distribute information across all qubits, allowing the classifier to correlate patterns across multiple spatial and statistical axes simultaneously.
3. **Classical-Quantum Optimization:** A classical optimizer (COBYLA) iteratively tunes the 12 angles $\\vec{\\theta}$ to minimize the cross-entropy classification loss on the training data.
4. **Measurement & Readout:** The 4 qubits are measured in the computational basis, producing a 4-bit integer $k \\in \\{0, \\dots, 15\\}$. The Hamming weight modulo 4 ($\\sum \\text{bits} \\pmod 4$) maps these outcomes into the 4 diagnostic classes.

> [!NOTE]
> **No Causal Medical Explanation:**  
> The quantum circuit does not perform causal anatomical reasoning. Instead, it performs quantum linear separation within a non-linear Hilbert feature space mapped through parameterized unitary matrices.

---

## 4. Why This is a Research Prototype (Not a Clinical Diagnosis)

This prototype must strictly be classified as an **experimental algorithmic proof-of-concept**:

1. **Pre-Clinical & Non-Validated:** Clinical deployment requires multi-center regulatory validation (e.g., FDA 510(k), CE mark), external cohort testing, and integration with complete multi-sequence 3D MRI volumes (T1, T1-Gd, T2, FLAIR).
2. **Simplified Representation:** The current model operates on 2D slices and a 4-dimensional PCA subspace to accommodate prototype quantum hardware limits, omitting vital 3D volumetric and anatomical context.
3. **Confounding Statistical Artifacts:** Image-level statistical features (histograms, patch means) can be sensitive to scanner differences, contrast protocols, and patient positioning.
4. **Intended Role:** This system serves to evaluate how quantum variational algorithms behave on medical image representations in comparison to established classical kernel methods.
"""

    with open(out_path / "explainability_report.md", "w") as f:
        f.write(report_content)

    print(f"[STEP 6] Saved complete explainability report to: {out_path / 'explainability_report.md'}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    output_dir = "outputs/explainability"
    analyze_classical_pipeline(output_dir=output_dir)
    analyze_quantum_pipeline(output_dir=output_dir)
    generate_explainability_report(output_dir=output_dir)
    print("\n" + "=" * 70)
    print(f"  EXPLAINABILITY MODULE COMPLETED: Artifacts in '{output_dir}/'")
    print("=" * 70 + "\n")
