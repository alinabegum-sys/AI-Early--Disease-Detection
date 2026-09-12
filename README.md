# Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection
**Smart India Hackathon (SIH) 2026 — Research Practice Prototype**

> [!IMPORTANT]
> **Research & Ethical Disclaimer:**  
> This software is strictly an experimental educational prototype built for comparing quantum variational classifiers with classical machine learning on compressed MRI representations. It is **NOT** a medical diagnostic device, has not undergone clinical certification, and must **NOT** be used for clinical decision-making or patient evaluation.

---

## 🔬 System Architecture

```
Brain MRI Scan (2D Slice)
       │
       ▼
1. Preprocessing & Normalization
   • Resizing to 64x64 resolution
   • Normalization [0, 1]
       │
       ▼
2. Classical Feature Extraction
   • Global statistical moments (7)
   • Intensity histogram (16 bins)
   • 4x4 spatial patch pooling (32)
   • Sobel-like edge gradients (4)
   • Total: 59 descriptors
       │
       ▼
3. Dimensionality Reduction (PCA)
   • 59 features ➔ 4 Principal Components (52.4% cumulative variance)
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
4A. Classical Model                  4B. Quantum Model
   • Support Vector Machine (SVC)        • 4-Qubit Variational Quantum Classifier (VQC)
   • RBF Kernel                          • Feature Map: ZZFeatureMap (reps=1)
   • Infinite-dim Hilbert space          • Ansatz: RealAmplitudes (reps=2, 12 params)
   • Low-latency benchmark               • Local Simulator: Qiskit StatevectorSampler
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼
             5. Evaluation & Explainability
                • Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1
                • Side-by-side benchmarking & confusion matrices
                • Permutation feature importance & PCA loadings heatmap
                • Quantum feature-to-qubit angle encoding visualization
```

---

## 🚀 Quick Start Guide

### 1. Activate Environment
Ensure you are using the project's Python 3.11 virtual environment:
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts (Optional — Precomputed Artifacts Included)
To re-run any stage from scratch:
```bash
# 1. Preprocess dataset & extract 4D PCA features:
python preprocess.py

# 2. Train and evaluate the Classical SVM baseline:
python train_svm.py

# 3. Train and evaluate the 4-Qubit Quantum VQC:
python train_vqc.py

# 4. Generate explainability & feature mapping reports:
python explainability.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Or directly using the virtual environment binary:
```bash
./.venv/bin/streamlit run app.py
```

---

## 📊 Experimental Results Summary (Held-Out Test Set)

| Metric | Classical SVM (RBF Kernel) | Quantum VQC (4 Qubits, 12 Params) |
| :--- | :--- | :--- |
| **Accuracy** | **100.00%** | **37.50%** |
| **Macro Precision** | **100.00%** | **39.85%** |
| **Macro Sensitivity / Recall** | **100.00%** | **37.50%** |
| **Macro Specificity** | **100.00%** | **79.17%** |
| **Macro F1-Score** | **100.00%** | **35.46%** |
| **Training Latency** | **3.00 ms** | **9.22 s** |
| **Inference Latency** | **0.0062 ms/sample** | **1.9351 ms/sample** |

---

## 📁 Repository Structure

```text
├── app.py                      # Interactive Streamlit Web Dashboard
├── preprocess.py               # Data loading, 59-feature extraction, 4D PCA
├── train_svm.py                # Classical SVM training & evaluation
├── train_vqc.py                # 4-Qubit Qiskit VQC training & evaluation
├── explainability.py           # Feature importance & quantum encoding visualizations
├── requirements.txt            # Package dependencies
├── README.md                   # Project documentation & execution guide
├── data/
│   └── BRISC/                  # Brain MRI scans (glioma, meningioma, no_tumor, pituitary)
├── processed_data/             # Preprocessed PCA arrays, scalers & metadata
├── models/                     # Serialized SVM, VQC weights & circuit diagrams
├── results/                    # Confusion matrix graphics & JSON metric logs
└── outputs/
    └── explainability/         # Interpretability diagrams, heatmaps & report
```
