# Explainability & Model Interpretability Report
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
2. **Hadamard Superposition ($H$):** Each qubit is initialized into $|+angle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$, providing an equal superposition of all computational basis states.
3. **Single-Qubit Phase Rotation ($P(2x_i)$):** Injects individual feature values directly into the quantum relative phase of each qubit:
   $$\exp(i x_i Z)$$
4. **Two-Qubit Entanglement ($R_{ZZ}$):** CNOT gates coupled with phase rotations introduce non-linear cross-feature terms:
   $$U_{\Phi}(x) = \exp\left(i \sum_{j > k} 2(\pi - x_j)(\pi - x_k) Z_j Z_k\right)$$
   This maps the 4-dimensional Euclidean input into an entangled state residing in a $2^4 = 16$-dimensional complex Hilbert space.

---

## 3. What the Variational Circuit Does

The variational ansatz (**`RealAmplitudes`**, depth $2$, $12$ parameters) acts as a parameterized quantum decision engine:

1. **Rotational Search:** Parameterized $R_y(\theta_k)$ rotation gates adjust the probability amplitudes of the quantum state vectors.
2. **Entangling Layers:** CNOT entanglers distribute information across all qubits, allowing the classifier to correlate patterns across multiple spatial and statistical axes simultaneously.
3. **Classical-Quantum Optimization:** A classical optimizer (COBYLA) iteratively tunes the 12 angles $\vec{\theta}$ to minimize the cross-entropy classification loss on the training data.
4. **Measurement & Readout:** The 4 qubits are measured in the computational basis, producing a 4-bit integer $k \in \{0, \dots, 15\}$. The Hamming weight modulo 4 ($\sum \text{bits} \pmod 4$) maps these outcomes into the 4 diagnostic classes.

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
