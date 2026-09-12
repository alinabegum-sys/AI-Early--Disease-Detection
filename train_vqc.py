"""
Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection
------------------------------------------------------------------------
Module: Variational Quantum Classifier (VQC) using Qiskit Machine Learning

RESEARCH USE ONLY:
This module is developed strictly for algorithmic research and educational
benchmarking of quantum machine learning against classical baselines. It is NOT
intended, certified, or validated for medical diagnostic use, clinical decision
support, or patient health evaluation. NO quantum advantage is claimed.

Architecture:
- Hardware/Simulation: Qiskit StatevectorSampler (local noise-free statevector simulation)
- Circuit Register: 4 Qubits (q0, q1, q2, q3)
- Feature Map: 4-Qubit Second-order Pauli-Z Expansion (ZZFeatureMap / zz_feature_map, reps=1)
  * Encodes continuous 4D PCA features x in [0, pi] into quantum phase and entanglement.
- Variational Ansatz: RealAmplitudes (real_amplitudes, reps=2, linear entanglement)
  * Parameterized single-qubit Ry rotation gates interleaved with entangling CNOT gates.
- Measurement & Interpretation: Bitstring parity modulo 4 mapping 16 basis states to 4 classes.
- Optimizer: Simultaneous Perturbation Stochastic Approximation (SPSA).
"""

import os
import sys

# Configure non-interactive matplotlib backend and cache
os.environ["MPLCONFIGDIR"] = "/tmp"

import json
import time
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Qiskit and Qiskit Machine Learning
import qiskit
import qiskit_machine_learning
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SPSA


def parity_mod4(bitstring_int: int) -> int:
    """
    Legacy interpretation function: Hamming weight modulo 4.
    Kept for backward compatibility and reference.
    """
    return sum(int(b) for b in bin(bitstring_int)[2:]) % 4


def balanced_readout_mod4(bitstring_int: int) -> int:
    """
    Balanced interpretation function for mapping 4-qubit measurement outcomes
    (integers 0 to 15) into 4 diagnostic classes (0, 1, 2, 3).
    Measures the 2-qubit computational subspace (bitstring_int % 4), assigning
    EXACTLY 4 basis states (25% uniform base rate) to each class:
      - Class 0 (glioma)    : |0000>, |0100>, |1000>, |1100> (states 0, 4, 8, 12)
      - Class 1 (meningioma): |0001>, |0101>, |1001>, |1101> (states 1, 5, 9, 13)
      - Class 2 (no_tumor)  : |0010>, |0110>, |1010>, |1110> (states 2, 6, 10, 14)
      - Class 3 (pituitary) : |0011>, |0111>, |1011>, |1111> (states 3, 7, 11, 15)
    """
    return bitstring_int % 4



def load_preprocessed_quantum_data(data_dir: str = "processed_data"):
    """
    Loads the exact same 4D PCA features and labels used by the classical baseline,
    using the quantum-scaled representation (range [0, pi]) for angle encoding.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory '{data_dir}' not found. "
            f"Please run 'preprocess.py' first."
        )

    # Quantum angle-scaled features in [0, pi]
    X_train = np.load(path / "X_train_quantum.npy")
    X_test = np.load(path / "X_test_quantum.npy")
    y_train = np.load(path / "y_train.npy")
    y_test = np.load(path / "y_test.npy")

    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    class_map = {int(k): v for k, v in metadata["classes"].items()}
    class_names = [class_map[i] for i in range(len(class_map))]

    return X_train, X_test, y_train, y_test, class_names, metadata


def calculate_multiclass_specificity(cm: np.ndarray) -> tuple[dict[int, float], float]:
    """
    Computes per-class and macro-averaged Specificity (True Negative Rate)
    using One-vs-Rest formulation for multiclass confusion matrices.
    """
    num_classes = cm.shape[0]
    total_samples = np.sum(cm)
    per_class_specificity = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - (tp + fp + fn)

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_specificity[i] = float(spec)

    macro_specificity = float(np.mean(list(per_class_specificity.values())))
    return per_class_specificity, macro_specificity


def build_and_train_vqc(
    data_dir: str = "processed_data",
    output_dir: str = "results",
    model_dir: str = "models",
    max_iter: int = 100,
    ansatz_reps: int = 2
):
    """
    Constructs the 4-qubit quantum classifier, trains with SPSA on local simulator,
    evaluates performance, and saves all experimental artifacts.
    """
    out_path = Path(output_dir)
    mod_path = Path(model_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  VARIATIONAL QUANTUM CLASSIFIER (VQC): RESEARCH PROTOTYPE")
    print("=" * 70)
    print(f" Qiskit Core Version          : {qiskit.__version__}")
    print(f" Qiskit ML Version            : {qiskit_machine_learning.__version__}")

    # 1. Load Data (exact same train/test split as classical SVM)
    X_train, X_test, y_train, y_test, class_names, meta = load_preprocessed_quantum_data(data_dir)
    num_qubits = X_train.shape[1]
    num_classes = len(class_names)

    print(f" Train Samples                : {len(X_train)} scans")
    print(f" Test Samples                 : {len(X_test)} scans")
    print(f" Input Feature Dimension      : {num_qubits} (mapped 1:1 to {num_qubits} Qubits)")
    print(f" Feature Scaling Range        : [0, π] (standard angle embedding)")
    print(f" Target Classes               : {class_names}")

    # 2. Design Quantum Circuit Components
    print("\n[STEP 1] Constructing 4-Qubit Quantum Circuits...")

    # Feature map: ZZ second-order Pauli expansion
    feature_map = zz_feature_map(
        feature_dimension=num_qubits,
        reps=1,
        entanglement="linear"
    )

    # Variational ansatz: RealAmplitudes with Ry rotations and CNOT entanglers
    ansatz = real_amplitudes(
        num_qubits=num_qubits,
        reps=ansatz_reps,
        entanglement="linear"
    )

    num_parameters = ansatz.num_parameters
    print(f"         Feature Map          : ZZFeatureMap (reps=1, qubits={num_qubits})")
    print(f"         Variational Ansatz   : RealAmplitudes (reps={ansatz_reps}, params={num_parameters})")
    print(f"         Local Simulator      : Qiskit StatevectorSampler (noise-free)")

    # Combined circuit visualization saved to file
    full_circuit = feature_map.compose(ansatz)
    circuit_str = full_circuit.draw(output="text").single_string()
    with open(mod_path / "quantum_circuit_architecture.txt", "w") as f:
        f.write("=== 4-QUBIT VQC CIRCUIT ARCHITECTURE ===\n\n")
        f.write("FEATURE MAP (ZZFeatureMap):\n")
        f.write(feature_map.draw(output="text").single_string())
        f.write("\n\nVARIATIONAL ANSATZ (RealAmplitudes):\n")
        f.write(ansatz.draw(output="text").single_string())
        f.write("\n\nCOMPOSED CIRCUIT (Feature Map + Ansatz):\n")
        f.write(circuit_str)
    print(f"         Circuit ASCII saved  : {mod_path / 'quantum_circuit_architecture.txt'}")

    # 3. Setup Optimizer and VQC Model
    training_loss_history = []

    def optimization_callback(*args, **kwargs):
        # Supports both SciPyOptimizer (weights, obj_val) and SPSA (nfev, x_next, fx_next, step, accepted)
        if len(args) == 2:
            weights, obj_val = args
        elif len(args) == 5:
            nfev, x_next, fx_next, step_size, accepted = args
            weights, obj_val = x_next, fx_next
        else:
            obj_val = args[0] if args else 0.0

        training_loss_history.append(float(obj_val))
        step = len(training_loss_history)
        if step % 20 == 0 or step == 1 or step == max_iter:
            print(f"         [Iter {step:03d}/{max_iter:03d}] Cross-Entropy Loss: {obj_val:.4f}")

    optimizer = SPSA(maxiter=max_iter)
    sampler = StatevectorSampler()

    print("\n[STEP 1b] Exact 16-State Computational Basis -> Class Readout Mapping (Balanced 4 States/Class):")
    for b in range(16):
        c = balanced_readout_mod4(b)
        print(f"         State {b:02d} (|{b:04b}>) -> Class {c} ({class_names[c]})")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        interpret=balanced_readout_mod4,
        output_shape=num_classes,
        callback=optimization_callback
    )

    # 4. Train VQC on Training Partition
    print(f"\n[STEP 2] Training VQC via SPSA Optimizer ({max_iter} iterations) with Balanced Readout...")
    train_start = time.perf_counter()
    vqc.fit(X_train, y_train)
    train_duration = time.perf_counter() - train_start
    print(f"         Quantum model training completed in: {train_duration:.2f} s ({train_duration * 1000:.1f} ms)")

    # 5. Evaluate on Training and Held-Out Test Sets
    print("\n[STEP 3] Generating Predictions on Train and Held-Out Test Partitions...")
    y_train_pred = vqc.predict(X_train)
    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    final_loss = float(training_loss_history[-1]) if training_loss_history else None

    pred_start = time.perf_counter()
    y_pred = vqc.predict(X_test)
    pred_duration = time.perf_counter() - pred_start
    per_sample_latency = (pred_duration / len(X_test)) * 1000

    # 6. Compute Evaluation Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    # Confusion matrix & Specificity
    cm = confusion_matrix(y_test, y_pred)
    per_class_spec, macro_spec = calculate_multiclass_specificity(cm)

    # 7. Save Confusion Matrix Plot
    cm_plot_path = out_path / "confusion_matrix_vqc.png"
    plt.figure(figsize=(7, 6), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title("Confusion Matrix - Variational Quantum Classifier (VQC)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=10, labelpad=8)
    plt.ylabel("True Class", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"[STEP 4] Saved VQC confusion matrix plot: {cm_plot_path}")

    # 8. Save Trained Model & Variational Parameters
    weights_path = mod_path / "vqc_optimal_weights.npy"
    np.save(weights_path, vqc.weights)
    print(f"[STEP 5] Saved optimal variational parameters: {weights_path}")

    vqc_save_path = mod_path / "vqc_model.pkl"
    try:
        # Clear callback attribute prior to serialization
        vqc._callback = None
        if hasattr(vqc.optimizer, "callback"):
            vqc.optimizer.callback = None
        with open(vqc_save_path, "wb") as f:
            pickle.dump(vqc, f)
        print(f"         Serialized VQC model: {vqc_save_path}")
    except Exception as e:
        print(f"         [WARN] Full model serialization note: {e}")

    # 9. Compile Metrics Record
    metrics_record = {
        "model_type": "Variational Quantum Classifier (VQC)",
        "framework": f"Qiskit {qiskit.__version__} / Qiskit-ML {qiskit_machine_learning.__version__}",
        "num_qubits": num_qubits,
        "feature_map": "ZZFeatureMap (reps=1, linear)",
        "variational_ansatz": f"RealAmplitudes (reps={ansatz_reps}, params={num_parameters})",
        "optimizer": f"SPSA (maxiter={max_iter})",
        "readout_mapping": "Balanced 2-Qubit Computational Subspace (bitstring_int % 4, exactly 4 states/class)",
        "simulator": "StatevectorSampler",
        "disclaimer": "Experimental research classification prototype. Not for clinical diagnosis. No quantum advantage claimed.",
        "input_features": "4D PCA Features scaled to [0, pi] angle range",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "training_accuracy": float(train_accuracy),
        "final_loss": float(final_loss) if final_loss is not None else None,
        "training_time_seconds": float(train_duration),
        "total_inference_time_ms": float(pred_duration * 1000),
        "per_sample_inference_latency_ms": float(per_sample_latency),
        "overall_metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(precision_macro),
            "macro_recall_sensitivity": float(recall_macro),
            "macro_specificity": float(macro_spec),
            "macro_f1_score": float(f1_macro),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1_score": float(f1_weighted)
        },
        "per_class_metrics": {
            class_names[i]: {
                "precision": float(prec_per_class[i]),
                "recall_sensitivity": float(rec_per_class[i]),
                "specificity": float(per_class_spec[i]),
                "f1_score": float(f1_per_class[i])
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist()
    }

    metrics_save_path = out_path / "vqc_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_record, f, indent=2)

    # 10. Print Comprehensive Evaluation Report
    print_vqc_summary(metrics_record, class_names)

    return metrics_record


def print_vqc_summary(m: dict, class_names: list[str]):
    """
    Step 6: Prints formatted evaluation table, quantum specifications, and timing profile.
    """
    ov = m["overall_metrics"]
    print("\n" + "-" * 70)
    print("  EVALUATION METRICS SUMMARY (VARIATIONAL QUANTUM CLASSIFIER)")
    print("-" * 70)
    if "training_accuracy" in m:
        print(f" Training Accuracy        : {m['training_accuracy'] * 100:6.2f}%")
    if "final_loss" in m and m["final_loss"] is not None:
        print(f" Final Optimization Loss  : {m['final_loss']:6.4f}")
    print(f" Test Accuracy            : {ov['accuracy'] * 100:6.2f}%")
    print(f" Macro Precision          : {ov['macro_precision'] * 100:6.2f}%")
    print(f" Macro Recall/Sensitivity : {ov['macro_recall_sensitivity'] * 100:6.2f}%")
    print(f" Macro Specificity        : {ov['macro_specificity'] * 100:6.2f}%")
    print(f" Macro F1-Score           : {ov['macro_f1_score'] * 100:6.2f}%")
    print(f" Weighted F1-Score        : {ov['weighted_f1_score'] * 100:6.2f}%")

    print("\n Per-Class Performance Breakdown:")
    header = f" {'Class Name':<14} | {'Precision':<10} | {'Recall/Sens':<12} | {'Specificity':<12} | {'F1-Score':<10}"
    print(header)
    print(" " + "-" * len(header))
    for cname in class_names:
        c = m["per_class_metrics"][cname]
        print(f" {cname:<14} | {c['precision']*100:8.2f}% | {c['recall_sensitivity']*100:10.2f}% | {c['specificity']*100:10.2f}% | {c['f1_score']*100:8.2f}%")

    print("\n Computational Benchmarks:")
    print(f" • Training Duration        : {m['training_time_seconds']:.2f} s ({m['training_time_seconds'] * 1000:.1f} ms)")
    print(f" • Total Test Inference Time: {m['total_inference_time_ms']:6.2f} ms ({m['test_samples']} samples)")
    print(f" • Latency per Sample       : {m['per_sample_inference_latency_ms']:6.4f} ms/sample")

    print("\n Quantum System Specifications:")
    print(f" • Qubits Allocated         : {m['num_qubits']}")
    print(f" • Feature Map              : {m['feature_map']}")
    print(f" • Variational Ansatz       : {m['variational_ansatz']}")
    print(f" • Optimizer                : {m['optimizer']}")
    print(f" • Backend Simulator        : {m['simulator']}")

    print("\n Research Artifacts Stored:")
    print(" • Trained Model            : models/vqc_model.pkl")
    print(" • Optimal Variational Angles: models/vqc_optimal_weights.npy")
    print(" • Circuit Architecture ASCII: models/quantum_circuit_architecture.txt")
    print(" • Confusion Matrix Graphic : results/confusion_matrix_vqc.png")
    print(" • Detailed JSON Metrics    : results/vqc_metrics.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    build_and_train_vqc()
