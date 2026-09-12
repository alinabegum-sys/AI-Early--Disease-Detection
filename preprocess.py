"""
Hybrid Quantum Machine Learning Platform for Early Brain Tumor Detection
------------------------------------------------------------------------
Data Preprocessing Pipeline for the BRISC Brain MRI Dataset

Steps:
1. Load brain MRI images and labels from class-based directory structure.
2. Resize images to prototype resolution (64x64) and normalize pixel values [0, 1].
3. Perform stratified train/test split.
4. Extract lightweight classical numerical features (spatial patch statistics,
   intensity histograms, and gradient/edge descriptors).
5. Apply Principal Component Analysis (PCA) to reduce feature space to 4 dimensions
   (optimized for a 4-qubit Quantum Variational Classifier / VQC and classical SVM).
6. Save preprocessed train/test arrays and metadata for downstream model training.
7. Print a comprehensive pipeline summary.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# Default BRISC tumor classes
BRISC_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# Default settings
DEFAULT_IMAGE_SIZE = (64, 64)   # Prototype resolution
PCA_COMPONENTS = 4              # Aligned with a 4-qubit Quantum Circuit
TEST_SPLIT_RATIO = 0.20
RANDOM_STATE = 42


def generate_synthetic_prototype_dataset(data_dir: Path, samples_per_class: int = 25) -> None:
    """
    Creates a synthetic prototype dataset mimicking the BRISC MRI structure.
    Used for end-to-end pipeline verification when the full dataset is not yet staged locally.
    """
    print(f"\n[INFO] Generating synthetic prototype dataset in: {data_dir}")
    rng = np.random.default_rng(RANDOM_STATE)

    for class_name in BRISC_CLASSES:
        class_folder = data_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            # Generate synthetic brain-slice-like circular phantom with noise
            img_arr = np.zeros((128, 128), dtype=np.float32)
            y, x = np.ogrid[:128, :128]
            center_dist = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)

            # Brain skull / tissue mask
            brain_mask = center_dist < 48
            img_arr[brain_mask] = 0.5 + rng.normal(0, 0.08, size=np.count_nonzero(brain_mask))

            # Simulate distinct tumor signature per pathology
            if class_name == "glioma":
                tumor_mask = np.sqrt((x - 45) ** 2 + (y - 45) ** 2) < 14
                img_arr[tumor_mask] += 0.35
            elif class_name == "meningioma":
                tumor_mask = np.sqrt((x - 85) ** 2 + (y - 50) ** 2) < 12
                img_arr[tumor_mask] += 0.45
            elif class_name == "pituitary":
                tumor_mask = np.sqrt((x - 64) ** 2 + (y - 82) ** 2) < 10
                img_arr[tumor_mask] += 0.38
            # no_tumor leaves the standard tissue pattern

            # Clip and save as grayscale 8-bit PNG
            img_arr = np.clip(img_arr, 0.0, 1.0)
            img_uint8 = (img_arr * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8, mode="L")
            img.save(class_folder / f"sample_{i+1:03d}.png")

    print(f"[INFO] Synthetic dataset generated with {len(BRISC_CLASSES) * samples_per_class} images across 4 classes.\n")


def discover_and_load_images(data_dir: Path, target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE):
    """
    Step 1 & 2: Traverses dataset directories, loads MRI scans, resizes and normalizes them.
    Handles standard BRISC folder layout:
      data_dir/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── no_tumor/
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    # Find candidate class subdirectories
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Check for nested train/test structure if present
    if len(subdirs) == 1 and subdirs[0].name.lower() in ["train", "dataset", "images"]:
        subdirs = [d for d in subdirs[0].iterdir() if d.is_dir() and not d.name.startswith(".")]

    # Normalize class mapping
    class_map = {}
    for d in sorted(subdirs):
        canonical_name = d.name.lower().replace("-", "_").replace(" ", "_")
        class_map[canonical_name] = d

    if not class_map:
        raise ValueError(f"No class folders found inside '{data_dir}'.")

    # Build consistent label indexing
    unique_classes = sorted(list(class_map.keys()))
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    images = []
    labels = []
    filepaths = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    print(f"[STEP 1] Scanning dataset in: {data_dir.resolve()}")
    print(f"         Detected classes: {unique_classes}")

    for class_name, folder_path in class_map.items():
        class_files = [f for f in folder_path.glob("*") if f.suffix.lower() in valid_extensions]
        for img_path in sorted(class_files):
            try:
                # Load as grayscale (1-channel luminance for MRI)
                with Image.open(img_path) as pil_img:
                    resized = pil_img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
                    # Normalize pixel values to [0.0, 1.0]
                    norm_arr = np.asarray(resized, dtype=np.float32) / 255.0

                    images.append(norm_arr)
                    labels.append(class_to_idx[class_name])
                    filepaths.append(str(img_path))
            except Exception as e:
                print(f"[WARN] Skipping corrupted file {img_path.name}: {e}")

    if len(images) == 0:
        raise ValueError(f"No valid image files found in '{data_dir}'.")

    images = np.stack(images, axis=0)  # Shape: (N, H, W)
    labels = np.array(labels, dtype=np.int64)

    return images, labels, class_to_idx, idx_to_class, filepaths


def extract_classical_features(images: np.ndarray) -> np.ndarray:
    """
    Step 3: Lightweight classical feature extraction.
    Computes for each 2D brain MRI scan:
      1. Global intensity moments (mean, std, min, max, median, 25th/75th percentiles)
      2. 16-bin normalized intensity histogram
      3. Block-level spatial intensity statistics (4x4 grid = 16 patches x [mean, std])
      4. Horizontal and vertical finite-difference gradient magnitudes (edge density)
    """
    num_samples, h, w = images.shape
    features_list = []

    for i in range(num_samples):
        img = images[i]

        # 1. Global statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        median_val = np.median(img)
        p25 = np.percentile(img, 25)
        p75 = np.percentile(img, 75)
        global_stats = [mean_val, std_val, min_val, max_val, median_val, p25, p75]

        # 2. Intensity histogram (16 bins in [0, 1])
        hist, _ = np.histogram(img, bins=16, range=(0.0, 1.0), density=True)

        # 3. Block-level spatial pooling (4x4 grid)
        grid_rows, grid_cols = 4, 4
        patch_h, patch_w = h // grid_rows, w // grid_cols
        block_features = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                patch = img[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
                block_features.extend([np.mean(patch), np.std(patch)])

        # 4. Gradient / edge approximation (Sobel-like difference)
        grad_x = np.abs(img[:, 1:] - img[:, :-1])
        grad_y = np.abs(img[1:, :] - img[:-1, :])
        gradient_stats = [
            np.mean(grad_x), np.std(grad_x),
            np.mean(grad_y), np.std(grad_y)
        ]

        # Concatenate into a single feature vector
        sample_feat = np.concatenate([
            global_stats,
            hist,
            block_features,
            gradient_stats
        ])
        features_list.append(sample_feat)

    features = np.array(features_list, dtype=np.float32)
    return features


def preprocess_pipeline(
    data_dir: str = "data/BRISC",
    output_dir: str = "processed_data",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    test_size: float = TEST_SPLIT_RATIO,
    n_pca_components: int = PCA_COMPONENTS,
    random_state: int = RANDOM_STATE,
    generate_synthetic_if_missing: bool = True
):
    """
    Full preprocessing execution pipeline.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # If dataset folder doesn't exist or is empty, create synthetic prototype
    if not data_path.exists() or not any(data_path.iterdir()):
        if generate_synthetic_if_missing:
            print(f"[NOTICE] Dataset directory '{data_dir}' is not populated.")
            generate_synthetic_prototype_dataset(data_path, samples_per_class=30)
        else:
            raise FileNotFoundError(f"Dataset path '{data_dir}' not found.")

    # 1. Load, resize, normalize images
    images, labels, class_to_idx, idx_to_class, filepaths = discover_and_load_images(
        data_path, target_size=image_size
    )
    total_images = len(labels)

    # 2. Extract classical numerical features
    print(f"[STEP 2] Extracting lightweight classical image features...")
    raw_features = extract_classical_features(images)

    # 3. Stratified Train / Test Split
    print(f"[STEP 3] Performing stratified train/test split ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")
    indices = np.arange(total_images)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    X_train_feat = raw_features[train_idx]
    X_test_feat = raw_features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # 4. Standardize and apply PCA (4 dimensions for 4-qubit Quantum Circuit)
    print(f"[STEP 4] Fitting PCA to reduce feature space to {n_pca_components} dimensions...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    pca = PCA(n_components=n_pca_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Normalize PCA outputs to [0, pi] for angle embedding in Quantum VQC
    quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_quantum = quantum_scaler.fit_transform(X_train_pca)
    X_test_quantum = quantum_scaler.transform(X_test_pca)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.sum(explained_var)

    # 5. Save all preprocessed outputs for Classical SVM & Quantum VQC
    print(f"[STEP 5] Saving preprocessed arrays and transformation artifacts to: {out_path.resolve()}")

    # Save as separate .npy arrays
    np.save(out_path / "X_train_pca.npy", X_train_pca)
    np.save(out_path / "X_test_pca.npy", X_test_pca)
    np.save(out_path / "X_train_quantum.npy", X_train_quantum)
    np.save(out_path / "X_test_quantum.npy", X_test_quantum)
    np.save(out_path / "y_train.npy", y_train)
    np.save(out_path / "y_test.npy", y_test)

    # Save single composite compressed archive
    np.savez_compressed(
        out_path / "dataset_preprocessed.npz",
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        X_train_quantum=X_train_quantum,
        X_test_quantum=X_test_quantum,
        y_train=y_train,
        y_test=y_test,
        train_indices=train_idx,
        test_indices=test_idx
    )

    # Save metadata and label mappings
    metadata = {
        "classes": idx_to_class,
        "class_to_idx": class_to_idx,
        "total_images": total_images,
        "image_shape": [image_size[0], image_size[1], 1],
        "extracted_feature_dim": int(raw_features.shape[1]),
        "pca_components": n_pca_components,
        "pca_explained_variance_ratio": [float(v) for v in explained_var],
        "pca_cumulative_variance": float(cumulative_var),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "class_distribution_train": {idx_to_class[k]: int(v) for k, v in Counter(y_train).items()},
        "class_distribution_test": {idx_to_class[k]: int(v) for k, v in Counter(y_test).items()}
    }

    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save transformation models (scaler, PCA, quantum_scaler)
    with open(out_path / "transformers.pkl", "wb") as f:
        pickle.dump({
            "feature_scaler": scaler,
            "pca": pca,
            "quantum_scaler": quantum_scaler
        }, f)

    # 6. Print Formatted Summary
    print_pipeline_summary(metadata, images.shape, raw_features.shape, X_train_pca.shape)

    return metadata


def print_pipeline_summary(meta: dict, image_shape: tuple, feature_shape: tuple, pca_shape: tuple):
    """
    Step 6: Prints a structured, human-readable summary of the preprocessing results.
    """
    print("\n" + "=" * 68)
    print("  PREPROCESSING PIPELINE SUMMARY: BRAIN TUMOR DETECTION (BRISC)")
    print("=" * 68)
    print(f" Total Images Processed       : {meta['total_images']}")
    print(f" Number of Target Classes     : {len(meta['classes'])}")
    print(" Class Distribution (Total)   :")
    for idx, name in meta['classes'].items():
        total_cls = meta['class_distribution_train'].get(name, 0) + meta['class_distribution_test'].get(name, 0)
        print(f"   • {name:<14}: {total_cls:4d} scans")

    print("\n Dimension Profiles           :")
    print(f"   • Image Array Shape        : {image_shape[1:]} (Grayscale)")
    print(f"   • Extracted Features       : {feature_shape[1]} descriptors per scan")
    print(f"   • PCA Reduced Dimension    : {meta['pca_components']} features (for 4-qubit Quantum VQC)")
    print(f"   • Cumulative Variance (PCA): {meta['pca_cumulative_variance'] * 100:.2f}%")

    print("\n Dataset Partitioning         :")
    print(f"   • Training Set Size        : {meta['train_samples']} samples ({meta['train_samples']/meta['total_images']*100:.1f}%)")
    print(f"   • Testing Set Size         : {meta['test_samples']} samples ({meta['test_samples']/meta['total_images']*100:.1f}%)")
    print(f"   • Stratified Train Counts  : {meta['class_distribution_train']}")
    print(f"   • Stratified Test Counts   : {meta['class_distribution_test']}")

    print("\n Output Artifacts Ready for Training:")
    print("   • Classical SVM Inputs     : processed_data/X_train_pca.npy, X_test_pca.npy")
    print("   • Quantum VQC Inputs       : processed_data/X_train_quantum.npy, X_test_quantum.npy (range [0, π])")
    print("   • Target Labels            : processed_data/y_train.npy, y_test.npy")
    print("   • Pipeline Metadata        : processed_data/metadata.json")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BRISC Brain MRI Dataset for Quantum-Classical ML.")
    parser.add_argument("--data-dir", type=str, default="data/BRISC", help="Path to BRISC dataset folder.")
    parser.add_argument("--output-dir", type=str, default="processed_data", help="Output directory for processed data.")
    parser.add_argument("--img-size", type=int, default=64, help="Target image square resolution (default: 64).")
    parser.add_argument("--pca-dim", type=int, default=4, help="Number of PCA components (default: 4).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion (default: 0.2).")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic prototype if data folder is missing.")

    args = parser.parse_args()

    preprocess_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=(args.img_size, args.img_size),
        test_size=args.test_size,
        n_pca_components=args.pca_dim,
        generate_synthetic_if_missing=True
    )
