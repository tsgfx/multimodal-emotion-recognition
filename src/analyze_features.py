from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import FIGURE_DIR, LABELS, PROCESSED_ROOT
from utils import ensure_dirs


def load_audio_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        feature_path = getattr(row, "audio_feature_path", "")
        if not isinstance(feature_path, str) or not Path(feature_path).exists():
            continue
        data = np.load(feature_path)
        mfcc = data["mfcc"]
        rows.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "rms_mean": float(data["rms_mean"][0]),
                "zcr_mean": float(data["zcr_mean"][0]),
                "centroid_mean": float(data["centroid_mean"][0]),
                **{f"mfcc_{i + 1}": float(mfcc[i].mean()) for i in range(mfcc.shape[0])},
            }
        )
    return pd.DataFrame(rows)


def plot_audio_stats(stats: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=stats, x="label", y="rms_mean", order=LABELS)
    plt.title("Audio Energy by Emotion")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "audio_energy_boxplot.png", dpi=180)
    plt.close()

    mfcc_cols = [col for col in stats.columns if col.startswith("mfcc_")]
    mfcc_mean = stats.groupby("label")[mfcc_cols].mean().reindex(LABELS)
    plt.figure(figsize=(10, 5))
    sns.heatmap(mfcc_mean, cmap="viridis")
    plt.title("Mean MFCC Features by Emotion")
    plt.tight_layout()
    plt.savefig(output_dir / "mfcc_mean_heatmap.png", dpi=180)
    plt.close()


def plot_mel_examples(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for ax, label in zip(axes, LABELS):
        subset = df[df["label"] == label]
        feature_path = ""
        for candidate in subset["audio_feature_path"].dropna():
            if Path(candidate).exists():
                feature_path = candidate
                break
        if not feature_path:
            ax.axis("off")
            continue
        log_mel = np.load(feature_path)["log_mel"]
        ax.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel bin")
    plt.tight_layout()
    plt.savefig(output_dir / "log_mel_examples.png", dpi=180)
    plt.close()


def first_face_path(face_dir: str) -> Path | None:
    if not isinstance(face_dir, str) or not Path(face_dir).exists():
        return None
    paths = sorted(Path(face_dir).glob("*.jpg"))
    return paths[0] if paths else None


def plot_face_grid(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for ax, label in zip(axes, LABELS):
        face_path = None
        for row in df[df["label"] == label].itertuples(index=False):
            face_path = first_face_path(getattr(row, "face_dir", ""))
            if face_path is not None:
                break
        if face_path is None:
            ax.axis("off")
            continue
        image = Image.open(face_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "face_examples_grid.png", dpi=180)
    plt.close()


def plot_face_pca(df: pd.DataFrame, output_dir: Path, max_samples: int = 400) -> None:
    features = []
    labels = []
    for row in df.sample(frac=1.0, random_state=42).itertuples(index=False):
        face_path = first_face_path(getattr(row, "face_dir", ""))
        if face_path is None:
            continue
        image = Image.open(face_path).convert("L").resize((64, 64))
        features.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
        labels.append(row.label)
        if len(features) >= max_samples:
            break
    if len(features) < 3:
        return

    feature_matrix = np.stack(features).astype(np.float64)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)
    feature_matrix -= feature_matrix.mean(axis=0, keepdims=True)

    variances = feature_matrix.var(axis=0)
    feature_matrix = feature_matrix[:, variances > 1e-12]
    if feature_matrix.shape[1] < 2:
        return

    xy = PCA(n_components=2, svd_solver="full").fit_transform(feature_matrix)
    plot_df = pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "label": labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="label", hue_order=LABELS, s=35)
    plt.title("Face Frame PCA by Emotion")
    plt.tight_layout()
    plt.savefig(output_dir / "face_pca.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature analysis figures for the report.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--output_dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()

    cache_dir = args.output_dir / ".cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    global Image, PCA, np, pd, plt, sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from PIL import Image
    from sklearn.decomposition import PCA

    ensure_dirs(args.output_dir)
    df = pd.read_csv(args.metadata)
    audio_stats = load_audio_stats(df)
    if not audio_stats.empty:
        audio_stats.to_csv(args.output_dir / "audio_feature_stats.csv", index=False)
        plot_audio_stats(audio_stats, args.output_dir)
        plot_mel_examples(df, args.output_dir)

    plot_face_grid(df, args.output_dir)
    plot_face_pca(df, args.output_dir)
    print(f"Feature analysis outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
