import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR, LABELS, METRIC_DIR, PROCESSED_ROOT
from dataset import MultimodalEmotionDataset
from models import AudioCNN, FaceResNet
from train_utils import detailed_metrics, load_checkpoint, save_confusion_matrix_csv
from utils import get_device, save_json


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probs = []
    labels = []
    sample_ids = []
    for batch in tqdm(loader, desc=f"predict-{mode}", leave=False):
        sample_ids.extend(batch["sample_id"])
        labels.extend(batch["label"].numpy().tolist())
        if mode == "audio":
            logits = model(batch["audio"].to(device))
        else:
            logits = model(batch["face"].to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0), np.array(labels), sample_ids


def evaluate_split(args, split: str, alpha: float | None = None) -> tuple[dict, float]:
    device = get_device()
    ds = MultimodalEmotionDataset(args.metadata, split, "fusion")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    audio_model = AudioCNN(num_classes=len(LABELS)).to(device)
    face_model = FaceResNet(num_classes=len(LABELS), pretrained=False).to(device)
    load_checkpoint(audio_model, args.audio_checkpoint, device)
    load_checkpoint(face_model, args.face_checkpoint, device)

    audio_probs, labels, sample_ids = predict(audio_model, loader, device, "audio")
    face_probs, face_labels, _ = predict(face_model, loader, device, "face")
    if not np.array_equal(labels, face_labels):
        raise RuntimeError("Audio and face prediction labels are not aligned.")

    if alpha is None:
        candidates = np.arange(0.1, 1.0, 0.1)
        best_alpha = 0.5
        best_f1 = -1.0
        for candidate in candidates:
            fused = candidate * audio_probs + (1.0 - candidate) * face_probs
            preds = fused.argmax(axis=1)
            metrics = detailed_metrics(labels, preds)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_alpha = float(candidate)
        alpha = best_alpha

    fused_probs = alpha * audio_probs + (1.0 - alpha) * face_probs
    fused_preds = fused_probs.argmax(axis=1)
    metrics = detailed_metrics(labels, fused_preds)

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": [LABELS[i] for i in labels],
            "prediction": [LABELS[i] for i in fused_preds],
        }
    )
    pred_df.to_csv(METRIC_DIR / f"{split}_fusion_predictions.csv", index=False)
    save_confusion_matrix_csv(metrics["confusion_matrix"], METRIC_DIR / f"{split}_fusion_confusion_matrix.csv")
    return metrics, float(alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate late fusion on validation/test splits.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--audio_checkpoint", type=Path, default=CHECKPOINT_DIR / "audio_cnn.pt")
    parser.add_argument("--face_checkpoint", type=Path, default=CHECKPOINT_DIR / "face_resnet18.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    val_metrics, alpha = evaluate_split(args, "val", alpha=None)
    test_metrics, _ = evaluate_split(args, "test", alpha=alpha)

    summary = {
        "fusion": "weighted_average",
        "alpha_audio": alpha,
        "alpha_face": 1.0 - alpha,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_classification_report": test_metrics["classification_report"],
    }
    save_json(summary, METRIC_DIR / "fusion_metrics.json")
    print(summary)


if __name__ == "__main__":
    main()
