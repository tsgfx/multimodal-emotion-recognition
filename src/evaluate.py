import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR, FACE_FRAMES_PER_SAMPLE, LABELS, METRIC_DIR, PROCESSED_ROOT, RANDOM_SEED
from dataset import MultimodalEmotionDataset
from models import FaceResNet, build_audio_model, infer_audio_model_config_from_checkpoint
from train_utils import (
    autocast_context,
    create_data_loader,
    detailed_metrics,
    load_checkpoint,
    load_checkpoint_payload,
    save_confusion_matrix_csv,
)
from utils import configure_torch_runtime, describe_torch_runtime, get_device, save_json


def tagged_path(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def resolve_audio_model_config(args, device: torch.device) -> dict:
    checkpoint = load_checkpoint_payload(args.audio_checkpoint, device)
    if args.audio_model == "auto":
        config = infer_audio_model_config_from_checkpoint(checkpoint)
        config["wav2vec2_pretrained"] = args.wav2vec2_pretrained
        return config
    return {
        "audio_model": args.audio_model,
        "audio_dropout": args.audio_dropout,
        "crnn_hidden_size": args.crnn_hidden_size,
        "crnn_num_layers": args.crnn_num_layers,
        "cnn_variant": "modern",
        "wav2vec2_pretrained": args.wav2vec2_pretrained,
    }


def global_weight_search(audio_probs: np.ndarray, face_probs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    candidates = np.arange(0.0, 1.01, 0.1)
    best_alpha = 0.5
    best_metrics = None
    best_f1 = -1.0
    for candidate in candidates:
        fused = candidate * audio_probs + (1.0 - candidate) * face_probs
        preds = fused.argmax(axis=1)
        metrics = detailed_metrics(labels, preds)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_alpha = float(candidate)
            best_metrics = metrics
    return best_alpha, best_metrics


def apply_classwise_fusion(audio_probs: np.ndarray, face_probs: np.ndarray, alpha_vector: np.ndarray) -> np.ndarray:
    return alpha_vector.reshape(1, -1) * audio_probs + (1.0 - alpha_vector.reshape(1, -1)) * face_probs


def classwise_weight_search(
    audio_probs: np.ndarray,
    face_probs: np.ndarray,
    labels: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, dict]:
    base_alpha, _ = global_weight_search(audio_probs, face_probs, labels)
    alpha_vector = np.full(audio_probs.shape[1], base_alpha, dtype=np.float32)
    candidate_values = np.arange(0.0, 1.01, 0.1)

    fused = apply_classwise_fusion(audio_probs, face_probs, alpha_vector)
    preds = fused.argmax(axis=1)
    best_metrics = detailed_metrics(labels, preds)
    best_f1 = best_metrics["macro_f1"]

    for _ in range(iterations):
        improved = False
        for class_index in range(audio_probs.shape[1]):
            best_local_alpha = float(alpha_vector[class_index])
            best_local_metrics = best_metrics
            best_local_f1 = best_f1
            for candidate in candidate_values:
                trial_alpha = alpha_vector.copy()
                trial_alpha[class_index] = candidate
                fused = apply_classwise_fusion(audio_probs, face_probs, trial_alpha)
                preds = fused.argmax(axis=1)
                metrics = detailed_metrics(labels, preds)
                if metrics["macro_f1"] > best_local_f1:
                    best_local_alpha = float(candidate)
                    best_local_metrics = metrics
                    best_local_f1 = metrics["macro_f1"]
            if best_local_f1 > best_f1:
                alpha_vector[class_index] = best_local_alpha
                best_metrics = best_local_metrics
                best_f1 = best_local_f1
                improved = True
        if not improved:
            break

    return alpha_vector, best_metrics


def apply_confidence_weighted_fusion(
    audio_probs: np.ndarray,
    face_probs: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    audio_conf = np.clip(audio_probs.max(axis=1), 1e-6, 1.0)
    face_conf = np.clip(face_probs.max(axis=1), 1e-6, 1.0)
    audio_weight = np.power(audio_conf, beta)
    face_weight = np.power(face_conf, beta)
    alpha = audio_weight / (audio_weight + face_weight)
    fused = alpha[:, None] * audio_probs + (1.0 - alpha[:, None]) * face_probs
    return fused, alpha


def confidence_weight_search(
    audio_probs: np.ndarray,
    face_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict]:
    candidate_betas = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    best_beta = 1.0
    best_metrics = None
    best_f1 = -1.0
    for beta in candidate_betas:
        fused, _ = apply_confidence_weighted_fusion(audio_probs, face_probs, beta)
        preds = fused.argmax(axis=1)
        metrics = detailed_metrics(labels, preds)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_beta = float(beta)
            best_metrics = metrics
    return best_beta, best_metrics


@torch.no_grad()
def predict(
    model,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probs = []
    labels = []
    sample_ids = []
    for batch in tqdm(loader, desc=f"predict-{mode}", leave=False):
        sample_ids.extend(batch["sample_id"])
        labels.extend(batch["label"].numpy().tolist())
        with autocast_context(device, use_amp):
            if mode == "audio":
                logits = model(batch["audio"].to(device, non_blocking=device.type == "cuda"))
            elif mode == "wav2vec2":
                logits = model(batch["wav2vec2"].to(device, non_blocking=device.type == "cuda"))
            elif mode == "raw_audio":
                logits = model(batch["raw_audio"].to(device, non_blocking=device.type == "cuda"))
            else:
                logits = model(batch["face"].to(device, non_blocking=device.type == "cuda"))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0), np.array(labels), sample_ids


def evaluate_split(
    args,
    split: str,
    alpha: float | None = None,
    alpha_vector: np.ndarray | None = None,
    beta: float | None = None,
) -> tuple[dict, float | None, np.ndarray | None, float | None]:
    device = get_device(args.device)
    configure_torch_runtime(device)
    use_amp = device.type == "cuda" and not args.no_amp
    audio_model_config = resolve_audio_model_config(args, device)
    dataset_mode = "wav2vec2_fusion" if audio_model_config["audio_model"] == "wav2vec2_classifier" else ("raw_audio_fusion" if audio_model_config["audio_model"] == "wav2vec2_finetune" else "fusion")
    ds = MultimodalEmotionDataset(args.metadata, split, dataset_mode, frames_per_sample=args.frames_per_sample)
    loader = create_data_loader(ds, args.batch_size, False, args.num_workers, device)

    audio_model = build_audio_model(
        audio_model=audio_model_config["audio_model"],
        num_classes=len(LABELS),
        dropout=audio_model_config["audio_dropout"],
        crnn_hidden_size=audio_model_config["crnn_hidden_size"],
        crnn_num_layers=audio_model_config["crnn_num_layers"],
        cnn_variant=audio_model_config["cnn_variant"],
        wav2vec2_pretrained=audio_model_config.get("wav2vec2_pretrained", "facebook/wav2vec2-base"),
    ).to(device)
    face_model = FaceResNet(num_classes=len(LABELS), pretrained=False).to(device)
    load_checkpoint(audio_model, args.audio_checkpoint, device)
    load_checkpoint(face_model, args.face_checkpoint, device)

    audio_mode = "wav2vec2" if audio_model_config["audio_model"] == "wav2vec2_classifier" else ("raw_audio" if audio_model_config["audio_model"] == "wav2vec2_finetune" else "audio")
    audio_probs, labels, sample_ids = predict(audio_model, loader, device, audio_mode, use_amp=use_amp)
    face_probs, face_labels, _ = predict(face_model, loader, device, "face", use_amp=use_amp)
    if not np.array_equal(labels, face_labels):
        raise RuntimeError("Audio and face prediction labels are not aligned.")

    if args.fusion_strategy == "weighted_average":
        if alpha is None:
            alpha, _ = global_weight_search(audio_probs, face_probs, labels)
        fused_probs = alpha * audio_probs + (1.0 - alpha) * face_probs
    elif args.fusion_strategy == "classwise_weighted_average":
        if alpha_vector is None:
            alpha_vector, _ = classwise_weight_search(audio_probs, face_probs, labels)
        fused_probs = apply_classwise_fusion(audio_probs, face_probs, alpha_vector)
    else:
        if beta is None:
            beta, _ = confidence_weight_search(audio_probs, face_probs, labels)
        fused_probs, _ = apply_confidence_weighted_fusion(audio_probs, face_probs, beta)

    fused_preds = fused_probs.argmax(axis=1)
    metrics = detailed_metrics(labels, fused_preds)

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": [LABELS[i] for i in labels],
            "prediction": [LABELS[i] for i in fused_preds],
        }
    )
    pred_df.to_csv(tagged_path(METRIC_DIR / f"{split}_fusion_predictions.csv", args.output_tag), index=False)
    save_confusion_matrix_csv(
        metrics["confusion_matrix"],
        tagged_path(METRIC_DIR / f"{split}_fusion_confusion_matrix.csv", args.output_tag),
    )
    return metrics, float(alpha) if alpha is not None else None, alpha_vector, beta


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate late fusion on validation/test splits.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--audio_checkpoint", type=Path, default=CHECKPOINT_DIR / "audio_cnn.pt")
    parser.add_argument("--audio_model", choices=["auto", "cnn", "crnn", "wav2vec2_classifier", "wav2vec2_finetune"], default="auto")
    parser.add_argument("--audio_dropout", type=float, default=0.3)
    parser.add_argument("--crnn_hidden_size", type=int, default=128)
    parser.add_argument("--crnn_num_layers", type=int, default=1)
    parser.add_argument("--face_checkpoint", type=Path, default=CHECKPOINT_DIR / "face_resnet18.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frames_per_sample", type=int, default=FACE_FRAMES_PER_SAMPLE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision inference.")
    parser.add_argument(
        "--fusion_strategy",
        choices=["weighted_average", "classwise_weighted_average", "confidence_weighted_average"],
        default="weighted_average",
    )
    parser.add_argument("--wav2vec2_pretrained", type=str, default="facebook/wav2vec2-base",
        help="Path to local wav2vec2 pretrained model or HuggingFace model name.")
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix for fusion metric filenames.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(describe_torch_runtime(device))
    print(f"num_workers={args.num_workers} pin_memory={device.type == 'cuda'} amp={device.type == 'cuda' and not args.no_amp}")
    audio_model_config = resolve_audio_model_config(args, device)
    print(
        f"audio_model={audio_model_config['audio_model']} "
        f"audio_dropout={audio_model_config['audio_dropout']} "
        f"crnn_hidden_size={audio_model_config['crnn_hidden_size'] if audio_model_config['audio_model'] == 'crnn' else 'n/a'} "
        f"crnn_num_layers={audio_model_config['crnn_num_layers'] if audio_model_config['audio_model'] == 'crnn' else 'n/a'} "
        f"cnn_variant={audio_model_config['cnn_variant'] if audio_model_config['audio_model'] == 'cnn' else 'n/a'}"
    )

    val_metrics, alpha, alpha_vector, beta = evaluate_split(
        args, "val", alpha=None, alpha_vector=None, beta=None
    )
    test_metrics, _, _, _ = evaluate_split(
        args, "test", alpha=alpha, alpha_vector=alpha_vector, beta=beta
    )

    summary = {
        "fusion": args.fusion_strategy,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_classification_report": test_metrics["classification_report"],
    }
    if args.fusion_strategy == "weighted_average":
        summary["alpha_audio"] = alpha
        summary["alpha_face"] = 1.0 - alpha
    elif args.fusion_strategy == "classwise_weighted_average":
        summary["alpha_audio_by_class"] = {
            label: float(value) for label, value in zip(LABELS, alpha_vector.tolist())
        }
        summary["alpha_face_by_class"] = {
            label: float(1.0 - value) for label, value in zip(LABELS, alpha_vector.tolist())
        }
    else:
        summary["confidence_beta"] = beta
    output_tag = args.output_tag.strip()
    save_json(summary, tagged_path(METRIC_DIR / "fusion_metrics.json", output_tag))
    print(summary)


if __name__ == "__main__":
    main()
