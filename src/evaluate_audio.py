import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from config import CHECKPOINT_DIR, LABELS, METRIC_DIR, PROCESSED_ROOT, RANDOM_SEED
from dataset import MultimodalEmotionDataset
from models import build_audio_model, infer_audio_model_config_from_checkpoint
from train_utils import (
    autocast_context,
    create_data_loader,
    detailed_metrics,
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
        inferred = infer_audio_model_config_from_checkpoint(checkpoint)
        inferred["checkpoint"] = checkpoint
        return inferred
    return {
        "audio_model": args.audio_model,
        "audio_dropout": args.audio_dropout,
        "crnn_hidden_size": args.crnn_hidden_size,
        "crnn_num_layers": args.crnn_num_layers,
        "cnn_variant": "modern",
        "checkpoint": checkpoint,
    }


def metric_base_path(split: str, audio_model: str) -> Path:
    if audio_model == "cnn":
        return METRIC_DIR / f"audio_{split}_metrics.json"
    if audio_model == "wav2vec2_classifier":
        return METRIC_DIR / f"audio_{split}_metrics.wav2vec2_classifier.json"
    if audio_model == "wav2vec2_finetune":
        return METRIC_DIR / f"audio_{split}_metrics.wav2vec2_finetune.json"
    return METRIC_DIR / f"audio_{split}_metrics.{audio_model}.json"


def prediction_base_path(split: str, audio_model: str) -> Path:
    if audio_model == "cnn":
        return METRIC_DIR / f"audio_{split}_predictions.csv"
    if audio_model == "wav2vec2_classifier":
        return METRIC_DIR / f"audio_{split}_predictions.wav2vec2_classifier.csv"
    if audio_model == "wav2vec2_finetune":
        return METRIC_DIR / f"audio_{split}_predictions.wav2vec2_finetune.csv"
    return METRIC_DIR / f"audio_{split}_predictions.{audio_model}.csv"


def confusion_base_path(split: str, audio_model: str) -> Path:
    if audio_model == "cnn":
        return METRIC_DIR / f"audio_{split}_confusion_matrix.csv"
    if audio_model == "wav2vec2_classifier":
        return METRIC_DIR / f"audio_{split}_confusion_matrix.wav2vec2_classifier.csv"
    if audio_model == "wav2vec2_finetune":
        return METRIC_DIR / f"audio_{split}_confusion_matrix.wav2vec2_finetune.csv"
    return METRIC_DIR / f"audio_{split}_confusion_matrix.{audio_model}.csv"


@torch.no_grad()
def evaluate_split(args, split: str, model_config: dict) -> dict:
    device = get_device(args.device)
    configure_torch_runtime(device)
    use_amp = device.type == "cuda" and not args.no_amp
    dataset_mode = "wav2vec2" if model_config["audio_model"] == "wav2vec2_classifier" else ("raw_audio" if model_config["audio_model"] == "wav2vec2_finetune" else "audio")
    ds = MultimodalEmotionDataset(args.metadata, split, dataset_mode)
    loader = create_data_loader(ds, args.batch_size, False, args.num_workers, device)

    model = build_audio_model(
        audio_model=model_config["audio_model"],
        num_classes=len(LABELS),
        dropout=model_config["audio_dropout"],
        crnn_hidden_size=model_config["crnn_hidden_size"],
        crnn_num_layers=model_config["crnn_num_layers"],
        cnn_variant=model_config["cnn_variant"],
    ).to(device)
    model.load_state_dict(model_config["checkpoint"]["model_state"])
    model.eval()

    predictions = []
    labels = []
    sample_ids = []
    for batch in tqdm(loader, desc=f"predict-audio-{split}", leave=False):
        sample_ids.extend(batch["sample_id"])
        labels.extend(batch["label"].numpy().tolist())
        with autocast_context(device, use_amp):
            if model_config["audio_model"] == "wav2vec2_classifier":
                logits = model(batch["wav2vec2"].to(device, non_blocking=device.type == "cuda"))
            elif model_config["audio_model"] == "wav2vec2_finetune":
                logits = model(batch["raw_audio"].to(device, non_blocking=device.type == "cuda"))
            else:
                logits = model(batch["audio"].to(device, non_blocking=device.type == "cuda"))
        predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    metrics = detailed_metrics(labels, predictions)
    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": [LABELS[i] for i in labels],
            "prediction": [LABELS[i] for i in predictions],
        }
    )
    pred_df.to_csv(tagged_path(prediction_base_path(split, model_config["audio_model"]), args.output_tag), index=False)
    save_confusion_matrix_csv(
        metrics["confusion_matrix"],
        tagged_path(confusion_base_path(split, model_config["audio_model"]), args.output_tag),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio-only checkpoints on validation/test splits.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--audio_checkpoint", type=Path, default=CHECKPOINT_DIR / "audio_cnn.pt")
    parser.add_argument("--audio_model", choices=["auto", "cnn", "crnn", "wav2vec2_classifier", "wav2vec2_finetune"], default="auto")
    parser.add_argument("--audio_dropout", type=float, default=0.3)
    parser.add_argument("--crnn_hidden_size", type=int, default=128)
    parser.add_argument("--crnn_num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision inference.")
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix for metric filenames.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(describe_torch_runtime(device))
    print(f"num_workers={args.num_workers} pin_memory={device.type == 'cuda'} amp={device.type == 'cuda' and not args.no_amp}")
    model_config = resolve_audio_model_config(args, device)
    print(
        f"audio_model={model_config['audio_model']} "
        f"audio_dropout={model_config['audio_dropout']} "
        f"crnn_hidden_size={model_config['crnn_hidden_size'] if model_config['audio_model'] == 'crnn' else 'n/a'} "
        f"crnn_num_layers={model_config['crnn_num_layers'] if model_config['audio_model'] == 'crnn' else 'n/a'} "
        f"cnn_variant={model_config['cnn_variant'] if model_config['audio_model'] == 'cnn' else 'n/a'} "
        f"wav2vec2_classifier={model_config['audio_model'] == 'wav2vec2_classifier'}"
    )

    val_metrics = evaluate_split(args, "val", model_config)
    test_metrics = evaluate_split(args, "test", model_config)

    output_tag = args.output_tag.strip()
    save_json(val_metrics, tagged_path(metric_base_path("val", model_config["audio_model"]), output_tag))
    save_json(test_metrics, tagged_path(metric_base_path("test", model_config["audio_model"]), output_tag))
    print({"val": val_metrics, "test": test_metrics})


if __name__ == "__main__":
    main()
