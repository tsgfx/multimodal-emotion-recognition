import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from config import CHECKPOINT_DIR, LABELS, METRIC_DIR, PROCESSED_ROOT, RANDOM_SEED
from dataset import MultimodalEmotionDataset
from models import build_audio_model
from train_utils import create_data_loader, make_grad_scaler, run_epoch, save_checkpoint
from utils import configure_torch_runtime, describe_torch_runtime, get_device, save_json, set_seed


def tagged_path(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def default_checkpoint_path(audio_model: str) -> Path:
    if audio_model == "cnn":
        return CHECKPOINT_DIR / "audio_cnn.pt"
    return CHECKPOINT_DIR / f"audio_{audio_model}.pt"


def default_metric_path(audio_model: str) -> Path:
    if audio_model == "cnn":
        return METRIC_DIR / "audio_val_metrics.json"
    if audio_model == "wav2vec2_classifier":
        return METRIC_DIR / "audio_val_metrics.wav2vec2_classifier.json"
    return METRIC_DIR / f"audio_val_metrics.{audio_model}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio-only emotion classifier.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--audio_model", choices=["cnn", "crnn", "wav2vec2_classifier", "wav2vec2_finetune"], default="cnn")
    parser.add_argument("--audio_dropout", type=float, default=0.3)
    parser.add_argument("--crnn_hidden_size", type=int, default=128)
    parser.add_argument("--crnn_num_layers", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision training.")
    parser.add_argument("--class_weight", action="store_true", help="Use inverse-frequency class weights in loss.")
    parser.add_argument("--weighted_sampler", action="store_true", help="Use inverse-frequency weighted sampling.")
    parser.add_argument("--specaugment", action="store_true", help="Enable SpecAugment on training Log-Mel inputs.")
    parser.add_argument("--specaugment_time_masks", type=int, default=2)
    parser.add_argument("--specaugment_freq_masks", type=int, default=2)
    parser.add_argument("--specaugment_max_time_width", type=int, default=16)
    parser.add_argument("--specaugment_max_freq_width", type=int, default=8)
    parser.add_argument(
        "--output_tag",
        type=str,
        default="",
        help="Optional suffix for checkpoint and metric filenames, e.g. audio_cnn.<tag>.pt.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device(args.device)
    configure_torch_runtime(device)
    use_amp = device.type == "cuda" and not args.no_amp
    print(describe_torch_runtime(device))
    print(f"num_workers={args.num_workers} pin_memory={device.type == 'cuda'} amp={use_amp}")
    if args.class_weight and args.weighted_sampler:
        print("warning: class_weight and weighted_sampler are both enabled; this may over-emphasize minority classes.")

    if args.audio_model == "wav2vec2_classifier":
        dataset_mode = "wav2vec2"
    elif args.audio_model == "wav2vec2_finetune":
        dataset_mode = "raw_audio"
    else:
        dataset_mode = "audio"

    train_ds = MultimodalEmotionDataset(
        args.metadata,
        "train",
        dataset_mode,
        audio_augment=args.specaugment,
        specaugment_time_masks=args.specaugment_time_masks,
        specaugment_freq_masks=args.specaugment_freq_masks,
        specaugment_max_time_width=args.specaugment_max_time_width,
        specaugment_max_freq_width=args.specaugment_max_freq_width,
    )
    val_ds = MultimodalEmotionDataset(
        args.metadata,
        "val",
        dataset_mode,
    )
    label_ids = train_ds.get_label_ids()
    class_counts = np.bincount(label_ids, minlength=len(LABELS))
    class_weights = len(label_ids) / (len(LABELS) * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    sample_weights = torch.tensor(class_weights[label_ids], dtype=torch.double)
    sampler = None
    if args.weighted_sampler:
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"class_counts={class_counts.tolist()}")
    if args.class_weight:
        print(f"class_weights={class_weights.round(4).tolist()}")
    print(
        f"weighted_sampler={args.weighted_sampler} "
        f"specaugment={args.specaugment} "
        f"time_masks={args.specaugment_time_masks if args.specaugment else 0} "
        f"freq_masks={args.specaugment_freq_masks if args.specaugment else 0}"
    )
    print(
        f"audio_model={args.audio_model} "
        f"audio_dropout={args.audio_dropout} "
        f"crnn_hidden_size={args.crnn_hidden_size if args.audio_model == 'crnn' else 'n/a'} "
        f"crnn_num_layers={args.crnn_num_layers if args.audio_model == 'crnn' else 'n/a'} "
        f"cnn_variant={'modern' if args.audio_model == 'cnn' else 'n/a'}"
    )

    train_loader = create_data_loader(train_ds, args.batch_size, True, args.num_workers, device, sampler=sampler)
    val_loader = create_data_loader(val_ds, args.batch_size, False, args.num_workers, device)

    model = build_audio_model(
        audio_model=args.audio_model,
        num_classes=len(LABELS),
        dropout=args.audio_dropout,
        crnn_hidden_size=args.crnn_hidden_size,
        crnn_num_layers=args.crnn_num_layers,
        cnn_variant="modern",
        wav2vec2_trainable=(args.audio_model == "wav2vec2_finetune"),
    ).to(device)

    if args.audio_model == "wav2vec2_finetune":
        optimizer = torch.optim.AdamW([
            {"params": model.wav2vec2.parameters(), "lr": 1e-5},
            {"params": model.pooler.parameters(), "lr": 1e-4},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor if args.class_weight else None)
    scaler = make_grad_scaler(use_amp)
    model_config = {
        "audio_model": args.audio_model,
        "audio_dropout": args.audio_dropout,
        "crnn_hidden_size": args.crnn_hidden_size,
        "crnn_num_layers": args.crnn_num_layers,
        "cnn_variant": "modern",
    }

    best_f1 = -1.0
    best_metrics = {}
    checkpoint_path = tagged_path(default_checkpoint_path(args.audio_model), args.output_tag.strip())
    metric_path = tagged_path(default_metric_path(args.audio_model), args.output_tag.strip())
    train_mode = "raw_audio" if args.audio_model == "wav2vec2_finetune" else ("wav2vec2" if args.audio_model == "wav2vec2_classifier" else "audio")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train_mode, use_amp=use_amp, scaler=scaler
        )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device, train_mode, use_amp=use_amp)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = {"epoch": epoch, "val_loss": val_loss, **val_metrics}
            save_checkpoint(model, checkpoint_path, best_metrics, model_config=model_config)

    save_json(best_metrics, metric_path)
    print(f"Saved best audio model to {checkpoint_path}")
    print(f"Saved validation metrics to {metric_path}")


if __name__ == "__main__":
    main()
