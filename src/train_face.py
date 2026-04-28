import argparse
from pathlib import Path

import torch
from torch import nn

from config import CHECKPOINT_DIR, FACE_FRAMES_PER_SAMPLE, LABELS, METRIC_DIR, PROCESSED_ROOT, RANDOM_SEED
from dataset import MultimodalEmotionDataset
from models import FaceResNet
from train_utils import create_data_loader, make_grad_scaler, run_epoch, save_checkpoint
from utils import configure_torch_runtime, describe_torch_runtime, get_device, save_json, set_seed


def tagged_path(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train face-frame emotion classifier.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--face_augment", action="store_true", help="Enable face augmentation (hflip, brightness, contrast).")
    parser.add_argument("--face_dropout", type=float, default=0.0, help="Dropout rate for face model.")
    parser.add_argument("--frames_per_sample", type=int, default=FACE_FRAMES_PER_SAMPLE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision training.")
    parser.add_argument(
        "--output_tag",
        type=str,
        default="",
        help="Optional suffix for checkpoint and metric filenames.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=8,
        help="Stop training if monitored metric doesn't improve for this many consecutive epochs.",
    )
    parser.add_argument(
        "--early_stopping_monitor",
        choices=["val_loss", "val_macro_f1"],
        default="val_loss",
        help="Metric to monitor for early stopping. val_loss=lower is better; val_macro_f1=higher is better.",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device(args.device)
    configure_torch_runtime(device)
    use_amp = device.type == "cuda" and not args.no_amp
    print(describe_torch_runtime(device))
    print(f"num_workers={args.num_workers} pin_memory={device.type == 'cuda'} amp={use_amp}")

    train_ds = MultimodalEmotionDataset(args.metadata, "train", "face", frames_per_sample=args.frames_per_sample, face_augment=args.face_augment)
    val_ds = MultimodalEmotionDataset(args.metadata, "val", "face", frames_per_sample=args.frames_per_sample)
    train_loader = create_data_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = create_data_loader(val_ds, args.batch_size, False, args.num_workers, device)

    model = FaceResNet(num_classes=len(LABELS), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = make_grad_scaler(use_amp)

    best_f1 = -1.0
    best_metrics = {}
    checkpoint_path = tagged_path(CHECKPOINT_DIR / "face_resnet18.pt", args.output_tag.strip())
    metric_path = tagged_path(METRIC_DIR / "face_val_metrics.json", args.output_tag.strip())

    epochs_without_improvement = 0
    best_monitored = float("inf") if args.early_stopping_monitor == "val_loss" else -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, "face", use_amp=use_amp, scaler=scaler
        )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device, "face", use_amp=use_amp)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if args.early_stopping_monitor == "val_loss":
            current_monitored = val_loss
        else:
            current_monitored = val_metrics["macro_f1"]

        if args.early_stopping_monitor == "val_loss":
            improved = current_monitored < best_monitored
        else:
            improved = current_monitored > best_monitored

        if improved:
            best_monitored = current_monitored
            epochs_without_improvement = 0
            best_f1 = val_metrics["macro_f1"]
            best_metrics = {"epoch": epoch, "val_loss": val_loss, **val_metrics}
            save_checkpoint(model, checkpoint_path, best_metrics)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {epochs_without_improvement} epochs)")
                break

    save_json(best_metrics, metric_path)
    print(f"Saved best face model to {checkpoint_path}")
    print(f"Saved validation metrics to {metric_path}")


if __name__ == "__main__":
    main()
