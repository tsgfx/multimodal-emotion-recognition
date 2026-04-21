import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, LABELS, METRIC_DIR, PROCESSED_ROOT, RANDOM_SEED
from dataset import MultimodalEmotionDataset
from models import AudioCNN
from train_utils import run_epoch, save_checkpoint
from utils import get_device, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio-only emotion classifier.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device()
    train_ds = MultimodalEmotionDataset(args.metadata, "train", "audio")
    val_ds = MultimodalEmotionDataset(args.metadata, "val", "audio")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = AudioCNN(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_metrics = {}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, "audio")
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, None, device, "audio")
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = {"epoch": epoch, "val_loss": val_loss, **val_metrics}
            save_checkpoint(model, CHECKPOINT_DIR / "audio_cnn.pt", best_metrics)

    save_json(best_metrics, METRIC_DIR / "audio_val_metrics.json")
    print(f"Saved best audio model to {CHECKPOINT_DIR / 'audio_cnn.pt'}")


if __name__ == "__main__":
    main()
