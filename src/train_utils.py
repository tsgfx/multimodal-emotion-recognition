from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import LABELS


def batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def run_epoch(model, loader: DataLoader, criterion, optimizer, device: torch.device, mode: str) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="train" if is_train else "eval", leave=False):
        batch = batch_to_device(batch, device)
        labels = batch["label"]
        with torch.set_grad_enabled(is_train):
            if mode == "audio":
                logits = model(batch["audio"])
            else:
                logits = model(batch["face"])
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return float(np.mean(losses)), metrics


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def detailed_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


def save_confusion_matrix_csv(matrix: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(path)


def save_checkpoint(model: nn.Module, path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model
