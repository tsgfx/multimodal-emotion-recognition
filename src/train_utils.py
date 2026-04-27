from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from config import LABELS


def create_data_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    sampler: Sampler | None = None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def batch_to_device(batch: dict, device: torch.device, non_blocking: bool = False) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


def run_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    mode: str,
    use_amp: bool = False,
    scaler=None,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="train" if is_train else "eval", leave=False):
        batch = batch_to_device(batch, device, non_blocking=device.type == "cuda")
        labels = batch["label"]
        with torch.set_grad_enabled(is_train):
            with autocast_context(device, use_amp):
                if mode == "audio":
                    logits = model(batch["audio"])
                elif mode == "wav2vec2":
                    logits = model(batch["wav2vec2"])
                elif mode == "raw_audio":
                    logits = model(batch["raw_audio"])
                else:
                    logits = model(batch["face"])
                loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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


def save_checkpoint(model: nn.Module, path: Path, metrics: dict, model_config: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metrics": metrics, "model_config": model_config or {}}, path)


def load_checkpoint_payload(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint_payload(path, device)
    model.load_state_dict(checkpoint["model_state"])
    return model
