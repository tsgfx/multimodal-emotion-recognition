import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CHECKPOINT_DIR, FACE_FRAMES_PER_SAMPLE, LABELS, METRIC_DIR, PROCESSED_ROOT
from dataset import MultimodalEmotionDataset
from models import FaceResNet, build_audio_model
from evaluate import predict, resolve_audio_model_config
from train_utils import create_data_loader, detailed_metrics, load_checkpoint
from utils import configure_torch_runtime, describe_torch_runtime, get_device, save_json


def tagged_path(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


@torch.no_grad()
def analyze_split(args, split: str) -> tuple[dict, pd.DataFrame]:
    device = get_device(args.device)
    configure_torch_runtime(device)
    audio_model_config = resolve_audio_model_config(args, device)
    ds = MultimodalEmotionDataset(args.metadata, split, "fusion", frames_per_sample=args.frames_per_sample)
    loader = create_data_loader(ds, args.batch_size, False, args.num_workers, device)

    audio_model = build_audio_model(
        audio_model=audio_model_config["audio_model"],
        num_classes=len(LABELS),
        dropout=audio_model_config["audio_dropout"],
        crnn_hidden_size=audio_model_config["crnn_hidden_size"],
        crnn_num_layers=audio_model_config["crnn_num_layers"],
        cnn_variant=audio_model_config["cnn_variant"],
    ).to(device)
    face_model = FaceResNet(num_classes=len(LABELS), pretrained=False).to(device)
    load_checkpoint(audio_model, args.audio_checkpoint, device)
    load_checkpoint(face_model, args.face_checkpoint, device)

    audio_probs, labels, sample_ids = predict(audio_model, loader, device, "audio", use_amp=False)
    face_probs, face_labels, _ = predict(face_model, loader, device, "face", use_amp=False)
    if not np.array_equal(labels, face_labels):
        raise RuntimeError("Audio and face prediction labels are not aligned.")

    audio_preds = audio_probs.argmax(axis=1)
    face_preds = face_probs.argmax(axis=1)

    audio_correct = audio_preds == labels
    face_correct = face_preds == labels

    sample_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": [LABELS[i] for i in labels],
            "audio_prediction": [LABELS[i] for i in audio_preds],
            "face_prediction": [LABELS[i] for i in face_preds],
            "audio_correct": audio_correct,
            "face_correct": face_correct,
            "both_correct": audio_correct & face_correct,
            "both_wrong": (~audio_correct) & (~face_correct),
            "audio_only_correct": audio_correct & (~face_correct),
            "face_only_correct": (~audio_correct) & face_correct,
        }
    )

    oracle_preds = []
    for label, audio_pred, face_pred, a_ok, f_ok in zip(labels, audio_preds, face_preds, audio_correct, face_correct):
        if a_ok and not f_ok:
            oracle_preds.append(int(audio_pred))
        elif f_ok and not a_ok:
            oracle_preds.append(int(face_pred))
        elif a_ok and f_ok:
            oracle_preds.append(int(audio_pred))
        else:
            oracle_preds.append(int(face_pred))
    oracle_metrics = detailed_metrics(labels, oracle_preds)
    audio_metrics = detailed_metrics(labels, audio_preds)
    face_metrics = detailed_metrics(labels, face_preds)

    per_class_rows = []
    for label_name in LABELS:
        subset = sample_df[sample_df["label"] == label_name]
        per_class_rows.append(
            {
                "label": label_name,
                "support": int(len(subset)),
                "both_correct": int(subset["both_correct"].sum()),
                "both_wrong": int(subset["both_wrong"].sum()),
                "audio_only_correct": int(subset["audio_only_correct"].sum()),
                "face_only_correct": int(subset["face_only_correct"].sum()),
            }
        )

    summary = {
        "split": split,
        "audio_model": audio_model_config["audio_model"],
        "audio_accuracy": audio_metrics["accuracy"],
        "audio_macro_f1": audio_metrics["macro_f1"],
        "face_accuracy": face_metrics["accuracy"],
        "face_macro_f1": face_metrics["macro_f1"],
        "oracle_accuracy": oracle_metrics["accuracy"],
        "oracle_macro_f1": oracle_metrics["macro_f1"],
        "both_correct": int(sample_df["both_correct"].sum()),
        "both_wrong": int(sample_df["both_wrong"].sum()),
        "audio_only_correct": int(sample_df["audio_only_correct"].sum()),
        "face_only_correct": int(sample_df["face_only_correct"].sum()),
        "jaccard_error_overlap": float(
            sample_df["both_wrong"].sum()
            / max(1, (sample_df["both_wrong"] | sample_df["audio_only_correct"] | sample_df["face_only_correct"]).sum())
        ),
        "per_class_overlap": per_class_rows,
        "audio_classification_report": audio_metrics["classification_report"],
        "face_classification_report": face_metrics["classification_report"],
    }
    return summary, sample_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sample-level error overlap between audio and face models.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--audio_checkpoint", type=Path, default=CHECKPOINT_DIR / "audio_cnn.pt")
    parser.add_argument("--audio_model", choices=["auto", "cnn", "crnn"], default="auto")
    parser.add_argument("--audio_dropout", type=float, default=0.3)
    parser.add_argument("--crnn_hidden_size", type=int, default=128)
    parser.add_argument("--crnn_num_layers", type=int, default=1)
    parser.add_argument("--face_checkpoint", type=Path, default=CHECKPOINT_DIR / "face_resnet18.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frames_per_sample", type=int, default=FACE_FRAMES_PER_SAMPLE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix for overlap analysis files.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(describe_torch_runtime(device))

    for split in ["val", "test"]:
        summary, sample_df = analyze_split(args, split)
        save_json(summary, tagged_path(METRIC_DIR / f"{split}_fusion_overlap.json", args.output_tag))
        sample_df.to_csv(tagged_path(METRIC_DIR / f"{split}_fusion_overlap_samples.csv", args.output_tag), index=False)
        print(summary)


if __name__ == "__main__":
    main()
