import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_ROOT, RANDOM_SEED, TARGET_EMOTIONS
from utils import ensure_dirs


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}


def parse_ravdess_name(path: Path) -> dict | None:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    emotion_code = parts[2]
    if emotion_code not in TARGET_EMOTIONS:
        return None
    return {
        "modality_code": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": emotion_code,
        "label": TARGET_EMOTIONS[emotion_code],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor_id": int(parts[6]),
        "pair_key": "-".join(parts[1:]),
        "ravdess_stem": path.stem,
    }


def collect_files(data_root: Path) -> tuple[dict, dict]:
    audio_by_key: dict[str, Path] = {}
    video_by_key: dict[str, Path] = {}
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        parsed = parse_ravdess_name(path)
        if parsed is None:
            continue
        key = parsed["pair_key"]
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTS:
            current = video_by_key.get(key)
            if current is None or parsed["modality_code"] == "01":
                video_by_key[key] = path
        elif suffix in AUDIO_EXTS:
            current = audio_by_key.get(key)
            if current is None or parsed["modality_code"] == "03":
                audio_by_key[key] = path
    return audio_by_key, video_by_key


def assign_actor_split(actor_id: int) -> str:
    if actor_id <= 18:
        return "train"
    if actor_id <= 21:
        return "val"
    return "test"


def assign_random_split(df: pd.DataFrame) -> pd.Series:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )
    split = pd.Series(index=df.index, dtype="object")
    split.loc[train_df.index] = "train"
    split.loc[val_df.index] = "val"
    split.loc[test_df.index] = "test"
    return split


def build_metadata(data_root: Path, split_strategy: str) -> pd.DataFrame:
    audio_by_key, video_by_key = collect_files(data_root)
    rows = []

    for key in sorted(video_by_key):
        parsed = parse_ravdess_name(video_by_key[key])
        if parsed is None:
            continue
        audio_path = audio_by_key.get(key, video_by_key[key])
        sample_id = key.replace("-", "_")
        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "video_path": str(video_by_key[key]),
                "face_dir": "",
                "audio_feature_path": "",
                "label": parsed["label"],
                "actor_id": parsed["actor_id"],
                "split": "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No usable RAVDESS samples found under {data_root}. "
            "Place RAVDESS audio-video files in data/raw/ravdess first."
        )

    if split_strategy == "actor":
        df["split"] = df["actor_id"].map(assign_actor_split)
    else:
        df["split"] = assign_random_split(df)

    return df.sort_values(["split", "label", "sample_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAVDESS metadata for six-class emotion recognition.")
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/ravdess"))
    parser.add_argument("--output", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--split_strategy", choices=["actor", "random"], default="actor")
    args = parser.parse_args()

    ensure_dirs(args.output.parent)
    df = build_metadata(args.data_root, args.split_strategy)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} samples to {args.output}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
