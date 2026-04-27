import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import AUDIO_DURATION, AUDIO_FEATURE_DIR, AUDIO_SAMPLE_RATE, N_MELS, N_MFCC, PROCESSED_ROOT
from utils import ensure_dirs


def load_audio(path: Path) -> np.ndarray:
    target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
    y, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True, duration=AUDIO_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def extract_features(audio_path: Path) -> dict:
    y = load_audio(audio_path)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=AUDIO_SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=AUDIO_SAMPLE_RATE, n_mfcc=N_MFCC).astype(np.float32)
    rms = librosa.feature.rms(y=y).astype(np.float32)
    zcr = librosa.feature.zero_crossing_rate(y).astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=y, sr=AUDIO_SAMPLE_RATE).astype(np.float32)
    return {
        "log_mel": log_mel,
        "mfcc": mfcc,
        "rms_mean": np.array([float(rms.mean())], dtype=np.float32),
        "zcr_mean": np.array([float(zcr.mean())], dtype=np.float32),
        "centroid_mean": np.array([float(centroid.mean())], dtype=np.float32),
    }


def process_row(row: dict, output_dir: Path, skip_existing: bool) -> dict:
    out_path = output_dir / f"{row['sample_id']}.npz"
    if skip_existing and out_path.exists():
        return {
            "sample_id": row["sample_id"],
            "audio_feature_path": str(out_path),
            "error": "",
        }

    features = extract_features(Path(row["audio_path"]))
    np.savez_compressed(out_path, **features)
    return {
        "sample_id": row["sample_id"],
        "audio_feature_path": str(out_path),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Log-Mel and analysis features from audio.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--output_dir", type=Path, default=AUDIO_FEATURE_DIR)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing .npz files when present.")
    args = parser.parse_args()

    ensure_dirs(args.output_dir)
    df = pd.read_csv(args.metadata)
    failures = []
    results = {}

    rows = df.to_dict("records")
    if args.workers <= 1:
        for row in tqdm(rows, total=len(rows), desc="audio"):
            try:
                result = process_row(row, args.output_dir, args.skip_existing)
            except Exception as exc:
                result = {"sample_id": row["sample_id"], "audio_feature_path": "", "error": str(exc)}
            results[result["sample_id"]] = result
            if result["error"]:
                failures.append({"sample_id": result["sample_id"], "error": result["error"]})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, row, args.output_dir, args.skip_existing): row["sample_id"]
                for row in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="audio"):
                sample_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"sample_id": sample_id, "audio_feature_path": "", "error": str(exc)}
                results[sample_id] = result
                if result["error"]:
                    failures.append({"sample_id": result["sample_id"], "error": result["error"]})

    feature_paths = [results.get(row.sample_id, {}).get("audio_feature_path", "") for row in df.itertuples(index=False)]
    df["audio_feature_path"] = feature_paths
    df.to_csv(args.metadata, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "audio_failures.csv", index=False)
        print(f"Completed with {len(failures)} audio failures. See {args.output_dir / 'audio_failures.csv'}")
    else:
        print(f"Extracted audio features for {len(df)} samples.")


if __name__ == "__main__":
    main()
