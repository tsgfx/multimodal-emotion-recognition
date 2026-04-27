import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from config import AUDIO_DURATION, AUDIO_SAMPLE_RATE, PROCESSED_ROOT, WAV2VEC2_EMBED_DIR
from utils import ensure_dirs


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=target_sr, mono=True, duration=AUDIO_DURATION)
    target_len = int(target_sr * AUDIO_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def extract_embedding(audio_path: Path, model: Wav2Vec2Model, feature_extractor: Wav2Vec2FeatureExtractor) -> np.ndarray:
    y = load_audio(audio_path)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
    embedding = hidden.mean(dim=1).squeeze(0).numpy()
    return embedding.astype(np.float32)


def process_row(
    row: dict,
    output_dir: Path,
    model: Wav2Vec2Model,
    feature_extractor: Wav2Vec2FeatureExtractor,
    skip_existing: bool,
) -> dict:
    out_path = output_dir / f"{row['sample_id']}.npy"
    if skip_existing and out_path.exists():
        return {"sample_id": row["sample_id"], "embedding_path": str(out_path), "error": ""}
    try:
        emb = extract_embedding(Path(row["audio_path"]), model, feature_extractor)
        np.save(out_path, emb)
        return {"sample_id": row["sample_id"], "embedding_path": str(out_path), "error": ""}
    except Exception as exc:
        return {"sample_id": row["sample_id"], "embedding_path": "", "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract wav2vec2 embeddings from raw audio files.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--output_dir", type=Path, default=WAV2VEC2_EMBED_DIR)
    parser.add_argument("--pretrained_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    ensure_dirs(args.output_dir)

    print(f"Loading {args.pretrained_model}...")
    model = Wav2Vec2Model.from_pretrained(args.pretrained_model)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.pretrained_model)
    print("Model loaded.")

    df = pd.read_csv(args.metadata)
    rows = df.to_dict("records")
    results = {}
    failures = []

    if args.workers <= 1:
        for row in tqdm(rows, desc="wav2vec2"):
            result = process_row(row, args.output_dir, model, feature_extractor, args.skip_existing)
            results[result["sample_id"]] = result
            if result["error"]:
                failures.append({"sample_id": result["sample_id"], "error": result["error"]})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, row, args.output_dir, model, feature_extractor, args.skip_existing): row["sample_id"]
                for row in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="wav2vec2"):
                sample_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"sample_id": sample_id, "embedding_path": "", "error": str(exc)}
                results[result["sample_id"]] = result
                if result["error"]:
                    failures.append({"sample_id": result["sample_id"], "error": result["error"]})

    embed_paths = [results.get(row["sample_id"], {}).get("embedding_path", "") for row in rows]
    df["wav2vec2_embedding_path"] = embed_paths
    df.to_csv(args.metadata, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "wav2vec2_failures.csv", index=False)
        print(f"Completed with {len(failures)} failures. See {args.output_dir / 'wav2vec2_failures.csv'}")
    else:
        print(f"Extracted wav2vec2 embeddings for {len(df)} samples.")


if __name__ == "__main__":
    main()
