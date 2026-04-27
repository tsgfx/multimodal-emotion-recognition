import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from config import FACE_FRAME_DIR, FACE_FRAMES_PER_SAMPLE, FACE_IMAGE_SIZE, PROCESSED_ROOT
from utils import ensure_dirs


def center_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def crop_face(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return center_crop(frame)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(0.2 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    return frame[y0:y1, x0:x1]


def extract_frames(video_path: Path, out_dir: Path, frames_per_sample: int) -> int:
    ensure_dirs(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_indices = list(range(frames_per_sample))
    else:
        frame_indices = [int(i * max(frame_count - 1, 1) / max(frames_per_sample - 1, 1)) for i in range(frames_per_sample)]

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))

    saved = 0
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        face = crop_face(frame, detector)
        face = cv2.resize(face, (FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"frame_{idx:02d}.jpg"
        cv2.imwrite(str(out_path), face)
        saved += 1

    cap.release()
    return saved


def existing_frame_count(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("frame_*.jpg")))


def process_row(row: dict, output_dir: Path, frames_per_sample: int, skip_existing: bool) -> dict:
    out_dir = output_dir / row["sample_id"]
    count = existing_frame_count(out_dir)
    if skip_existing and count >= frames_per_sample:
        return {
            "sample_id": row["sample_id"],
            "face_dir": str(out_dir),
            "num_face_frames": count,
            "error": "",
        }

    count = extract_frames(Path(row["video_path"]), out_dir, frames_per_sample)
    error = "" if count > 0 else "no frames saved"
    return {
        "sample_id": row["sample_id"],
        "face_dir": str(out_dir),
        "num_face_frames": count,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face frames from RAVDESS videos.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--output_dir", type=Path, default=FACE_FRAME_DIR)
    parser.add_argument("--frames_per_sample", type=int, default=FACE_FRAMES_PER_SAMPLE)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing frame directories when complete.")
    args = parser.parse_args()

    ensure_dirs(args.output_dir)
    df = pd.read_csv(args.metadata)
    failures = []
    results = {}

    rows = df.to_dict("records")
    if args.workers <= 1:
        for row in tqdm(rows, total=len(rows), desc="faces"):
            try:
                result = process_row(row, args.output_dir, args.frames_per_sample, args.skip_existing)
            except Exception as exc:
                result = {"sample_id": row["sample_id"], "face_dir": "", "num_face_frames": 0, "error": str(exc)}
            results[result["sample_id"]] = result
            if result["error"]:
                failures.append({"sample_id": result["sample_id"], "error": result["error"]})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, row, args.output_dir, args.frames_per_sample, args.skip_existing): row["sample_id"]
                for row in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="faces"):
                sample_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"sample_id": sample_id, "face_dir": "", "num_face_frames": 0, "error": str(exc)}
                results[sample_id] = result
                if result["error"]:
                    failures.append({"sample_id": result["sample_id"], "error": result["error"]})

    face_dirs = [results.get(row.sample_id, {}).get("face_dir", "") for row in df.itertuples(index=False)]
    frame_counts = [results.get(row.sample_id, {}).get("num_face_frames", 0) for row in df.itertuples(index=False)]
    df["face_dir"] = face_dirs
    df["num_face_frames"] = frame_counts
    df.to_csv(args.metadata, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "face_failures.csv", index=False)
        print(f"Completed with {len(failures)} face extraction failures. See {args.output_dir / 'face_failures.csv'}")
    else:
        print(f"Extracted face frames for {len(df)} samples.")


if __name__ == "__main__":
    main()
