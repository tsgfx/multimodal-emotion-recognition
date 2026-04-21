import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face frames from RAVDESS videos.")
    parser.add_argument("--metadata", type=Path, default=PROCESSED_ROOT / "metadata.csv")
    parser.add_argument("--output_dir", type=Path, default=FACE_FRAME_DIR)
    parser.add_argument("--frames_per_sample", type=int, default=FACE_FRAMES_PER_SAMPLE)
    args = parser.parse_args()

    ensure_dirs(args.output_dir)
    df = pd.read_csv(args.metadata)
    face_dirs = []
    frame_counts = []
    failures = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="faces"):
        out_dir = args.output_dir / row.sample_id
        try:
            count = extract_frames(Path(row.video_path), out_dir, args.frames_per_sample)
            face_dirs.append(str(out_dir))
            frame_counts.append(count)
            if count == 0:
                failures.append({"sample_id": row.sample_id, "error": "no frames saved"})
        except Exception as exc:
            face_dirs.append("")
            frame_counts.append(0)
            failures.append({"sample_id": row.sample_id, "error": str(exc)})

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
