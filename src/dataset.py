from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import FACE_FRAMES_PER_SAMPLE, FACE_IMAGE_SIZE, LABEL_TO_ID


def _valid_path(value) -> bool:
    return isinstance(value, str) and len(value) > 0 and Path(value).exists()


class MultimodalEmotionDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
        split: str,
        mode: str,
        frames_per_sample: int = FACE_FRAMES_PER_SAMPLE,
    ) -> None:
        if mode not in {"audio", "face", "fusion"}:
            raise ValueError("mode must be one of: audio, face, fusion")
        self.mode = mode
        self.frames_per_sample = frames_per_sample
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No samples found for split={split} in {metadata_path}")

        if mode in {"audio", "fusion"}:
            self.df = self.df[self.df["audio_feature_path"].map(_valid_path)]
        if mode in {"face", "fusion"}:
            self.df = self.df[self.df["face_dir"].map(_valid_path)]
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No usable samples found for split={split}, mode={mode}")

        self.face_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.face_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> torch.Tensor:
        data = np.load(path)
        log_mel = data["log_mel"].astype(np.float32)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return torch.from_numpy(log_mel).unsqueeze(0)

    def _load_faces(self, face_dir: str) -> torch.Tensor:
        paths = sorted(Path(face_dir).glob("*.jpg"))
        if not paths:
            raise RuntimeError(f"No face frames found in {face_dir}")
        if len(paths) >= self.frames_per_sample:
            indices = np.linspace(0, len(paths) - 1, self.frames_per_sample).astype(int)
            selected = [paths[i] for i in indices]
        else:
            selected = paths + [paths[-1]] * (self.frames_per_sample - len(paths))

        frames = []
        for path in selected:
            image = Image.open(path).convert("RGB").resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            frames.append((tensor - self.face_mean) / self.face_std)
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        label = torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long)
        item = {
            "sample_id": row["sample_id"],
            "label": label,
        }
        if self.mode in {"audio", "fusion"}:
            item["audio"] = self._load_audio(row["audio_feature_path"])
        if self.mode in {"face", "fusion"}:
            item["face"] = self._load_faces(row["face_dir"])
        return item
