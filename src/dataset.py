from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import AUDIO_DURATION, AUDIO_SAMPLE_RATE, FACE_FRAMES_PER_SAMPLE, FACE_IMAGE_SIZE, LABEL_TO_ID, LABELS, RANDOM_SEED


def _valid_path(value) -> bool:
    return isinstance(value, str) and len(value) > 0 and Path(value).exists()


class MultimodalEmotionDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
        split: str,
        mode: str,
        frames_per_sample: int = FACE_FRAMES_PER_SAMPLE,
        audio_augment: bool = False,
        specaugment_time_masks: int = 0,
        specaugment_freq_masks: int = 0,
        specaugment_max_time_width: int = 16,
        specaugment_max_freq_width: int = 8,
    ) -> None:
        if mode not in {"audio", "face", "fusion", "wav2vec2", "wav2vec2_fusion", "raw_audio", "raw_audio_fusion"}:
            raise ValueError("mode must be one of: audio, face, fusion, wav2vec2, wav2vec2_fusion, raw_audio, raw_audio_fusion")
        self.mode = mode
        self.split = split
        self.frames_per_sample = frames_per_sample
        self.df = pd.read_csv(metadata_path)

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Filter to target label set (removes neutral when using 5-class config)
        self.df = self.df[self.df["label"].isin(LABELS)].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No samples found for split={split} in {metadata_path}")

        if mode in {"audio", "fusion"}:
            self.df = self.df[self.df["audio_feature_path"].map(_valid_path)]
        if mode == "wav2vec2":
            self.df = self.df[self.df["wav2vec2_embedding_path"].map(_valid_path)]
        if mode == "wav2vec2_fusion":
            self.df = self.df[self.df["wav2vec2_embedding_path"].map(_valid_path)]
        if mode in {"raw_audio", "raw_audio_fusion"}:
            self.df = self.df[self.df["audio_path"].map(_valid_path)]
        if mode in {"face", "fusion", "wav2vec2_fusion", "raw_audio_fusion"}:
            self.df = self.df[self.df["face_dir"].map(_valid_path)]
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No usable samples found for split={split}, mode={mode}")

        self.face_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.face_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.audio_augment = audio_augment and split == "train" and mode in {"audio", "fusion"}
        self.specaugment_time_masks = max(0, int(specaugment_time_masks))
        self.specaugment_freq_masks = max(0, int(specaugment_freq_masks))
        self.specaugment_max_time_width = max(0, int(specaugment_max_time_width))
        self.specaugment_max_freq_width = max(0, int(specaugment_max_freq_width))
        self._wav2vec2_cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def get_label_ids(self) -> np.ndarray:
        return self.df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)

    def _apply_specaugment(self, audio: torch.Tensor) -> torch.Tensor:
        augmented = audio.clone()
        _, freq_bins, time_steps = augmented.shape

        for _ in range(self.specaugment_freq_masks):
            width = int(torch.randint(0, self.specaugment_max_freq_width + 1, (1,)).item())
            if width <= 0 or width >= freq_bins:
                continue
            start = int(torch.randint(0, freq_bins - width + 1, (1,)).item())
            augmented[:, start : start + width, :] = 0.0

        for _ in range(self.specaugment_time_masks):
            width = int(torch.randint(0, self.specaugment_max_time_width + 1, (1,)).item())
            if width <= 0 or width >= time_steps:
                continue
            start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
            augmented[:, :, start : start + width] = 0.0

        return augmented

    def _load_audio(self, path: str) -> torch.Tensor:
        data = np.load(path)
        log_mel = data["log_mel"].astype(np.float32)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        audio = torch.from_numpy(log_mel).unsqueeze(0)
        if self.audio_augment:
            audio = self._apply_specaugment(audio)
        return audio

    def _load_wav2vec2_embedding(self, path: str) -> torch.Tensor:
        if path in self._wav2vec2_cache:
            return self._wav2vec2_cache[path]
        emb = np.load(path)
        tensor = torch.from_numpy(emb.astype(np.float32))
        self._wav2vec2_cache[path] = tensor
        return tensor

    def _load_raw_audio(self, path: str) -> torch.Tensor:
        try:
            import librosa
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Loading raw audio requires librosa. Install with `pip install librosa`.")
        y, _ = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True, duration=AUDIO_DURATION)
        target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return torch.from_numpy(y.astype(np.float32))

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
        if self.mode in {"wav2vec2", "wav2vec2_fusion"}:
            item["wav2vec2"] = self._load_wav2vec2_embedding(row["wav2vec2_embedding_path"])
        if self.mode in {"raw_audio", "raw_audio_fusion"}:
            item["raw_audio"] = self._load_raw_audio(row["audio_path"])
        if self.mode in {"face", "fusion", "wav2vec2_fusion", "raw_audio_fusion"}:
            item["face"] = self._load_faces(row["face_dir"])
        return item
