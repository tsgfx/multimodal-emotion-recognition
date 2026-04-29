import torch
from pathlib import Path
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_pretrained_model_name_or_path(pretrained_model: str) -> str:
    """Resolve local Wav2Vec2 paths across machines before falling back to HF ids."""
    aliases = {
        "base": "facebook/wav2vec2-base",
        "wav2vec2-base": "facebook/wav2vec2-base",
        "facebook-base": "facebook/wav2vec2-base",
    }
    value = aliases.get(pretrained_model, pretrained_model)
    candidates = [Path(value)]

    raw_path = Path(pretrained_model)
    if raw_path.name:
        candidates.extend(
            [
                PROJECT_ROOT / raw_path.name,
                Path.cwd() / raw_path.name,
            ]
        )

    if pretrained_model == "facebook/wav2vec2-base":
        candidates.extend([PROJECT_ROOT / "wav2vec2-base", Path.cwd() / "wav2vec2-base"])

    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            return str(candidate)
    return value


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(audio))


class AudioCNNLegacy(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(audio))


class AudioCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            nn.MaxPool2d(kernel_size=(2, 2)),
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveAvgPool2d((8, None)),
        )
        self.sequence_encoder = nn.GRU(
            input_size=128 * 8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.features(audio)
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        sequence, _ = self.sequence_encoder(x)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (sequence * weights).sum(dim=1)
        return self.classifier(pooled)


def build_audio_model(
    audio_model: str,
    num_classes: int,
    dropout: float = 0.3,
    crnn_hidden_size: int = 128,
    crnn_num_layers: int = 1,
    cnn_variant: str = "modern",
    wav2vec2_hidden_size: int = 768,
    wav2vec2_trainable: bool = False,
    wav2vec2_pretrained: str = "facebook/wav2vec2-base",
) -> nn.Module:
    if audio_model == "cnn":
        if cnn_variant == "legacy":
            return AudioCNNLegacy(num_classes=num_classes, dropout=dropout)
        return AudioCNN(num_classes=num_classes, dropout=dropout)
    if audio_model == "crnn":
        return AudioCRNN(
            num_classes=num_classes,
            dropout=dropout,
            hidden_size=crnn_hidden_size,
            num_layers=crnn_num_layers,
        )
    if audio_model == "wav2vec2":
        return AudioWav2Vec2(num_classes=num_classes, dropout=dropout, trainable=False, pretrained_model=wav2vec2_pretrained)
    if audio_model == "wav2vec2_classifier":
        return AudioWav2Vec2Classifier(num_classes=num_classes, hidden_size=wav2vec2_hidden_size, dropout=dropout)
    if audio_model == "wav2vec2_finetune":
        return AudioWav2Vec2(num_classes=num_classes, dropout=dropout, trainable=True, pretrained_model=wav2vec2_pretrained)
    raise ValueError(f"Unsupported audio_model={audio_model}")


def infer_audio_model_config_from_checkpoint(checkpoint: dict) -> dict:
    saved = checkpoint.get("model_config", {})
    if saved:
        return {
            "audio_model": saved.get("audio_model", "cnn"),
            "audio_dropout": float(saved.get("audio_dropout", 0.3)),
            "crnn_hidden_size": int(saved.get("crnn_hidden_size", 128)),
            "crnn_num_layers": int(saved.get("crnn_num_layers", 1)),
            "cnn_variant": saved.get("cnn_variant", "modern"),
            "wav2vec2_pretrained": saved.get("wav2vec2_pretrained", "facebook/wav2vec2-base"),
        }

    state_dict = checkpoint.get("model_state", {})
    if any(key.startswith("wav2vec2.") for key in state_dict):
        saved_model_type = saved.get("audio_model", "wav2vec2")
        return {
            "audio_model": saved_model_type if saved_model_type in ("wav2vec2", "wav2vec2_finetune") else "wav2vec2",
            "audio_dropout": float(saved.get("audio_dropout", 0.3)),
            "crnn_hidden_size": 128,
            "crnn_num_layers": 1,
            "cnn_variant": "modern",
            "wav2vec2_pretrained": saved.get("wav2vec2_pretrained", "facebook/wav2vec2-base"),
        }
    if any(key.startswith("classifier.0.") for key in state_dict):
        return {
            "audio_model": "wav2vec2_classifier",
            "audio_dropout": 0.3,
            "crnn_hidden_size": 128,
            "crnn_num_layers": 1,
            "cnn_variant": "modern",
        }
    if any(key.startswith("sequence_encoder.") for key in state_dict):
        return {
            "audio_model": "crnn",
            "audio_dropout": 0.3,
            "crnn_hidden_size": 128,
            "crnn_num_layers": 1,
            "cnn_variant": "modern",
        }
    if any(key.startswith("features.0.weight") for key in state_dict):
        return {
            "audio_model": "cnn",
            "audio_dropout": 0.3,
            "crnn_hidden_size": 128,
            "crnn_num_layers": 1,
            "cnn_variant": "legacy",
        }
    return {
        "audio_model": "cnn",
        "audio_dropout": 0.3,
        "crnn_hidden_size": 128,
        "crnn_num_layers": 1,
        "cnn_variant": "modern",
    }


class AudioWav2Vec2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "facebook/wav2vec2-base",
        dropout: float = 0.3,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AudioWav2Vec2 requires transformers. Install with `pip install transformers>=4.30`."
            ) from exc

        self.trainable_encoder = trainable
        pretrained_model = resolve_pretrained_model_name_or_path(pretrained_model)
        local_path = Path(pretrained_model)
        if local_path.exists() and (local_path / "config.json").exists():
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(str(local_path), local_files_only=True)
            self.hidden_size = self.wav2vec2.config.hidden_size
            if trainable:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(local_path), local_files_only=True)
            else:
                self.feature_extractor = None
        else:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
            self.hidden_size = self.wav2vec2.config.hidden_size
            if trainable:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model)
            else:
                self.feature_extractor = None

        if not trainable:
            self.wav2vec2.eval()
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        else:
            for param in self.wav2vec2.parameters():
                param.requires_grad = True

        self.pooler = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.trainable_encoder:
            with torch.no_grad():
                outputs = self.wav2vec2(inputs_embeds=audio)
            hidden = outputs.last_hidden_state
        else:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() > 2:
                audio = audio.squeeze()
            inputs = self.feature_extractor(
                audio.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(audio.device)
            outputs = self.wav2vec2(input_values=input_values)
            hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.pooler(pooled)


class AudioWav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 768, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class FaceResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FaceResNet requires torchvision. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, face: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = face.shape
        x = face.view(batch * frames, channels, height, width)
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        logits = logits.view(batch, frames, -1).mean(dim=1)
        return logits
