import torch
from torch import nn


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
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
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(audio))


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
