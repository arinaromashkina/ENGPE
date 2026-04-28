"""
EfficientNet-B0 binary classifier for Camelyon17 (WILDS).

Task: predict whether the central 32×32 region of a 96×96 histopathology
patch contains tumour tissue. Train set: hospitals 0, 3, 4.
Test set: hospitals 1, 2 (domain shift).

Head architecture:
  EfficientNet-B0 backbone (ImageNet pretrained, 1280-d output)
    → Dropout(0.4)
    → Linear [1280 → 256] + BN + ReLU  ← penultimate features F_R
    → Linear [256 → 1]                  ← binary logit F_C
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Camelyon17Classifier(nn.Module):
    """
    EfficientNet-B0 with a two-layer binary classification head.

    get_features() returns the 256-d penultimate representation F_R(x)
    used to condition the ENGPE normalizing flow.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone     = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None)
        self.backbone    = backbone.features
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(p=0.4)
        self.linear1     = nn.Linear(1280, 256)
        self.bn          = nn.BatchNorm1d(256)
        self.linear2     = nn.Linear(256, 1)
        self.feature_dim = 256

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return F_R(x): (B, 256) penultimate features."""
        h = self.pool(self.backbone(x)).flatten(1)
        return self.linear1(self.dropout(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(self.get_features(x)))
        return self.linear2(h)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

CAMELYON_TRAIN_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CAMELYON_TEST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def evaluate_accuracy(model: nn.Module, loader, device: str = 'cuda') -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds  = (model(images).squeeze(1) >= 0).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def train(model: Camelyon17Classifier, trainloader, valloader,
          epochs: int = 10, lr: float = 5e-4, device: str = 'cuda',
          save_path: str = 'camelyon17_classifier.pth',
          pos_weight: float = 3.14, patience: int = 3) -> None:
    """
    Fine-tune with AdamW + ReduceLROnPlateau.
    Weighted BCE to handle class imbalance (pos_weight ≈ 3.14 for Camelyon17).
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))

    best_acc, no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            images = images.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += ((logits >= 0) == labels.bool()).sum().item()
            total      += images.size(0)

        val_acc = evaluate_accuracy(model, valloader, device)
        scheduler.step(1 - val_acc)
        print(f"Epoch {epoch:2d}  loss={total_loss/total:.4f}  val={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience * 2:
                print(f"  Early stopping  (best val={best_acc:.4f})")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Training done. Best val_acc={best_acc:.4f}")
