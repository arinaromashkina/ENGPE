"""
ResNet-50 classifier for BREEDS subpopulation-shift benchmarks.

Head architecture:
  backbone (ResNet-50, ImageNet pretrained)
    → AvgPool → flatten [2048]
    → Linear [2048 → feature_dim]  ← penultimate features F_R
    → ReLU
    → Linear [feature_dim → K]     ← logits F_C

get_features() returns the feature_dim-dimensional vector F_R(x) used to
condition the ENGPE normalizing flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class BREEDSClassifier(nn.Module):
    """
    ResNet-50 backbone with a two-layer classification head.

    Parameters
    ----------
    num_classes  : K — number of source classes
    feature_dim  : dimension of the penultimate feature space (default 640)
    pretrained   : load ImageNet-1K weights (default True)
    freeze_backbone : freeze ResNet-50 body (default False)
    """

    def __init__(self, num_classes: int, feature_dim: int = 640,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone    = nn.Sequential(*list(backbone.children())[:-1])
        self.linear1     = nn.Linear(2048, feature_dim)
        self.linear2     = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return F_R(x): (B, feature_dim) penultimate features."""
        return self.linear1(self.backbone(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.get_features(x)))

    def forward_with_features(self, x: torch.Tensor):
        feats  = self.get_features(x)
        logits = self.linear2(F.relu(feats))
        return logits, feats


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

BREEDS_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575]),
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
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def train(model: BREEDSClassifier, trainloader, valloader,
          epochs: int = 30, lr: float = 0.01, device: str = 'cuda',
          save_path: str = 'breeds_classifier.pth', patience: int = 5) -> None:
    """
    Fine-tune with SGD + Nesterov momentum + cosine annealing.
    Saves the checkpoint with the best validation accuracy.
    """
    model = model.to(device)
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        scheduler.step()
        val_acc = evaluate_accuracy(model, valloader, device)
        print(f"Epoch {epoch:3d}  loss={total_loss/total:.4f}"
              f"  train={correct/total:.4f}  val={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping  (best val={best_acc:.4f})")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Training done. Best val_acc={best_acc:.4f}")
