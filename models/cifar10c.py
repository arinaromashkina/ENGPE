"""
Wide ResNet-28-10 classifier for CIFAR-10 / CIFAR-10-C experiments.

Architecture:
  WideResNet-28-10 trained from scratch on clean CIFAR-10.
  Global average pooling → 640-d feature vector F_R ∈ R^640.
  Two-layer head: F_C(h) = W2 · ReLU(W1 · h),  W1, W2 ∈ R^{640×640}, R^{10×640}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ---------------------------------------------------------------------------
# WideResNet building blocks
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int,
                 dropout: float = 0.3):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.drop  = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
            if stride != 1 or in_planes != out_planes else nn.Identity()
        )

    def forward(self, x):
        out = self.drop(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class _NetworkBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, out_planes: int,
                 stride: int, dropout: float = 0.3):
        super().__init__()
        layers = [_BasicBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(1, nb_layers):
            layers.append(_BasicBlock(out_planes, out_planes, 1, dropout))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Wide ResNet-28-10 for CIFAR-10 with a two-layer classification head.

    get_features() returns the 640-d penultimate representation F_R(x).
    """

    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n          = (depth - 4) // 6

        self.conv1   = nn.Conv2d(3, n_channels[0], 3, 1, 1, bias=False)
        self.block1  = _NetworkBlock(n, n_channels[0], n_channels[1], 1, dropout)
        self.block2  = _NetworkBlock(n, n_channels[1], n_channels[2], 2, dropout)
        self.block3  = _NetworkBlock(n, n_channels[2], n_channels[3], 2, dropout)
        self.bn1     = nn.BatchNorm2d(n_channels[3])
        self.pool    = nn.AdaptiveAvgPool2d(1)

        feature_dim      = n_channels[3]   # 640
        self.feature_dim = feature_dim
        self.linear1     = nn.Linear(feature_dim, feature_dim)
        self.linear2     = nn.Linear(feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return F_R(x): (B, 640) penultimate features."""
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out))
        out = self.pool(out).flatten(1)
        return self.linear1(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.get_features(x)))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

CIFAR10_TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])

CIFAR10_TEST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
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


def train(model: WideResNet, trainloader, valloader,
          epochs: int = 200, lr: float = 0.1, device: str = 'cuda',
          save_path: str = 'wresnet_cifar10.pth', patience: int = 20) -> None:
    """Train WideResNet-28-10 with SGD + cosine annealing."""
    model = model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9,
                    weight_decay=5e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        scheduler.step()
        if epoch % 10 == 0:
            val_acc = evaluate_accuracy(model, valloader, device)
            print(f"Epoch {epoch:3d}  loss={total_loss/total:.4f}  val={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc, no_improve = val_acc, 0
                torch.save(model.state_dict(), save_path)
            else:
                no_improve += 1
                if no_improve >= patience // 10:
                    print(f"  Early stopping  (best val={best_acc:.4f})")
                    break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Training done. Best val_acc={best_acc:.4f}")
