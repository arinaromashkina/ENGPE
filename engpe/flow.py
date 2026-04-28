"""
Conditional Normalizing Flow for Empirical Null Score Generation.

Architecture (Y-branch):
  - One branch: F_R → F_C → classification logits
  - Other branch: F_R → encoder → CNF → null logits

The CNF is trained to approximate the null logit distribution conditioned
on the penultimate features F_R(t). Because both branches share F_R, any
distribution shift in the input propagates equally to both the model scores
and the generated null scores — enabling accurate FDR estimation without
retraining on the shifted test data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Feature normalizer
# ---------------------------------------------------------------------------

class RobustFeatureNormalizer(nn.Module):
    """
    Robust per-coordinate normalization for penultimate features.

    Uses running median and IQR (instead of mean/std) to tolerate
    heavy-tailed OOD deviations. The tanh compression bounds the output
    to (-1, 1) regardless of how far features deviate from training.

    Statistics are updated during training via exponential moving average
    (momentum=0.01) and frozen at inference time.
    """

    def __init__(self, feature_dim: int, clip_val: float = 5.0,
                 momentum: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.clip_val = clip_val
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_median', torch.zeros(feature_dim))
        self.register_buffer('running_iqr',    torch.ones(feature_dim))
        self.register_buffer('initialized',    torch.tensor(False))

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        batch_median = x.median(dim=0).values
        q75 = torch.quantile(x, 0.75, dim=0)
        q25 = torch.quantile(x, 0.25, dim=0)
        batch_iqr = (q75 - q25).clamp(min=self.eps)

        if not self.initialized:
            self.running_median.copy_(batch_median)
            self.running_iqr.copy_(batch_iqr)
            self.initialized.fill_(True)
        else:
            self.running_median.mul_(1 - self.momentum).add_(batch_median * self.momentum)
            self.running_iqr.mul_(1 - self.momentum).add_(batch_iqr * self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_stats(x)
        if not self.initialized:
            return torch.tanh(x * 0.01)
        x_norm = (x - self.running_median) / (self.running_iqr + self.eps)
        x_norm = x_norm.clamp(-self.clip_val, self.clip_val)
        return torch.tanh(x_norm / self.clip_val)


# ---------------------------------------------------------------------------
# ActNorm
# ---------------------------------------------------------------------------

class ActNorm(nn.Module):
    """
    Activation normalization (data-driven per-channel affine).

    Initialized from the first forward batch so the output has zero mean
    and unit variance. Parameters are stored in state_dict for checkpointing.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias      = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if not self.initialized and not reverse:
            with torch.no_grad():
                self.bias.data      = -x.mean(0)
                self.log_scale.data = -x.std(0).clamp(min=1e-6).log()
            self.initialized.fill_(True)

        if not reverse:
            y       = (x + self.bias) * self.log_scale.exp()
            log_det = self.log_scale.sum().expand(x.size(0))
            return y, log_det
        else:
            x_rec   = x * (-self.log_scale).exp() - self.bias
            log_det = -self.log_scale.sum().expand(x.size(0))
            return x_rec, log_det


# ---------------------------------------------------------------------------
# Coupling layer  (multi-class: K >= 2)
# ---------------------------------------------------------------------------

class CouplingLayer(nn.Module):
    """
    Affine coupling layer for K-dimensional score vectors.

    Split: x1 = x[:d], x2 = x[d:]
    Forward: z2 = x2 * exp(s(x1, e)) + t(x1, e)
    Reverse: x2 = (z2 - t) * exp(-s)

    Scale head applies tanh so s ∈ (-1, 1), preventing exp blow-up.
    All output-head weights are initialized to zero (identity transform).
    """

    def __init__(self, dim: int, encoder_dim: int,
                 hidden_dim: int = 256, mask_type: str = 'first_half'):
        super().__init__()
        self.mask_type = mask_type

        if mask_type == 'first_half':
            self.d_in  = dim // 2
            self.d_out = dim - dim // 2
        else:
            self.d_in  = dim - dim // 2
            self.d_out = dim // 2

        self.net = nn.Sequential(
            nn.Linear(self.d_in + encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.scale_head     = nn.Sequential(nn.Linear(hidden_dim // 2, self.d_out), nn.Tanh())
        self.translate_head = nn.Linear(hidden_dim // 2, self.d_out)

        nn.init.zeros_(self.scale_head[0].weight)
        nn.init.zeros_(self.scale_head[0].bias)
        nn.init.zeros_(self.translate_head.weight)
        nn.init.zeros_(self.translate_head.bias)

    def _split(self, x):
        if self.mask_type == 'first_half':
            return x[:, :self.d_in], x[:, self.d_in:]
        return x[:, self.d_out:], x[:, :self.d_out]

    def _merge(self, x1, x2):
        if self.mask_type == 'first_half':
            return torch.cat([x1, x2], dim=1)
        return torch.cat([x2, x1], dim=1)

    def forward(self, x: torch.Tensor, features: torch.Tensor,
                reverse: bool = False):
        x1, x2     = self._split(x)
        cond_input = torch.cat([x1, features], dim=1)
        h          = self.net(cond_input)
        s          = self.scale_head(h)
        t          = self.translate_head(h)
        if not reverse:
            z2      = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
            return self._merge(x1, z2), log_det
        else:
            x2_rec  = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
            return self._merge(x1, x2_rec), log_det


# ---------------------------------------------------------------------------
# Element-wise affine layer  (binary: K = 1)
# ---------------------------------------------------------------------------

class ElementwiseAffineLayer(nn.Module):
    """
    Scalar affine layer for binary classification (K=1).

    z = x * exp(s(e)) + t(e)   where e is the encoded feature vector.
    Scale uses tanh so s ∈ (-1, 1).
    """

    def __init__(self, encoder_dim: int, hidden_dim: int = 256):
        super().__init__()
        trunk = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.scale_net     = nn.Sequential(trunk, nn.Linear(hidden_dim, 1), nn.Tanh())
        self.translate_net = nn.Sequential(trunk, nn.Linear(hidden_dim, 1))

        for net in [self.scale_net, self.translate_net]:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, x: torch.Tensor, features: torch.Tensor,
                reverse: bool = False):
        s = self.scale_net(features)      # (B, 1)
        t = self.translate_net(features)  # (B, 1)
        if not reverse:
            z       = x * torch.exp(s) + t
            log_det = s.squeeze(1)
            return z, log_det
        else:
            x_rec   = (x - t) * torch.exp(-s)
            log_det = -s.squeeze(1)
            return x_rec, log_det


# ---------------------------------------------------------------------------
# Full conditional normalizing flow
# ---------------------------------------------------------------------------

class ScoreShiftFlow(nn.Module):
    """
    Conditional normalizing flow P(null_scores | penultimate_features).

    Architecture:
      1. RobustFeatureNormalizer: median/IQR + clamp + tanh → always in (-1, 1)
      2. Two-layer feature encoder: feature_dim → encoder_dim
      3. n_flows coupling layers (alternating masks) interleaved with ActNorm

    For K=1 (binary), uses ElementwiseAffineLayer instead of CouplingLayer.
    """

    _LOG_2PI = float(np.log(2 * np.pi))

    def __init__(self, score_dim: int = 10, feature_dim: int = 640,
                 n_flows: int = 12, hidden_dim: int = 256,
                 encoder_dim: int = 128, clip_val: float = 5.0):
        super().__init__()
        self.score_dim = score_dim

        self.feature_norm = RobustFeatureNormalizer(
            feature_dim, clip_val=clip_val, momentum=0.01)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, encoder_dim), nn.LayerNorm(encoder_dim), nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for i in range(n_flows):
            if score_dim == 1:
                self.layers.append(ElementwiseAffineLayer(encoder_dim, hidden_dim))
            else:
                mask = 'first_half' if i % 2 == 0 else 'second_half'
                self.layers.append(CouplingLayer(score_dim, encoder_dim, hidden_dim, mask))
            if i < n_flows - 1:
                self.layers.append(ActNorm(score_dim))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(self.feature_norm(features))

    def forward(self, scores: torch.Tensor, features: torch.Tensor,
                reverse: bool = False):
        enc         = self.encode(features)
        log_det_sum = torch.zeros(scores.size(0), device=scores.device)
        layers      = list(reversed(self.layers)) if reverse else self.layers

        x = scores
        for layer in layers:
            if isinstance(layer, ActNorm):
                x, ld = layer(x, reverse=reverse)
            else:
                x, ld = layer(x, enc, reverse=reverse)
            log_det_sum = log_det_sum + ld
        return x, log_det_sum

    def log_prob(self, scores: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(scores, features, reverse=False)
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.score_dim * self._LOG_2PI
        return log_pz + log_det

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        z = torch.randn(features.size(0), self.score_dim, device=features.device)
        scores, _ = self.forward(z, features, reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Wrapper: training + inference
# ---------------------------------------------------------------------------

class ScoreShiftFlowWrapper(nn.Module):
    """
    High-level wrapper around ScoreShiftFlow.

    Provides:
      - ``train_flow(dataset, ...)``   — fit CNF on training data
      - ``generate_decoys(dataset, ...)`` — sample null scores for evaluation
    """

    def __init__(self, num_classes: int = 10, n_flows: int = 12,
                 feature_dim: int = 640, hidden_dim: int = 256,
                 encoder_dim: int = 128, clip_val: float = 5.0):
        super().__init__()
        self.num_classes = num_classes
        self.flow = ScoreShiftFlow(
            score_dim=num_classes, feature_dim=feature_dim,
            n_flows=n_flows, hidden_dim=hidden_dim,
            encoder_dim=encoder_dim, clip_val=clip_val,
        )

    def train_flow(self, score_dataset, epochs: int = 30, lr: float = 3e-4,
                   batch_size: int = 256, device: str = 'cuda',
                   patience: int = 5, grad_clip: float = 1.0) -> 'ScoreShiftFlowWrapper':
        """
        Train CNF with maximum log-likelihood on (null_scores, features) pairs.
        Uses AdamW + cosine annealing + early stopping.
        """
        self.flow.to(device).train()
        loader    = DataLoader(score_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_loss, best_state, no_improve = float('inf'), None, 0

        for epoch in range(epochs):
            total_loss, n_batches = 0.0, 0
            for _, features, target_decoy, _ in loader:
                features, target_decoy = features.to(device), target_decoy.to(device)
                loss = -self.flow.log_prob(target_decoy, features).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.flow.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  "
                      f"loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

            if avg_loss < best_loss - 1e-4:
                best_loss  = avg_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in self.flow.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}  (best loss={best_loss:.4f})")
                    break

        if best_state is not None:
            self.flow.load_state_dict(best_state)
        print(f"  Training done. Best loss: {best_loss:.4f}")
        return self

    def generate_decoys(self, score_dataset, device: str = 'cuda', n_samples: int = 1):
        """
        Generate one null score vector per sample.

        Returns
        -------
        model_scores : np.ndarray, shape (N, C)
        decoy_scores : np.ndarray, shape (N, C)
        labels       : np.ndarray, shape (N,)
        """
        self.flow.to(device).eval()
        cnn_list, decoy_list, lbl_list = [], [], []
        loader = DataLoader(score_dataset, batch_size=256, shuffle=False, num_workers=0)

        with torch.no_grad():
            for cnn_scores, features, _, labels in loader:
                features = features.to(device)
                if n_samples == 1:
                    decoy = self.flow.sample(features)
                else:
                    decoy = torch.stack(
                        [self.flow.sample(features) for _ in range(n_samples)], dim=0
                    ).mean(dim=0)
                cnn_list.append(cnn_scores.cpu().numpy())
                decoy_list.append(decoy.cpu().numpy())
                lbl_list.append(labels.cpu().numpy())

        return (np.concatenate(cnn_list),
                np.concatenate(decoy_list),
                np.concatenate(lbl_list))
