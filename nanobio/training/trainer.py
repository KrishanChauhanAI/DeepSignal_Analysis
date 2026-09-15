"""
Trainers for both ProtoNet and standard baseline DL classifiers.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

from nanobio.config import ProtoNetCfg, TrainingCfg
from nanobio.models.protonet import HierarchicalProtoNet
from nanobio.training.episodic import FileEpisodeSampler

logger = logging.getLogger("nanobio.training.trainer")


@dataclass
class TrainingHistory:
    """Per-epoch training history."""
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_f1_macro: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_time_s: List[float] = field(default_factory=list)

    def to_dict(self):
        return {k: getattr(self, k) for k in (
            "epoch", "train_loss", "train_acc", "val_loss",
            "val_acc", "val_f1_macro", "learning_rate", "epoch_time_s",
        )}


# ────────────────────────────────────────────────────────────────────────────
# Baseline classifier trainer (raw windows → ResNet/TCN/…)
# ────────────────────────────────────────────────────────────────────────────

class TorchClassifierTrainer:
    """Standard supervised trainer for baseline DL models on raw windows."""

    def __init__(
        self, model: nn.Module, device: torch.device,
        cfg: TrainingCfg, num_classes: int,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.num_classes = num_classes

    def fit(
        self,
        train_ds: Dataset,
        y_train: np.ndarray,
        val_ds: Optional[Dataset] = None,
    ) -> TrainingHistory:
        cfg = self.cfg
        counts = Counter(y_train.tolist())
        cw = torch.tensor(
            [1.0 / max(1, counts[i]) for i in range(self.num_classes)],
            dtype=torch.float32,
        )
        cw = cw / cw.sum() * self.num_classes

        sampler = None
        if cfg.use_weighted_sampler:
            sample_w = cw[torch.from_numpy(y_train.astype(np.int64))]
            sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        if val_ds is None:
            val_size = max(1, int(0.15 * len(train_ds)))
            train_ds, val_ds = random_split(
                train_ds, [len(train_ds) - val_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        tl = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            sampler=sampler, shuffle=sampler is None,
            pin_memory=self.device.type == "cuda",
        )
        vl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False)

        opt = AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        """
        crit = nn.CrossEntropyLoss(
            weight=cw.to(self.device) if cfg.use_class_weights else None,
            label_smoothing=cfg.label_smoothing,
        )"""
        # NEW: Section 29 — configurable loss (focal or cross-entropy)
        loss_type = getattr(cfg, "loss_type", "cross_entropy")
        focal_gamma = getattr(cfg, "focal_gamma", 2.0)

        # ── BUG-011 FIX: Prevent double class-imbalance correction ──
        # When WeightedRandomSampler is active, the data loader already
        # oversamples the minority class.  Applying class weights in the
        # loss function *on top of that* squares the effective correction
        # (ratio²), causing the model to over-predict the minority class.
        # Solution: use loss weights only when the sampler is NOT active.
        if cfg.use_class_weights and not cfg.use_weighted_sampler:
            loss_weights = cw.to(self.device)
        else:
            loss_weights = None
        # ── END BUG-011 FIX ──

        if loss_type == "focal":
            from nanobio.training.losses import FocalLoss
            crit = FocalLoss(
                alpha=loss_weights,  # BUG-011 FIX: was cw.to(self.device) if cfg.use_class_weights else None
                gamma=focal_gamma,
                label_smoothing=cfg.label_smoothing,
            )
            logger.info("Using FocalLoss (gamma=%.2f)", focal_gamma)
        else:
            crit = nn.CrossEntropyLoss(
                weight=loss_weights,  # BUG-011 FIX: was cw.to(self.device) if cfg.use_class_weights else None
                label_smoothing=cfg.label_smoothing,
            )

        history = TrainingHistory()
        best_val, best_state, no_improve = float("inf"), None, 0

        for ep in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            self.model.train()
            tr_loss = tr_cor = tr_n = 0
            for xb, yb in tl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = crit(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip_norm)
                opt.step()
                tr_loss += loss.item() * xb.size(0)
                tr_cor += (logits.argmax(1) == yb).sum().item()
                tr_n += xb.size(0)

            self.model.eval()
            va_loss = va_cor = va_n = 0
            # ── BUG-006 FIX: Collect validation predictions for F1 ──
            all_va_preds: List[int] = []
            all_va_targets: List[int] = []
            # ── END BUG-006 FIX ──
            with torch.no_grad():
                for xb, yb in vl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    loss = crit(logits, yb)
                    va_loss += loss.item() * xb.size(0)
                    va_cor += (logits.argmax(1) == yb).sum().item()
                    va_n += xb.size(0)
                    # ── BUG-006 FIX: Accumulate batch predictions ──
                    all_va_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                    all_va_targets.extend(yb.cpu().numpy().tolist())
                    # ── END BUG-006 FIX ──

            dt = time.time() - t0
            history.epoch.append(ep)
            history.train_loss.append(tr_loss / tr_n)
            history.train_acc.append(tr_cor / tr_n)
            history.val_loss.append(va_loss / va_n)
            history.val_acc.append(va_cor / va_n)
            # ── BUG-006 FIX: Compute actual val F1-macro ──
            # Previously this was hardcoded to 0.0, which caused early
            # stopping on "val_f1_macro" to trigger at epoch 1.
            epoch_val_f1 = float(f1_score(
                all_va_targets, all_va_preds,
                average="macro", zero_division=0,
            ))
            history.val_f1_macro.append(epoch_val_f1)
            # ── END BUG-006 FIX ──
            history.learning_rate.append(opt.param_groups[0]["lr"])
            history.epoch_time_s.append(dt)

            logger.info(
                "  ep %02d train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f f1=%.4f | %.1fs",
                # ── BUG-006 FIX: Added f1 to log line ──
                ep, history.train_loss[-1], history.train_acc[-1],
                history.val_loss[-1], history.val_acc[-1],
                epoch_val_f1, dt,
                # ── END BUG-006 FIX ──
            )
            """
            if history.val_loss[-1] < best_val - 1e-4:
                best_val = history.val_loss[-1]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    logger.info("  Early stopping at epoch %d", ep)
                    break"""
                        # NEW: Section 32 — user-selectable checkpoint metric
            metric_name = getattr(cfg, "best_checkpoint_metric", "val_loss")
            if metric_name == "val_accuracy":
                current = -history.val_acc[-1]     # negate: we minimize
            elif metric_name == "val_f1_macro":
                current = -history.val_f1_macro[-1] if history.val_f1_macro else 1e9
            else:  # val_loss (default)
                current = history.val_loss[-1]

            if current < best_val - 1e-4:
                best_val = current
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    logger.info("  Early stopping at epoch %d (metric=%s)",
                                ep, metric_name)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict(self, ds: Dataset, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(ds, batch_size=batch_size)
        self.model.eval()
        preds, probs = [], []
        for xb, _ in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)
            p = F.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)


# ────────────────────────────────────────────────────────────────────────────
# ProtoNet trainer
# ────────────────────────────────────────────────────────────────────────────

class ProtoNetTrainer:
    """Episodic trainer for the Hierarchical ProtoNet."""

    def __init__(
        self,
        model: HierarchicalProtoNet,
        device: torch.device,
        cfg: ProtoNetCfg,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
        self.history = TrainingHistory()

    def _pack(self, cache: List[dict], indices: List[int]):
        segs, phys = [], []
        for i in indices:
            e = cache[i]
            segs.append(torch.from_numpy(e["segments"]).unsqueeze(1).to(self.device))
            phys.append(torch.from_numpy(e["phys_features"]).to(self.device))
        return segs, phys

    def _episode(self, ep_out, cache, n_way, training):
        s_i, s_y, q_i, q_y = ep_out
        s_segs, s_phys = self._pack(cache, s_i)
        q_segs, q_phys = self._pack(cache, q_i)
        s_y_t = torch.tensor(s_y, dtype=torch.long, device=self.device)
        q_y_t = torch.tensor(q_y, dtype=torch.long, device=self.device)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            out = self.model(
                s_segs, s_phys, s_y_t, q_segs, q_phys, n_way,
            )
            loss = F.cross_entropy(out.logits, q_y_t)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm,
                )
                self.optimizer.step()
            preds = out.logits.argmax(dim=1)
            acc = (preds == q_y_t).float().mean().item()
        return loss.item(), acc, q_y_t.cpu().numpy(), preds.cpu().numpy()

    def fit(
        self,
        train_cache: List[dict], train_labels: List[int],
        val_cache: List[dict], val_labels: List[int],
        n_way: int, seed: int = 42,
    ) -> TrainingHistory:
        cfg = self.cfg
        train_sampler = FileEpisodeSampler(
            train_labels, n_way, cfg.k_shot, cfg.q_query,
            cfg.episodes_per_epoch, seed=seed,
        )
        val_sampler = FileEpisodeSampler(
            val_labels, n_way, cfg.k_shot, cfg.q_query,
            episodes_per_epoch=max(20, cfg.episodes_per_epoch // 5), seed=seed + 1,
        )

        best_f1, best_state, no_improve = -1.0, None, 0

        for ep in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            self.model.train()
            tr_losses, tr_accs = [], []
            for ep_out in train_sampler:
                l, a, _, _ = self._episode(ep_out, train_cache, n_way, training=True)
                tr_losses.append(l); tr_accs.append(a)
            self.scheduler.step()

            self.model.eval()
            va_losses, va_accs, all_true, all_pred = [], [], [], []
            for ep_out in val_sampler:
                l, a, yt, yp = self._episode(ep_out, val_cache, n_way, training=False)
                va_losses.append(l); va_accs.append(a)
                all_true.extend(yt.tolist()); all_pred.extend(yp.tolist())
            val_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

            dt = time.time() - t0
            self.history.epoch.append(ep)
            self.history.train_loss.append(float(np.mean(tr_losses)))
            self.history.train_acc.append(float(np.mean(tr_accs)))
            self.history.val_loss.append(float(np.mean(va_losses)))
            self.history.val_acc.append(float(np.mean(va_accs)))
            self.history.val_f1_macro.append(float(val_f1))
            self.history.learning_rate.append(self.optimizer.param_groups[0]["lr"])
            self.history.epoch_time_s.append(dt)

            logger.info(
                "Epoch %02d/%02d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f f1=%.4f | %.1fs",
                ep, cfg.num_epochs,
                self.history.train_loss[-1], self.history.train_acc[-1],
                self.history.val_loss[-1], self.history.val_acc[-1],
                val_f1, dt,
            )

            if val_f1 > best_f1 + 1e-4:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
                if self.checkpoint_dir:
                    torch.save(best_state, self.checkpoint_dir / "protonet_best.pt")
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    logger.info("Early stopping at epoch %d (best val F1=%.4f)", ep, best_f1)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self.history

    def evaluate(
        self,
        test_cache: List[dict], test_labels: List[int],
        n_way: int, num_episodes: int = 100, seed: int = 999,
    ) -> Dict[str, float]:
        sampler = FileEpisodeSampler(
            test_labels, n_way, self.cfg.k_shot, self.cfg.q_query,
            num_episodes, seed=seed,
        )
        self.model.eval()
        losses, accs, all_true, all_pred = [], [], [], []
        for ep_out in sampler:
            l, a, yt, yp = self._episode(ep_out, test_cache, n_way, training=False)
            losses.append(l); accs.append(a)
            all_true.extend(yt.tolist()); all_pred.extend(yp.tolist())
        return {
            "test_loss": float(np.mean(losses)),
            "test_acc": float(np.mean(accs)),
            "test_f1_macro": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
            "test_f1_weighted": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
            "num_episodes": num_episodes,
        }

# ═══════════════════════════════════════════════════════════════════
# CONTRASTIVE / SELF-SUPERVISED TRAINER
# Added to existing training/trainer.py for consistency.
# ═══════════════════════════════════════════════════════════════════

class ContrastiveTrainer:
    """Self-supervised contrastive trainer (SimCLR / TS2Vec style).

    Trains an encoder on unlabeled windows using augmented view pairs.
    Labels are NEVER accessed during training.

    After training, use encode() to get embeddings for downstream tasks.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        device: torch.device,
        temperature: float = 0.07,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.encoder = encoder.to(device)
        self.projection = projection.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size

        params = list(encoder.parameters()) + list(projection.parameters())
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6,
        )
        self.history = TrainingHistory()

    def _info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent contrastive loss."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)

    def fit(self, windows: np.ndarray, augmenter) -> TrainingHistory:
        """Train on unlabeled windows.

        Args:
            windows: (N, T) unlabeled windows.
            augmenter: callable that takes (T,) array → (T,) augmented array.
        """
        from torch.utils.data import DataLoader, Dataset

        class _ContrastiveDS(Dataset):
            def __init__(self, w, aug):
                self.w = w.astype(np.float32)
                self.aug = aug
            def __len__(self):
                return len(self.w)
            def __getitem__(self, i):
                x = self.w[i]
                v1 = self.aug(x)
                v2 = self.aug(x)
                return (torch.from_numpy(v1[None, :]),
                        torch.from_numpy(v2[None, :]))

        ds = _ContrastiveDS(windows, augmenter)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            drop_last=True, pin_memory=self.device.type == "cuda",
        )

        logger.info("ContrastiveTrainer: %d windows, %d epochs", len(windows), self.epochs)

        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            self.encoder.train()
            self.projection.train()
            ep_loss, n_b = 0.0, 0

            for v1, v2 in loader:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                z1 = self.encoder(v1)
                z2 = self.encoder(v2)
                p1 = F.normalize(self.projection(z1), dim=1)
                p2 = F.normalize(self.projection(z2), dim=1)

                loss = self._info_nce_loss(p1, p2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.optimizer.step()
                ep_loss += loss.item()
                n_b += 1

            self.scheduler.step()
            avg = ep_loss / max(n_b, 1)
            dt = time.time() - t0

            self.history.epoch.append(ep)
            self.history.train_loss.append(avg)
            self.history.train_acc.append(0.0)  # N/A for SSL
            self.history.val_loss.append(0.0)
            self.history.val_acc.append(0.0)
            self.history.val_f1_macro.append(0.0)
            self.history.learning_rate.append(self.optimizer.param_groups[0]["lr"])
            self.history.epoch_time_s.append(dt)

            if ep % max(1, self.epochs // 10) == 0 or ep == 1:
                logger.info("  SSL ep %03d/%03d loss=%.4f (%.1fs)",
                            ep, self.epochs, avg, dt)

        logger.info("ContrastiveTrainer: complete.")
        return self.history

    @torch.no_grad()
    def encode(self, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Encode windows → embeddings using trained encoder."""
        self.encoder.eval()
        ds = torch.from_numpy(windows[:, None, :].astype(np.float32))
        embs = []
        for i in range(0, len(ds), batch_size):
            batch = ds[i:i + batch_size].to(self.device)
            embs.append(self.encoder(batch).cpu().numpy())
        return np.concatenate(embs, axis=0)