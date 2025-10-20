#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Callable, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Length sampler (asymmetric)
# ---------------------------

def sample_l_vec_asymmetric(n: int, l: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bounded, skewed two-uniform mixture with E[l_i] = l (in expectation).

    Left bucket:  Uniform{a, ..., L}
    Right bucket: Uniform{L, ..., b}
    where a = floor(l/2), b = ceil(3l/2), L = floor(l).
    The mixture weight w is chosen so that E[l_i] = l.

    Returns:
        l_vec: (n,) integer array with l_i >= 1
    """
    L = int(np.floor(l))
    a = max(1, L // 2)
    b = max(L, int(np.ceil(1.5 * l)))

    mu_left  = 0.5 * (a + L)
    mu_right = 0.5 * (L + b)
    denom = (mu_right - mu_left)
    if denom <= 0:
        w = 0.5
    else:
        w = (mu_right - l) / denom
        w = float(np.clip(w, 0.0, 1.0))

    choose_left = rng.random(n) < w
    l_left  = rng.integers(a, L + 1, size=n)  # inclusive upper via +1
    l_right = rng.integers(L, b + 1, size=n)
    l_vec = np.where(choose_left, l_left, l_right).astype(int)
    l_vec[l_vec < 1] = 1
    return l_vec


# ---------------------------
# Data generator (ragged)
# ---------------------------

class FullTaskSampler:
    def __init__(
        self,
        d: int,
        l: int,
        n: int,
        rho: float,
        Ctask: np.ndarray,
        *,
        functionality: float = 1.0,
        length_sampler: Optional[Callable[[int, int, np.random.Generator], np.ndarray]] = None,
        seed: Optional[int] = None,
        nonvariable: int = 0,
    ) -> None:
        """
        d: feature dimension
        l: baseline target mean/context (passed to length_sampler)
        n: batch size
        rho: label noise variance
        Ctask: (d x d) task covariance
        functionality: power applied to <x, w> (1.0 = linear). If non-integer,
                       uses sign(x)|x|^p to stay in R.
        length_sampler: callable(n, l, rng) -> (n,) int lengths l_i >= 1. If None, constant l.
        seed: RNG seed
        """
        self.d = d
        self.l = l
        self.n = n
        self.rho = rho
        self.Ctask = Ctask
        self.functionality = functionality
        self.rng = np.random.default_rng(seed)
        self.length_sampler = length_sampler
        self.nonvariable = nonvariable

    def __iter__(self):
        return self

    def _sample_lengths(self) -> np.ndarray:
        if self.length_sampler is None:
            return np.full(self.n, int(max(1, self.l)), dtype=int)
        if self.nonvariable == 1:
            return np.full(self.n, int(max(1, self.l)), dtype=int)
        l_vec = self.length_sampler(self.n, self.l, self.rng).astype(int)
        if np.any(l_vec < 1):
            raise ValueError("All sampled context lengths must be >= 1.")
        return l_vec

    def _power_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply signed power to keep outputs real for non-integer exponents."""
        p = self.functionality
        if p == 1 or p == 1.0:
            return x
        return np.sign(x) * (np.abs(x) ** p)

    def __next__(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns:
          Z_list: list length n; each Z_i has shape (l_i+1, d+1):
                  - Z_i[:, :d] = x tokens (context + query)
                  - Z_i[:,  d] = y tokens; query label at index l_i is zeroed
          y_query: (n,) true query labels
        """
        d, n = self.d, self.n
        l_vec = self._sample_lengths()

        # One task vector per sample
        ws = self.rng.multivariate_normal(mean=np.zeros(d), cov=self.Ctask, size=n)  # (n, d)
        ws = ws[:, :, None]  # (n, d, 1)

        Z_list: List[np.ndarray] = []
        y_query = np.empty(n, dtype=float)

        for i in range(n):
            Li = int(l_vec[i])  # context length; query at index Li
            # x_i: (Li+1, d)
            x_i = self.rng.normal(loc=0.0, scale=1.0/np.sqrt(d), size=(Li + 1, d))
            # noise: (Li+1, 1)
            noise_i = self.rng.normal(loc=0.0, scale=np.sqrt(self.rho), size=(Li + 1, 1))
            # y_i: (Li+1, 1)
            dot_i = (x_i @ ws[i])  # (Li+1, 1)
            y_i = self._power_fn(dot_i) + noise_i

            # Build Z_i: (Li+1, d+1)
            Z_i = np.zeros((Li + 1, d + 1), dtype=float)
            Z_i[:, :d] = x_i
            Z_i[:, d] = y_i.squeeze(-1)

            # Save true query, hide in Z
            y_query[i] = Z_i[Li, d]
            Z_i[Li, d] = 0.0

            Z_list.append(Z_i)

        return Z_list, y_query


# ---------------------------
# Torch models
# ---------------------------

class LinearAttentionModel(nn.Module):
    """
    Expects Z of shape (d+1, l_i+1).
    Returns scalar prediction at channel -1, query position -1.
    """
    def __init__(self, d: int):
        super().__init__()
        self.V = nn.Parameter(torch.eye(d + 1))
        self.K = nn.Parameter(torch.eye(d + 1))
        self.Q = nn.Parameter(torch.eye(d + 1))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        dp1, lp1 = Z.shape
        # VZ: (d+1, l_i+1)
        VZ = self.V @ Z
        KZ = self.K @ Z
        QZ = self.Q @ Z
        # (KZ.T @ QZ): (l_i+1, l_i+1)
        A = VZ @ ((KZ.T @ QZ) / (lp1 - 1))
        return A[-1, -1]


class BatchedLinearAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.core = LinearAttentionModel(d)

    def forward(self, batch_Z: List[torch.Tensor]) -> torch.Tensor:
        # Each Z_i: (d+1, l_i+1)
        return torch.stack([self.core(Z) for Z in batch_Z], dim=0)  # (n,)


# ---------------------------
# Utilities
# ---------------------------

def to_torch_for_linear_attention(
    Z_list: List[np.ndarray],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    # Sampler yields (l_i+1, d+1); model expects (d+1, l_i+1)
    return [torch.from_numpy(Z_i.T).to(device=device, dtype=dtype) for Z_i in Z_list]


def set_torch_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm() + 1e-12)
    return (x @ y) / denom


# ---------------------------
# Training
# ---------------------------

def train(cfg):
    import wandb

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    set_torch_seed(cfg.seed)

    # Ctask
    if cfg.ctask == "identity":
        C = np.eye(cfg.d)
    else:
        # simple diagonal spectrum example
        diag = np.linspace(1.0, 0.1, cfg.d)
        C = np.diag(diag)

    # Sampler
    if cfg.variablecontext:
        print("variable context sampling", flush = True)
        sampler = FullTaskSampler(
            d=cfg.d,
            l=int(cfg.alpha * cfg.d),
            n=int(cfg.tau * cfg.d * cfg.d),
            rho=cfg.rho,
            Ctask=C,
            functionality=cfg.functionality,
            length_sampler=sample_l_vec_asymmetric,  # plug your own if you like
            seed=cfg.seed,
        )
    else:
        print("fixed context sampling", flush = True)
        sampler = FullTaskSampler(
            d=cfg.d,
            l=int(cfg.alpha * cfg.d),
            n=int(cfg.tau * cfg.d * cfg.d),
            rho=cfg.rho,
            Ctask=C,
            functionality=cfg.functionality,
            length_sampler=None,  # plug your own if you like
            seed=cfg.seed,
        ) 

    # Model/opt
    model = BatchedLinearAttention(cfg.d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.amp))

    # wandb
    run = wandb.init(
        project=cfg.project,
        name=cfg.run_name or None,
        config=vars(cfg),
        mode=("offline" if os.environ.get("WANDB_MODE", "").lower() == "offline" else "online"),
    )
    wandb.watch(model, log="all", log_freq=cfg.log_every)

    global_step = 0
    for step in range(1, cfg.steps + 1):
        model.train()
        Z_list, yq_np = next(sampler)                     # ragged batch
        batch_Z = to_torch_for_linear_attention(Z_list, device=device, dtype=torch.float32)
        y_true = torch.from_numpy(yq_np).to(device=device, dtype=torch.float32)  # (n,)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and cfg.amp)):
            y_pred = model(batch_Z)                      # (n,)
            loss = torch.mean((y_pred - y_true) ** 2)

        scaler.scale(loss).backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                corr = pearsonr_torch(y_pred.detach(), y_true.detach()).item()
                wandb.log({
                    "train/mse": loss.item(),
                    "train/corr": corr,
                    "train/y_pred_mean": y_pred.mean().item(),
                    "train/y_true_mean": y_true.mean().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)
        global_step += 1

        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                Z_list_val, yq_val_np = next(sampler)     # fresh batch
                batch_Z_val = to_torch_for_linear_attention(Z_list_val, device=device)
                y_true_val = torch.from_numpy(yq_val_np).to(device=device, dtype=torch.float32)
                y_pred_val = model(batch_Z_val)
                mse_val = torch.mean((y_pred_val - y_true_val) ** 2).item()
                corr_val = pearsonr_torch(y_pred_val, y_true_val).item()
                wandb.log({
                    "val/mse": mse_val,
                    "val/corr": corr_val,
                }, step=global_step)

    run.finish()
    print("Training complete.")


# ---------------------------
# CLI
# ---------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Linear Attention on Ragged Contexts + wandb")
    p.add_argument("--project", type=str, default="linear-attn", help="wandb project name")
    p.add_argument("--run_name", type=str, default="", help="wandb run name")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--d", type=int, default=64, help="feature dimension")
    p.add_argument("--alpha", type=float, default=50, help="baseline/target mean context length")
    p.add_argument("--variablecontext", action="store_true", help="enable variable context lengths")
    p.add_argument("--no-variablecontext", dest="variablecontext", action="store_false")
    p.set_defaults(variablecontext=True)
    p.add_argument("--tau", type=float, default=256, help="batch size (#contexts)")
    p.add_argument("--rho", type=float, default=0.01, help="label noise variance")
    p.add_argument("--functionality", type=float, default=1.0, help="power on <x,w>")
    p.add_argument("--ctask", type=str, choices=["identity", "diag_decay"], default="identity")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--amp", action="store_true", help="use CUDA AMP mixed precision")
    return p


if __name__ == "__main__":
    cfg = build_argparser().parse_args()

    # Light checks for wandb availability
    try:
        import wandb  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "wandb is required. Install with `pip install wandb` and login via `wandb login`."
        ) from e

    train(cfg)