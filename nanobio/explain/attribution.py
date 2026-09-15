"""
Time-series attribution methods: Integrated Gradients and Occlusion.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    steps: int = 32,
    device: torch.device = None,
) -> np.ndarray:
    """
    Integrated Gradients attribution for a single sample.

    Args:
        model: PyTorch model.
        x: (1, C, T) input tensor.
        target_class: Class index for attribution.
        baseline: Reference baseline (default = zeros).
        steps: Number of integration steps.

    Returns:
        (C, T) attribution array.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = x.to(device)
    if baseline is None:
        baseline = torch.zeros_like(x)
    baseline = baseline.to(device)

    alphas = torch.linspace(0, 1, steps).to(device)
    grads_sum = torch.zeros_like(x)
    for a in alphas:
        interp = baseline + a * (x - baseline)
        interp.requires_grad_(True)
        logits = model(interp)
        target = logits[:, target_class].sum()
        grad = torch.autograd.grad(target, interp)[0]
        grads_sum += grad

    avg_grads = grads_sum / steps
    ig = ((x - baseline) * avg_grads).detach().cpu().numpy()[0]
    return ig


def occlusion_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    window_size: int = 32,
    stride: int = 16,
    device: torch.device = None,
) -> np.ndarray:
    """
    Occlusion-based saliency: score drops when patches are zeroed.

    Returns a (T,) importance curve where higher = more important.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        base_prob = F.softmax(model(x), dim=1)[0, target_class].item()

    T = x.size(-1)
    importance = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)
    for start in range(0, T - window_size + 1, stride):
        occluded = x.clone()
        occluded[..., start: start + window_size] = 0
        with torch.no_grad():
            prob = F.softmax(model(occluded), dim=1)[0, target_class].item()
        drop = base_prob - prob
        importance[start: start + window_size] += drop
        counts[start: start + window_size] += 1
    return importance / np.maximum(counts, 1)