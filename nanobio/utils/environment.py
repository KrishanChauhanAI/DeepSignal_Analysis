"""
Environment capture for full reproducibility (Section 37).

Records Python version, OS, CUDA, GPU/CPU details, installed package
versions, and git commit hash. Written once per experiment to a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("nanobio.utils.environment")

# Packages we care about tracking. Missing packages are silently skipped.
TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn",
    "torch", "torchvision", "xgboost", "joblib", "pyyaml",
    "umap-learn", "pacmap", "shap", "hmmlearn", "tabpfn", "tsai", "aeon",
    "reportlab",
]


def _get_git_info() -> Dict[str, str]:
    """Capture git commit / branch / dirty state. Returns empty dict if not a repo."""
    info = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Check if working tree is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["commit"] = commit
        info["branch"] = branch
        info["dirty"] = bool(status)
        if status:
            info["dirty_files_count"] = len(status.splitlines())
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo, or git not installed
        pass
    return info


def _get_package_versions() -> Dict[str, str]:
    """Return {package_name: version} for TRACKED_PACKAGES that are installed."""
    versions = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
        for pkg in TRACKED_PACKAGES:
            try:
                versions[pkg] = version(pkg)
            except PackageNotFoundError:
                pass
    except ImportError:
        # Python < 3.8 fallback
        for pkg in TRACKED_PACKAGES:
            try:
                mod = __import__(pkg.replace("-", "_"))
                v = getattr(mod, "__version__", None)
                if v:
                    versions[pkg] = str(v)
            except ImportError:
                pass
    return versions


def _get_cpu_info() -> Dict[str, Any]:
    """CPU description without external dependencies."""
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
    }
    # Physical core count (best-effort, Linux only)
    try:
        with open("/proc/cpuinfo") as f:
            physical = len({
                line for line in f
                if line.startswith("physical id")
            })
            if physical > 0:
                info["physical_sockets"] = physical
    except (OSError, FileNotFoundError):
        pass
    return info


def _get_gpu_info() -> Dict[str, Any]:
    """GPU / CUDA details (torch-based; safe if CUDA unavailable)."""
    info = {"cuda_available": False}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass
    return info


def capture_environment(
    experiment_id: str,
    config_snapshot: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Build a complete environment snapshot for reproducibility.

    Args:
        experiment_id: Unique identifier for the experiment run.
        config_snapshot: Optional dict of the experiment config for cross-reference.

    Returns:
        Dictionary with all captured environment info.
    """
    env = {
        "experiment_id": experiment_id,
        "captured_at": datetime.now().isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "cpu": _get_cpu_info(),
        "gpu": _get_gpu_info(),
        "packages": _get_package_versions(),
        "git": _get_git_info(),
        "environment_variables": {
            k: v for k, v in os.environ.items()
            if k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "PYTHONHASHSEED", "PYTORCH_CUDA_ALLOC_CONF")
        },
    }
    if config_snapshot is not None:
        env["config_snapshot_keys"] = sorted(config_snapshot.keys())
    return env


def save_environment(env: Dict[str, Any], output_path: Path) -> Path:
    """Write environment snapshot to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, sort_keys=False)
    logger.info("Environment snapshot saved: %s", output_path)
    return output_path


def log_environment_summary(env: Dict[str, Any]) -> None:
    """One-liner summary for the pipeline log."""
    py = env["python"]["version"]
    os_name = env["os"]["system"]
    n_pkgs = len(env["packages"])
    gpu_line = "no GPU"
    if env["gpu"].get("cuda_available"):
        gpus = env["gpu"].get("gpus", [])
        if gpus:
            gpu_line = f"{len(gpus)}× {gpus[0]['name']} (CUDA {env['gpu'].get('cuda_version')})"
    git_line = ""
    if env["git"].get("commit"):
        git_line = f" | git={env['git']['commit'][:8]}"
        if env["git"].get("dirty"):
            git_line += "-dirty"
    logger.info(
        "Environment: Python %s | %s | %s | %d packages tracked%s",
        py, os_name, gpu_line, n_pkgs, git_line,
    )