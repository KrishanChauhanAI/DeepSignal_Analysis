"""
Local experiment registry (Section 43 — experiment tracking).

Maintains a JSON index of every experiment run in the project. Each entry
records timestamp, config hash, key results, and paths to artifacts.
Enables comparing runs without a full MLflow/W&B install.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("nanobio.utils.experiment_registry")

REGISTRY_FILENAME = "experiment_registry.json"


def _config_hash(config_dict: Dict[str, Any]) -> str:
    """Stable hash of a nested config dict (order-independent)."""
    return hashlib.md5(
        json.dumps(config_dict, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


class ExperimentRegistry:
    """
    Local, file-based experiment tracker.

    Args:
        project_root: Root directory where `experiment_registry.json` lives.
    """

    def __init__(self, project_root: Path):
        self.registry_path = Path(project_root) / REGISTRY_FILENAME

    def _load(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read registry (%s): %s", self.registry_path, e)
            return []

    def _save(self, entries: List[Dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)

    def register(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        artifact_paths: Dict[str, str],
        results_summary: Dict[str, Any] = None,
    ) -> str:
        """
        Register a new experiment run.

        Args:
            experiment_id: Timestamp-based experiment id (e.g. exp_20260904_170512).
            config: Full config dict (from PipelineConfig.to_dict()).
            artifact_paths: Dict of {label: path} for figures/reports/models.
            results_summary: Optional dict of headline metrics.

        Returns:
            The config hash used to identify duplicate runs.
        """
        entries = self._load()
        entry = {
            "experiment_id": experiment_id,
            "registered_at": datetime.now().isoformat(),
            "config_hash": _config_hash(config),
            "artifacts": artifact_paths,
            "results": results_summary or {},
        }
        entries.append(entry)
        self._save(entries)
        logger.info(
            "Experiment registered: %s (config_hash=%s, total_experiments=%d)",
            experiment_id, entry["config_hash"], len(entries),
        )
        return entry["config_hash"]

    def update_results(
        self, experiment_id: str, results_summary: Dict[str, Any],
    ) -> None:
        """Attach post-run results to an existing entry."""
        entries = self._load()
        for e in entries:
            if e["experiment_id"] == experiment_id:
                e["results"].update(results_summary)
                e["updated_at"] = datetime.now().isoformat()
                self._save(entries)
                return
        logger.warning("Experiment %s not found in registry — cannot update",
                       experiment_id)

    def find_by_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return all previous runs whose config matches the given one."""
        target_hash = _config_hash(config)
        return [e for e in self._load() if e.get("config_hash") == target_hash]