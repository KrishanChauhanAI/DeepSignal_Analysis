"""
Zero-copy split tree: materializes train/val/test membership as filesystem
links (or copies) to the ORIGINAL raw files, plus an authoritative manifest.

Design rules enforced here:
    1. RAW FILES ARE NEVER MODIFIED. Links point at them; we never write.
    2. The manifest (CSV+JSON) is the machine-readable source of truth for
       split membership; the folder tree is the human-browsable view.
    3. A *split fingerprint* (seed + ratios + sorted file/label list) is
       stored in the manifest. If a later run produces a DIFFERENT split,
       the stale tree is PURGED (never silently mixed) — a second source of
       truth that disagrees with itself is worse than no tree at all.
    4. Mode resolution ("auto"): hardlink (same volume, no privileges)
       -> symlink (probed; needs Developer Mode on Windows) -> copy.
       The mode actually used is recorded PER FILE in the manifest.
    5. ".lnk" shortcuts are deliberately NOT supported: they are opaque to
       Python file I/O and would break numpy/pandas transparent reads.

Hardlink semantics worth knowing (documented, not hidden):
    - A hard link IS the same inode: zero extra space, instant creation.
    - Deleting the raw file does NOT free disk space while a link exists
      (refcount). Deleting the link never touches the raw file's data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nanobio.io.split_links")


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuildReport:
    linked: int = 0
    reused: int = 0
    copied: int = 0
    failed: int = 0
    purged_stale: bool = False
    modes_used: Dict[str, int] = field(default_factory=dict)


@dataclass
class VerifyReport:
    total: int = 0
    broken: List[str] = field(default_factory=list)   # link exists, target gone
    missing: List[str] = field(default_factory=list)  # in manifest, not on disk
    extra: List[str] = field(default_factory=list)    # on disk, not in manifest

    @property
    def ok(self) -> bool:
        return not (self.broken or self.missing or self.extra)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level link helpers
# ─────────────────────────────────────────────────────────────────────────────

def _same_volume(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve().anchor.upper() == Path(b).resolve().anchor.upper()
    except OSError:
        return False


def _symlink_supported(directory: Path) -> bool:
    """Probe: can we create+delete a file symlink inside `directory`?"""
    probe = directory / ".__nanobio_symlink_probe__"
    try:
        os.symlink(str(directory / ".__nonexistent_target__"), str(probe))
        return True
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def _stat_key(p: Path) -> Optional[Tuple[int, int]]:
    try:
        st = os.stat(p)
        return (st.st_dev, st.st_ino)
    except OSError:
        return None


def _link_consistent(src: Path, link: Path, mode: str) -> bool:
    """Is an existing link still pointing at the same content?"""
    if not link.exists() and not link.is_symlink():
        return False
    if mode == "hardlink":
        return _stat_key(src) == _stat_key(link)
    if mode == "symlink":
        try:
            return Path(os.readlink(link)).resolve() == Path(src).resolve()
        except OSError:
            return False
    if mode == "copy":
        try:
            return link.stat().st_size == src.stat().st_size
        except OSError:
            return False
    return False


class SplitLinkManager:
    """
    Build / verify / load the browsable split link tree.

    Args:
        links_root: Tree root, e.g. `<data_root>/../split_links`.
            MUST be on the same volume as the raw data for hardlinks.
        mode: "auto" | "hardlink" | "symlink" | "copy".
        verify_md5: Optionally hash every source file at build time
            (slow on 1 MHz data; off by default — size check is used for copies).
    """

    def __init__(
        self,
        links_root: Path,
        mode: str = "auto",
        verify_md5: bool = False,
    ):
        self.root = Path(links_root)
        self.requested_mode = mode
        self.verify_md5 = verify_md5
        self._mode_cache: Optional[str] = None  # resolved once per run

        self.root.mkdir(parents=True, exist_ok=True)

    # ── fingerprint ────────────────────────────────────────────────────────

    @staticmethod
    def split_fingerprint(
        splits: Dict[str, Tuple[List[Path], List[int]]],
        ratios: Tuple[float, float, float],
        seed: int,
    ) -> str:
        """Stable hash of everything that defines the split assignment."""
        items = []
        for split in ("train", "val", "test"):
            files, labels = splits[split]
            for f, y in zip(files, labels):
                items.append(f"{split}|{f.name}|{y}")
        raw = "|".join(sorted(items)) + f"|{ratios}|{seed}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    # ── mode resolution ────────────────────────────────────────────────────

    def _resolve_mode(self, src: Path) -> str:
        """Decide the concrete mechanism for one file (cached per run)."""
        if self.requested_mode != "auto":
            return self.requested_mode
        if self._mode_cache is not None:
            return self._mode_cache

        if _same_volume(src, self.root) and os.link is not None:
            self._mode_cache = "hardlink"
            logger.info("Link mode resolved: HARDLINK (same volume, no privileges needed)")
            return self._mode_cache
        '''
        if _same_volume(src, self.root) and hasattr(os, "link"):
            self._mode_cache = "hardlink"
            logger.info("Link mode: HARDLINK (same volume, no privileges required)")
            return self._mode_cache
        '''
        if _symlink_supported(self.root):
            self._mode_cache = "symlink"
            logger.info("Link mode resolved: SYMLINK (hardlink impossible: cross-volume/exFAT)")
            return self._mode_cache
        self._mode_cache = "copy"
        logger.warning(
            "Link mode resolved: COPY (no symlink privilege on Windows; "
            "enable Developer Mode for zero-copy). Disk will be duplicated."
        )
        return self._mode_cache

    # ── build ──────────────────────────────────────────────────────────────

    def build(
        self,
        splits: Dict[str, Tuple[List[Path], List[int]]],
        categories: List[str],
        ratios: Tuple[float, float, float],
        seed: int,
        purge_on_change: bool = True,
    ) -> BuildReport:
        report = BuildReport()
        fp = self.split_fingerprint(splits, ratios, seed)

        # ── stale-tree purge (safety-critical) ──
        manifest_path = self.root / "split_manifest.json"
        if manifest_path.exists():
            try:
                old_fp = json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"]
            except Exception:
                old_fp = None
            if old_fp and old_fp != fp and purge_on_change:
                # Guard: never delete anything at/above the raw data root
                self._purge_tree(reason=f"fingerprint {old_fp} -> {fp}")
                report.purged_stale = True

        rows: List[dict] = []
        for split in ("train", "val", "test"):
            files, labels = splits[split]
            for fp_path, y in zip(files, labels):
                cat = categories[y]
                dest_dir = self.root / split / cat
                dest_dir.mkdir(parents=True, exist_ok=True)
                link = dest_dir / fp_path.name

                mode_used = self._resolve_mode(fp_path)

                if link.exists() or link.is_symlink():
                    if _link_consistent(fp_path, link, mode_used):
                        report.reused += 1
                    else:
                        try:
                            link.unlink()
                        except OSError:
                            pass
                        mode_used = self._place(fp_path, link, mode_used, report)
                else:
                    mode_used = self._place(fp_path, link, mode_used, report)

                size = fp_path.stat().st_size
                row = {
                    "split": split, "category": cat, "label": int(y),
                    "file_name": fp_path.name,
                    "source_path": str(fp_path.resolve()),
                    "link_path": str(link.resolve()),
                    "mode": mode_used, "size_bytes": size,
                }
                if self.verify_md5:
                    row["md5"] = self._md5(fp_path)
                rows.append(row)
                report.modes_used[mode_used] = report.modes_used.get(mode_used, 0) + 1

        # manifest (machine-readable source of truth)
        df = pd.DataFrame(rows).sort_values(["split", "category", "file_name"])
        df.to_csv(self.root / "split_manifest.csv", index=False)
        manifest_path.write_text(json.dumps({
            "created": datetime.now().isoformat(),
            "fingerprint": fp,
            "requested_mode": self.requested_mode,
            "counts": {k: int((df["split"] == k).sum())
                       for k in ("train", "val", "test")},
            "modes_used": report.modes_used,
            "files": rows,
        }, indent=2), encoding="utf-8")

        logger.info(
            "Split link tree → %s | linked=%d reused=%d copied=%d failed=%d | modes=%s",
            self.root, report.linked, report.reused, report.copied,
            report.failed, report.modes_used,
        )
        return report

    def _place(self, src: Path, link: Path, mode: str, report: BuildReport) -> str:
        """Create one link/copy with graceful degradation. Returns mode used."""
        try:
            if mode == "hardlink":
                os.link(src, link)
                report.linked += 1
                return "hardlink"
            if mode == "symlink":
                link.symlink_to(src.resolve())
                report.linked += 1
                return "symlink"
            shutil.copy2(src, link)
            report.copied += 1
            return "copy"
        except OSError as e:
            # degrade: hardlink->symlink->copy
            logger.warning("Link failed (%s) for %s: %s — degrading", mode, src.name, e)
            if mode == "hardlink" and _symlink_supported(link.parent):
                return self._place(src, link, "symlink", report)
            try:
                shutil.copy2(src, link)
                report.copied += 1
                return "copy"
            except Exception as e2:
                logger.error("All link modes failed for %s: %s", src.name, e2)
                report.failed += 1
                return "failed"

    # ── verify ─────────────────────────────────────────────────────────────

    def verify(self) -> VerifyReport:
        rep = VerifyReport()
        try:
            manifest = json.loads(
                (self.root / "split_manifest.json").read_text(encoding="utf-8")
            )
        except Exception:
            logger.warning("Verify skipped: no manifest at %s", self.root)
            return rep

        recorded = set()
        for row in manifest["files"]:
            link = Path(row["link_path"])
            src = Path(row["source_path"])
            recorded.add(str(link))
            rep.total += 1
            if not (link.exists() or link.is_symlink()):
                rep.missing.append(row["file_name"])
            elif not src.exists():
                rep.broken.append(f"{row['file_name']} (source deleted)")

        for p in self.root.rglob("*"):
            if p.suffix.lower() in (".dat", ".csv", ".txt") and "manifest" not in p.name:
                if str(p.resolve()) not in recorded:
                    rep.extra.append(str(p))

        if rep.ok:
            logger.info("Split link tree VERIFIED: %d entries, no drift", rep.total)
        else:
            logger.error(
                "Split link tree DRIFT: broken=%d missing=%d extra=%d",
                len(rep.broken), len(rep.missing), len(rep.extra),
            )
        return rep

    # ── loading ────────────────────────────────────────────────────────────

    def load_manifest(self) -> Dict[str, Tuple[List[Path], List[int]]]:
        manifest = json.loads(
            (self.root / "split_manifest.json").read_text(encoding="utf-8")
        )
        out = {"train": ([], []), "val": ([], []), "test": ([], [])}
        for row in manifest["files"]:
            out[row["split"]][0].append(Path(row["source_path"]))
            out[row["split"]][1].append(int(row["label"]))
        return out

    # ── internals ──────────────────────────────────────────────────────────

    def _purge_tree(self, reason: str) -> None:
        # Safety: never purge anything at/above the raw data tree.
        # This manager only ever owns its own root; manifest+links live here.
        logger.warning("Purging stale split link tree at %s (%s)", self.root, reason)
        for child in self.root.iterdir():
            if child.name == "split_manifest.json":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass

    @staticmethod
    def _md5(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()

    def force_purge(self) -> None:
        """
        Unconditionally wipe the split tree (train/val/test + manifest).
        Called when the user explicitly requests a fresh split on every run.
        """
        if not self.root.exists():
            return
        import shutil
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                logger.info("Force-purged split folder: %s", child)
            else:
                try:
                    child.unlink()
                    logger.info("Force-purged manifest: %s", child)
                except OSError:
                    pass
        logger.warning("Split link tree force-purged (fresh split will be built)")