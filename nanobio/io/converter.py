"""
.dat → CSV + TXT converter.

Never overwrites original .dat files. Writes copies to a separate
folder tree, preserving the category subdirectory structure.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from nanobio.io.reader import FileMetadata, UniversalDataReader

logger = logging.getLogger("nanobio.io.converter")


class DataConverter:
    """
    Convert a .dat file to CSV and TXT copies.

    Args:
        csv_output_dir: Root output directory for CSV copies.
        txt_output_dir: Root output directory for TXT copies.
        sampling_rate_hz: Sampling rate stored in metadata.
    """

    def __init__(
        self,
        csv_output_dir: Path,
        txt_output_dir: Path,
        sampling_rate_hz: int = 1_000_000,
    ):
        self.csv_dir = Path(csv_output_dir)
        self.txt_dir = Path(txt_output_dir)
        self.sampling_rate = sampling_rate_hz

    def convert_file(
        self,
        source_path: Path,
        category: str,
        signal_columns: Optional[List[int]] = None,
    ) -> Tuple[Optional[Path], Optional[Path], FileMetadata]:
        """Convert one file. Returns (csv_path, txt_path, metadata)."""
        signal, meta = UniversalDataReader.read_file(
            source_path, self.sampling_rate, signal_columns, category,
        )
        meta.conversion_timestamp = datetime.now().isoformat()

        stem = source_path.stem
        csv_dir = self.csv_dir / category
        txt_dir = self.txt_dir / category
        csv_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        csv_path = csv_dir / f"{stem}.csv"
        if not csv_path.exists():
            header = [f"channel_{i}" for i in range(signal.shape[1])]
            pd.DataFrame(signal, columns=header).to_csv(
                csv_path, index=False, float_format="%.10g",
            )

        txt_path = txt_dir / f"{stem}.txt"
        if not txt_path.exists():
            np.savetxt(txt_path, signal, fmt="%.10g", delimiter="\t")

        return csv_path, txt_path, meta

    def convert_all(
        self,
        raw_root: Path,
        extensions: Tuple[str, ...] = (".dat",),
    ) -> List[FileMetadata]:
        """Convert every file under raw_root/<category>/*."""
        all_meta = []
        for cat_dir in sorted(raw_root.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            for ext in extensions:
                for fp in sorted(cat_dir.glob(f"*{ext}")):
                    try:
                        _, _, m = self.convert_file(fp, cat_dir.name)
                        all_meta.append(m)
                    except Exception as e:
                        logger.error("Conversion failed for %s: %s", fp, e)
        logger.info("Converted %d files", len(all_meta))
        return all_meta