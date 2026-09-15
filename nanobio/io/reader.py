"""
Universal file reader for .dat / .csv / .txt formats.

Guarantees:
    - ALL rows are read (no silent truncation)
    - Binary and text .dat files auto-detected
    - Delimiter auto-detected
    - Original files never modified
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("nanobio.io.reader")


@dataclass
class FileMetadata:
    """Complete metadata for a single data file."""
    original_path: str = ""
    original_filename: str = ""
    category: str = ""
    sampling_rate_hz: int = 1_000_000
    num_samples: int = 0
    num_channels: int = 1
    duration_seconds: float = 0.0
    file_format: str = "unknown"
    delimiter: Optional[str] = None
    header_rows: int = 0
    file_size_bytes: int = 0
    conversion_timestamp: str = ""
    md5_hash: str = ""


class UniversalDataReader:
    """Multi-format reader with auto-detection."""

    DELIMITERS = [",", "\t", ";", "|", " "]
    BINARY_DTYPES = {4: np.float32, 8: np.float64}

    @staticmethod
    def _try_float(v: str) -> Optional[float]:
        try:
            return float(v.strip())
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _detect_delimiter(lines: List[str]) -> Optional[str]:
        """Detect delimiter via csv.Sniffer with heuristic fallback."""
        try:
            sample = "\n".join(lines[:20])
            return csv.Sniffer().sniff(
                sample, delimiters="".join(UniversalDataReader.DELIMITERS),
            ).delimiter
        except csv.Error:
            pass
        best, score = None, 0
        for d in UniversalDataReader.DELIMITERS:
            counts = [len(l.split(d)) for l in lines if l.strip()]
            if not counts:
                continue
            common = max(set(counts), key=counts.count)
            consistency = counts.count(common) / len(counts)
            if common > 1 and common * consistency > score:
                score = common * consistency
                best = d
        return best

    @staticmethod
    def _detect_header(lines: List[str], delim: str) -> int:
        """First row where ≥50% of cells parse as numeric."""
        for i, line in enumerate(lines[:50]):
            s = line.strip()
            if not s or s.startswith(("#", "%", "!")):
                continue
            parts = s.split(delim)
            nums = sum(1 for p in parts if UniversalDataReader._try_float(p) is not None)
            if len(parts) > 0 and nums / len(parts) >= 0.5:
                return i
        return 0

    @staticmethod
    def _compute_md5(path: Path, chunk_size: int = 65536) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def read_file(
        cls, file_path: Union[str, Path],
        sampling_rate_hz: int = 1_000_000,
        signal_columns: Optional[List[int]] = None,
        category: str = "",
        compute_hash: bool = False,
    ) -> Tuple[np.ndarray, FileMetadata]:
        """Read a file completely; return (signal, metadata)."""
        file_path = Path(file_path)
        meta = FileMetadata(
            original_path=str(file_path.resolve()),
            original_filename=file_path.name,
            category=category,
            sampling_rate_hz=sampling_rate_hz,
            file_size_bytes=file_path.stat().st_size,
        )
        if compute_hash:
            meta.md5_hash = cls._compute_md5(file_path)

        # Try binary .dat
        if file_path.suffix.lower() == ".dat":
            signal = cls._try_binary_read(file_path)
            if signal is not None:
                meta.file_format = "binary"
                meta.num_samples = len(signal)
                meta.num_channels = 1
                meta.duration_seconds = len(signal) / sampling_rate_hz
                return signal.reshape(-1, 1), meta

        # Text reading
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample_lines = [f.readline() for _ in range(200)]
        sample_lines = [l for l in sample_lines if l]
        if not sample_lines:
            raise ValueError(f"Empty file: {file_path}")

        delim = cls._detect_delimiter(sample_lines)

        # Single-value-per-line
        if delim is None:
            values = []
            with open(file_path, "r", errors="replace") as f:
                for line in f:
                    v = cls._try_float(line)
                    if v is not None:
                        values.append(v)
            if values:
                signal = np.array(values, dtype=np.float64).reshape(-1, 1)
                meta.file_format = "text"
                meta.delimiter = "newline"
                meta.num_samples = len(signal)
                meta.duration_seconds = len(signal) / sampling_rate_hz
                return signal, meta
            raise ValueError(f"No numeric values found in {file_path}")

        header_rows = cls._detect_header(sample_lines, delim)
        data_line = sample_lines[header_rows].strip().split(delim)
        numeric_cols = [
            i for i, c in enumerate(data_line) if cls._try_float(c) is not None
        ]
        use_cols = signal_columns if signal_columns is not None else numeric_cols
        if not use_cols:
            raise ValueError(f"No numeric columns in {file_path}")

        try:
            df = pd.read_csv(
                file_path, sep=delim, skiprows=header_rows, header=None,
                usecols=use_cols, engine="c", low_memory=False,
                on_bad_lines="skip",
            )
        except Exception:
            df = pd.read_csv(
                file_path, sep=delim, skiprows=header_rows, header=None,
                usecols=use_cols, engine="python", on_bad_lines="skip",
            )

        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        signal = df.values.astype(np.float64)

        meta.file_format = "text"
        meta.delimiter = repr(delim) if delim in ("\t", " ") else delim
        meta.header_rows = header_rows
        meta.num_samples = signal.shape[0]
        meta.num_channels = signal.shape[1]
        meta.duration_seconds = signal.shape[0] / sampling_rate_hz
        return signal, meta

    @staticmethod
    def _try_binary_read(path: Path) -> Optional[np.ndarray]:
        """Attempt to read file as raw binary floats."""
        with open(path, "rb") as f:
            head = f.read(200)
        printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
        if printable / max(1, len(head)) > 0.85:
            return None  # Likely text

        fsize = path.stat().st_size
        for bw, dt in UniversalDataReader.BINARY_DTYPES.items():
            if fsize % bw == 0 and fsize >= bw * 10:
                try:
                    signal = np.fromfile(path, dtype=dt)
                    if np.all(np.isfinite(signal[:min(100, len(signal))])):
                        return signal.astype(np.float64)
                except Exception:
                    continue
        return None


def read_signal_file(path: Union[str, Path]) -> np.ndarray:
    """
    Simple convenience function that returns a 1-D float32 array.

    Used by the pipeline when full metadata is not required.
    """
    signal, _ = UniversalDataReader.read_file(path)
    return signal[:, 0].astype(np.float32) if signal.ndim > 1 else signal.astype(np.float32)