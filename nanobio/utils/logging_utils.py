"""
Structured logging utilities (Section 38).

Provides a decorator that logs operation start/end with input file, key
parameters, and elapsed wall-clock time — consistently across the pipeline.
"""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional


def log_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    log_args: bool = False,
    include_result: bool = False,
) -> Callable:
    """
    Decorator that adds structured start/end logging with timing.

    Usage:
        @log_operation("baseline_estimation", logger)
        def estimate_baseline(fp, method):
            ...

    Args:
        operation_name: Human-readable operation label.
        logger: Logger to use (defaults to the wrapped function's module logger).
        log_args: If True, log positional/keyword args (careful with large arrays!).
        include_result: If True, log the string representation of the return value.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            # Extract file-path hint if present in args
            file_hint = ""
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, (str, Path)) and (
                    str(a).endswith((".dat", ".csv", ".txt", ".npz", ".pt", ".joblib"))
                ):
                    file_hint = f" | file={Path(a).name}"
                    break

            arg_hint = ""
            if log_args:
                safe_args = [
                    a for a in args
                    if isinstance(a, (int, float, str, bool, Path))
                ]
                safe_kwargs = {
                    k: v for k, v in kwargs.items()
                    if isinstance(v, (int, float, str, bool, Path))
                }
                if safe_args or safe_kwargs:
                    arg_hint = f" | args={safe_args} kwargs={safe_kwargs}"

            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                result_hint = f" | result={result}" if include_result else ""
                log.info(
                    "%s ✓%s%s | elapsed=%.3fs%s",
                    operation_name, file_hint, arg_hint, elapsed, result_hint,
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                log.error(
                    "%s ✗%s%s | elapsed=%.3fs | error=%s",
                    operation_name, file_hint, arg_hint, elapsed, e,
                )
                raise
        return wrapper
    return decorator


class OperationTimer:
    """
    Context manager for timing arbitrary code blocks.

    Usage:
        with OperationTimer("phase3_cache_build", logger) as t:
            ... work ...
        # Logs: "phase3_cache_build ✓ | elapsed=12.345s"
    """
    def __init__(
        self, operation_name: str, logger: logging.Logger,
        extra_context: str = "",
    ):
        self.name = operation_name
        self.logger = logger
        self.extra = extra_context
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.t0
        ctx = f" | {self.extra}" if self.extra else ""
        if exc_type is None:
            self.logger.info("%s ✓%s | elapsed=%.3fs", self.name, ctx, self.elapsed)
        else:
            self.logger.error(
                "%s ✗%s | elapsed=%.3fs | error=%s",
                self.name, ctx, self.elapsed, exc_val,
            )
        return False  # do not suppress exception