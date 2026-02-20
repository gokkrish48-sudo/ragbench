"""Structured logging with rich console output."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(console=console, show_path=False, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[bold cyan]{label}[/] completed in [bold]{elapsed:.3f}s[/]"
    if logger:
        logger.info(msg)
    else:
        console.print(msg)
