from __future__ import annotations

import sys
import io
import contextlib
import logging as py_logging
import warnings

def configure_quiet_backend_logging(verbose: bool = False) -> None:
    """Configures logging levels for loggers likely used by NeMo and Lhotse before NeMo is imported."""
    level = py_logging.DEBUG if verbose else py_logging.ERROR
    
    loggers_to_mute = [
        "nemo",
        "nemo_logger",
        "nemo.utils",
        "nemo.collections",
        "lhotse",
        "pytorch_lightning",
        "lightning",
    ]
    for name in loggers_to_mute:
        logger = py_logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = verbose
        if not verbose:
            logger.handlers.clear()
        
    # Suppress warnings specifically from these modules
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="lhotse")
        warnings.filterwarnings("ignore", category=UserWarning, module="nemo.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)


def configure_nemo_logging(verbose: bool = False) -> None:
    """Configures NeMo's own logging level after NeMo is imported."""
    level = py_logging.DEBUG if verbose else py_logging.ERROR
    level_name = "DEBUG" if verbose else "ERROR"
    try:
        from nemo.utils import logging as nemo_logging
        level_val = getattr(nemo_logging, level_name, nemo_logging.ERROR)
        nemo_logging.setLevel(level_val)
    except ImportError:
        pass

    nemo_logger = py_logging.getLogger("nemo_logger")
    nemo_logger.setLevel(level)
    for h in nemo_logger.handlers:
        h.setLevel(level)
    if not verbose:
        nemo_logger.handlers.clear()


class suppress_backend_output:
    """Context manager to suppress stdout and stderr, capturing it in a buffer."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self._stdout_redirector = contextlib.redirect_stdout(self.stdout_buffer)
        self._stderr_redirector = contextlib.redirect_stderr(self.stderr_buffer)

    def __enter__(self):
        if self.enabled:
            self._stdout_redirector.__enter__()
            self._stderr_redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._stdout_redirector.__exit__(exc_type, exc_val, exc_tb)
            self._stderr_redirector.__exit__(exc_type, exc_val, exc_tb)
        return False  # Do not swallow exceptions

    def get_captured_output(self) -> str:
        stdout_val = self.stdout_buffer.getvalue()
        stderr_val = self.stderr_buffer.getvalue()
        parts = []
        if stdout_val:
            parts.append(f"--- STDOUT ---\n{stdout_val}")
        if stderr_val:
            parts.append(f"--- STDERR ---\n{stderr_val}")
        return "\n".join(parts)
