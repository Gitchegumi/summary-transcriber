from __future__ import annotations

import sys
import io
import contextlib
import logging as py_logging
import warnings

def configure_backend_logging(verbose: bool = False) -> None:
    """Configures NeMo and Lhotse logging levels depending on verbosity."""
    level = py_logging.WARNING if not verbose else py_logging.DEBUG
    
    # Configure python standard logger for lhotse
    py_logging.getLogger("lhotse").setLevel(level)
    
    # Suppress lhotse warnings specifically
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="lhotse")
        
    # Configure NeMo logger dynamically if installed
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(level)
    except ImportError:
        pass


class SuppressBackendOutput:
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
