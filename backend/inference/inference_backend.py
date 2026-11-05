"""Shared inference interface + metric helpers (BUILD 15).

Both backends conform to :class:`InferenceBackend`::

    chat(...)    -> Ollama-shaped dict {"message": {"role", "content",
                         "tool_calls": [...]}, ...}
    embed(...)   -> {"embeddings": [[float, ...], ...]}
    vision(...)  -> Ollama-shaped chat dict (image turn)
    health(...)  -> JSON-serialisable status dict (never raises)

Metrics recorded per request (both backends, same keys):

    full_latency_s        wall time of the whole call
    first_token_latency_s approximation (== full latency for non-streaming;
                          kept as a separate key so streaming upgrades slot in)
    tokens_per_s          from backend eval counters when present, else
                          len(content)/latency estimate, else None
    ram_mb / vram_mb      best-effort host readings, None when unavailable
    model_load_time_s     time spent (re)loading weights (0 on cache hit)

System metric helpers never raise — they return None on failure so that
health/audit paths can never crash inference.
"""
import ctypes
import shutil
import subprocess
import threading
import time


class InferenceError(RuntimeError):
    """Raised when a backend cannot serve a request.

    ``ModelRouter`` catches this and falls back to the other backend.
    The canonical user-facing prefix lives on ``ToolBrain.FALLBACK_NOTICE``.
    """


class InferenceBackend:
    """Interface every inference backend must conform to."""

    name = "base"

    def __init__(self):
        self._lock = threading.Lock()
        self.metrics = {
            "total_requests": 0,
            "total_failures": 0,
            "last_latency_s": None,
            "last_first_token_s": None,
            "last_tokens_per_s": None,
            "last_model": None,
            "last_backend": self.name,
            "last_error": None,
            "model_load_time_s": None,
        }
        self.history = []  # last 20 per-request metric dicts (capped)

    # -- interface (overridden by backends) -------------------------------
    def chat(self, messages, model, tools=None, options=None, timeout=240):
        raise NotImplementedError

    def embed(self, model, inputs, timeout=60):
        raise NotImplementedError

    def vision(self, prompt, jpeg_bytes, model=None, timeout=150):
        raise NotImplementedError

    def health(self):
        raise NotImplementedError

    def list_models(self):
        return []

    # -- shared metric plumbing -------------------------------------------
    def _record(self, latency_s, model=None, tokens_per_s=None,
                load_time_s=None, error=None, backend=None, kind="chat"):
        entry = {
            "t": time.strftime("%H:%M:%S"),
            "backend": backend or self.name,
            "kind": kind,
            "model": model,
            "full_latency_s": round(latency_s, 3) if latency_s is not None else None,
            # Non-streaming approximation: first token == full response.
            "first_token_latency_s": round(latency_s, 3) if latency_s is not None else None,
            "tokens_per_s": (round(tokens_per_s, 2)
                             if isinstance(tokens_per_s, (int, float)) else None),
            "ram_mb": ram_mb(),
            "vram_mb": vram_mb(),
            "model_load_time_s": load_time_s,
        }
        with self._lock:
            self.metrics["total_requests"] += 1
            self.metrics["last_latency_s"] = entry["full_latency_s"]
            self.metrics["last_first_token_s"] = entry["first_token_latency_s"]
            self.metrics["last_tokens_per_s"] = entry["tokens_per_s"]
            self.metrics["last_model"] = model
            self.metrics["last_backend"] = entry["backend"]
            if load_time_s is not None:
                self.metrics["model_load_time_s"] = load_time_s
            if error is not None:
                self.metrics["total_failures"] += 1
                self.metrics["last_error"] = str(error)[:300]
            else:
                self.metrics["last_error"] = None
            self.history.append(entry)
            del self.history[:-20]
        return entry

    def metric_snapshot(self):
        with self._lock:
            return {
                "backend": self.name,
                "total_requests": self.metrics["total_requests"],
                "total_failures": self.metrics["total_failures"],
                "first_token_latency_s": self.metrics["last_first_token_s"],
                "full_response_latency_s": self.metrics["last_latency_s"],
                "tokens_per_s": self.metrics["last_tokens_per_s"],
                "ram_mb": ram_mb(),
                "vram_mb": vram_mb(),
                "model_load_time_s": self.metrics["model_load_time_s"],
                "last_model": self.metrics["last_model"],
                "last_error": self.metrics["last_error"],
            }


def ram_mb():
    """Host RAM in use (MB). Best effort — None when unreadable."""
    try:
        import psutil  # optional; present on many dev machines
        return round(psutil.Process().memory_info().rss / 1048576, 1)
    except Exception:
        pass
    try:
        # Windows fallback: GlobalMemoryStatusEx -> used physical MB.
        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            used = (st.ullTotalPhys - st.ullAvailPhys) / 1048576
            return round(used, 1)
    except Exception:
        pass
    return None


def vram_mb():
    """GPU memory in use (MB) via nvidia-smi, else torch.cuda. None when N/A."""
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                vals = [float(x) for x in out.stdout.replace(",", " ").split()
                        if x.strip().replace(".", "", 1).isdigit()]
                if vals:
                    return round(sum(vals), 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1048576, 1)
    except Exception:
        pass
    return None


def estimate_tps(content, latency_s):
    """Rough tokens/sec from ~4 chars/token when no eval counters exist."""
    try:
        n = max(1, len(content or "") / 4.0)
        if latency_s and latency_s > 0:
            return round(n / latency_s, 2)
    except Exception:
        pass
    return None
