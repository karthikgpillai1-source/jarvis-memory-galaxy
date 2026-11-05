"""llama.cpp inference backend (BUILD 17).

Local-first GGUF inference via ``llama-cpp-python``. This module is the ONLY
place that talks to llama.cpp; everything else goes through
:class:`~backend.inference.model_router.ModelRouter`.

Roles (one resident GGUF per role, loaded once and cached)::

    tool / chat   Qwen2.5-Coder GGUF  - conversation + OS tool reasoning.
                  Both roles resolve to the same file by default, so they
                  SHARE one instance (never 2x VRAM).
    embed         nomic-embed GGUF    - semantic memory recall.
    vision        Qwen2.5-VL GGUF +   - screen understanding / grounding.
                  mmproj projector

The card cannot hold a 7B text model and a 7B VL model at the same time
(RTX 5050 Laptop, 8151 MiB). Heavy roles therefore evict each other on load;
the tiny embed model stays resident.

Design notes::

    * Defaults request full GPU offload + large context; the loader SELF-FITS
      to the card (shrinks context, then offload) so a big GGUF degrades
      gracefully instead of crashing the process.
    * Tool calling uses the ChatML function-calling template. Without it
      llama-cpp-python silently drops the ``tools`` payload and the model
      answers in prose, which makes every OS tool turn a no-op. Set
      ``JARVIS_LLAMACPP_CHAT_FORMAT`` to override (empty = keep the GGUF's
      own embedded template).
    * Everything degrades gracefully: a missing package, a missing GGUF, or a
      CUDA-less build all surface as ``model_loaded: False`` + an
      ``InferenceError`` rather than a crash.

Env knobs::

    JARVIS_LLAMACPP_MODEL           explicit GGUF path (wins over discovery)
    JARVIS_LLAMACPP_TOOL_MODEL      tool/chat GGUF path or HF-style hint
    JARVIS_LLAMACPP_CHAT_MODEL      chat GGUF path (defaults to the tool model)
    JARVIS_LLAMACPP_EMBED_MODEL     embedding GGUF (auto-discovered)
    JARVIS_LLAMACPP_VISION_MODEL    vision GGUF (auto-discovered)
    JARVIS_LLAMACPP_MMPROJ          vision projector GGUF (auto-discovered)
    JARVIS_LLAMACPP_DIR             extra directory to search for *.gguf
    JARVIS_LLAMACPP_N_CTX           context length (default 16384)
    JARVIS_LLAMACPP_N_CTX_VISION    vision context length (default 8192)
    JARVIS_LLAMACPP_GPU_LAYERS      GPU offload layers (default 60, 0 = CPU)
    JARVIS_LLAMACPP_THREADS         CPU threads (default: auto)
    JARVIS_LLAMACPP_CHAT_FORMAT     chat template (default chatml-function-calling)
"""
import glob
import json
import os
import re
import threading
import time

from .inference_backend import InferenceBackend, InferenceError, estimate_tps

INSTALL_HINT = ("llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python  "
                "(CUDA: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python)")

DEFAULT_TOOL_HINT = "qwen2.5-coder-7b-instruct-q4_k_m"
DEFAULT_EMBED_HINT = "nomic-embed-text"
DEFAULT_VISION_HINT = "qwen2.5-vl"
# BUILD 18 fix: 32768 looked fine on a short decode benchmark but is
# catastrophic for real turns. A 7B model's KV cache at 32k ctx needs
# ~13 GB, which cannot fit beside the 4.6 GB of weights in 8 GB of VRAM,
# so the cache spills to system RAM and every token thrashes. Measured on
# this RTX 5050, prefill of a ~2900-token prompt:
#     ctx=32768 -> 189.5s      ctx=16384 -> 2.3s
#     ctx=8192  ->   2.1s      ctx=4096  -> 2.2s
# 16384 is the largest context that keeps the KV cache on the GPU.
DEFAULT_N_CTX = 16384
DEFAULT_N_CTX_VISION = 8192
DEFAULT_GPU_LAYERS = 60
DEFAULT_CHAT_FORMAT = "chatml-function-calling"
ALL_LAYERS = 999  # internal sentinel for "offload everything" (passed as -1)
# Largest context whose KV cache still fits on an 8 GB card beside a 7B
# model. Above this the cache spills to system RAM and throughput collapses
# (measured: 32k ctx = 189s prefill vs 2.3s at 16k). See DEFAULT_N_CTX.
_MAX_OFFLOAD_CTX = 16384

# Roles that own a large slice of VRAM. Only one may be resident at a time.
HEAVY_ROLES = ("tool", "chat", "vision")
LIGHT_ROLES = ("embed",)

# File-name signals used to classify a discovered GGUF.
_EMBED_HINTS = ("nomic-embed", "nomic_embed", "-embed-", "bge-", "gte-",
                "e5-", "mxbai-embed")
_VISION_HINTS = ("qwen2.5-vl", "qwen2-vl", "qwen2.5-omni", "minicpm-v",
                 "llava", "-vl-", "vision")
_MMPROJ_HINTS = ("mmproj",)

_CUDA_DLL_READY = False


def _vram_fraction():
    """Used/total VRAM (0.0-1.0) via nvidia-smi. None when unavailable."""
    try:
        import shutil
        import subprocess as _sp
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        out = _sp.run([smi, "--query-gpu=memory.used,memory.total",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        parts = out.stdout.strip().split(",")
        used, total = float(parts[0].strip()), float(parts[1].strip())
        if total <= 0:
            return None
        return max(0.0, min(1.0, used / total))
    except Exception:
        return None


def _vram_free_mb():
    """Free VRAM in MB via nvidia-smi. None when unavailable."""
    try:
        import shutil
        import subprocess as _sp
        smi = shutil.which("nvidia-smi")
        if not smi:
            return None
        out = _sp.run([smi, "--query-gpu=memory.free,memory.total",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        parts = out.stdout.strip().split(",")
        free, total = float(parts[0].strip()), float(parts[1].strip())
        if total <= 0:
            return None
        return max(0.0, free)
    except Exception:
        return None


def _ensure_cuda_dll_path():
    """Expose pip-installed NVIDIA runtime DLLs to the Windows loader.

    CUDA wheels bundle ggml-cuda.dll but link cublas/cudart dynamically;
    without the CUDA toolkit on PATH, llama.dll fails with "one of its
    dependencies" missing. The pip ``nvidia-*`` packages ship those DLLs
    under site-packages/nvidia/*/bin; this makes them discoverable.
    Runs once per process. Never raises.
    """
    global _CUDA_DLL_READY
    if _CUDA_DLL_READY:
        return
    _CUDA_DLL_READY = True
    try:
        import sys as _sys
        cands = []
        for _sp in _sys.path:
            try:
                if not _sp or not os.path.isdir(_sp):
                    continue
                _nv = os.path.join(_sp, "nvidia")
                if not os.path.isdir(_nv):
                    continue
                for _root, _ds, _fs in os.walk(_nv):
                    low = _root.lower()
                    if low.endswith("bin") or low.endswith(os.path.join("bin", "x64")) \
                            or low.endswith(os.path.join("bin", "x86_64")):
                        cands.append(_root)
            except Exception:
                continue
        for _d in cands:
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(_d)
            except Exception:
                pass
            try:
                if _d.lower() not in os.environ.get("PATH", "").lower():
                    os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass
    except Exception:
        pass


def _gpu_arg(layers):
    """Map the internal ALL_LAYERS sentinel to llama.cpp's -1."""
    try:
        layers = int(layers)
    except Exception:
        layers = DEFAULT_GPU_LAYERS
    return -1 if layers >= ALL_LAYERS else layers


def _norm_path(p):
    try:
        return os.path.normcase(os.path.abspath(p))
    except Exception:
        return ""


def classify(path):
    """Classify a GGUF path: 'mmproj' | 'embed' | 'vision' | 'text'."""
    try:
        base = os.path.basename(path).lower()
    except Exception:
        return "text"
    if any(h in base for h in _MMPROJ_HINTS):
        return "mmproj"
    if any(h in base for h in _EMBED_HINTS):
        return "embed"
    if any(h in base for h in _VISION_HINTS):
        return "vision"
    return "text"


class ModelManager:
    """Load-once GGUF cache. One entry per role key, never reloaded per request."""

    _cache = {}        # key -> {"llm": Llama, "path": str, "meta": dict}
    _lock = threading.Lock()
    _load_times = {}   # key -> seconds spent loading

    @classmethod
    def loaded(cls):
        with cls._lock:
            return {k: {"path": v["path"], "meta": dict(v.get("meta", {}))}
                    for k, v in cls._cache.items()}

    @classmethod
    def is_loaded(cls, key):
        with cls._lock:
            return key in cls._cache

    @classmethod
    def get(cls, key):
        with cls._lock:
            entry = cls._cache.get(key)
            return entry["llm"] if entry else None

    @classmethod
    def load_time(cls, key):
        with cls._lock:
            return cls._load_times.get(key)

    @classmethod
    def _evict_conflicts(cls, key):
        """Unload other HEAVY roles so one large GGUF owns the GPU."""
        if key not in HEAVY_ROLES:
            return []
        with cls._lock:
            victims = [k for k in cls._cache
                       if k in HEAVY_ROLES and k != key]
        evicted = [v for v in victims if cls.unload(v)]
        if evicted:
            # Let the CUDA allocator release before we claim the memory.
            time.sleep(0.5)
        return evicted

    @classmethod
    def load(cls, key, path, n_ctx=DEFAULT_N_CTX, n_gpu_layers=DEFAULT_GPU_LAYERS,
             n_threads=None, verbose=False, llama_kwargs=None):
        """Return (llm, load_seconds) for ``key``, loading it exactly once."""
        with cls._lock:
            if key in cls._cache:
                return cls._cache[key]["llm"], 0.0
        _ensure_cuda_dll_path()  # CUDA runtime DLLs before first import/load
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise InferenceError(f"{INSTALL_HINT} ({e})")
        if not path or not os.path.exists(path):
            raise InferenceError(
                f"Inference backend unavailable - no GGUF found for role "
                f"'{key}'. Set the matching JARVIS_LLAMACPP_* environment "
                f"variable to a local .gguf path.")

        extra = dict(llama_kwargs or {})
        req_ctx = int(n_ctx or DEFAULT_N_CTX)
        req_layers = (ALL_LAYERS if _gpu_arg(n_gpu_layers) == -1
                      else int(n_gpu_layers if n_gpu_layers is not None
                               else DEFAULT_GPU_LAYERS))
        threads = {"n_threads": int(n_threads)} if n_threads else {}

        # Ordered fit ladder: requested first, then balanced mid configs
        # (smaller ctx + fewer layers - these are the FAST ones), then
        # big-context spill fallbacks, then CPU last. First success wins.
        mid_layers = [l for l in (40, 28, 20) if l <= req_layers]
        mid_ctx = [c for c in (16384, 8192, 4096) if c <= req_ctx]
        attempts, seen = [], set()
        attempts.append((req_ctx, req_layers))
        seen.add((req_ctx, req_layers))
        for l in mid_layers:
            for c in mid_ctx:
                if (c, l) not in seen:
                    seen.add((c, l))
                    attempts.append((c, l))
        for l in mid_layers:
            if req_ctx > 16384 and (req_ctx, l) not in seen:
                seen.add((req_ctx, l))
                attempts.append((req_ctx, l))
        attempts.append((min(req_ctx, 4096), 0))  # CPU fallback = last resort
        attempts = attempts[:12]

        t0 = time.time()
        last_err, llm, used_ctx, used_layers = None, None, req_ctx, req_layers
        for c, l in attempts:
            try:
                cand = Llama(model_path=path, n_ctx=c,
                             n_gpu_layers=_gpu_arg(l), verbose=bool(verbose),
                             **threads, **extra)
            except Exception as e:
                last_err = e
                continue
            # Oversubscription guard: a fit leaving <350MB free thrashes.
            # Treat it as a miss and keep descending.
            try:
                free_mb = _vram_free_mb()
            except Exception:
                free_mb = None
            if (free_mb is not None and free_mb < 350
                    and not (c == min(req_ctx, 4096) and l == 0)):
                try:
                    cand.close()
                except Exception:
                    pass
                last_err = RuntimeError(
                    f"only {free_mb:.0f}MB VRAM free at ctx={c}/layers={l} - "
                    "headroom too small, trying a smaller fit")
                time.sleep(1.0)
                continue
            # BUILD 18 fix: a fit can load cleanly and STILL be unusable. The
            # oversubscription guard above only sees free VRAM after the
            # weights land; the KV cache is allocated on first use and is what
            # actually spills to RAM. A 7B model needs ~13 GB of KV cache at
            # 32k ctx - far more than the whole card. Reject such fits up front
            # instead of serving 0.02 tok/s turns.
            if l > 0 and c > _MAX_OFFLOAD_CTX:
                try:
                    cand.close()
                except Exception:
                    pass
                last_err = RuntimeError(
                    f"ctx={c} needs a KV cache too large for this GPU "
                    f"(offload-safe limit {_MAX_OFFLOAD_CTX}); it would spill "
                    "to system RAM and crawl. Lower JARVIS_LLAMACPP_N_CTX.")
                time.sleep(0.2)
                continue
            llm, used_ctx, used_layers, last_err = cand, c, l, None
            break
        if llm is None:
            # BUILD 18: no second engine to absorb this. Spell out the VRAM
            # situation so the failure is actionable rather than a generic
            # "backend unavailable" the user has to decode.
            try:
                size_mb = round(os.path.getsize(path) / (1024 ** 2))
            except Exception:
                size_mb = None
            free_mb = _vram_free_mb()
            detail = (f"file='{os.path.basename(path)}'"
                      + (f" size={size_mb}MB" if size_mb else "")
                      + (f" vram_free={free_mb:.0f}MB" if free_mb else "")
                      + f" | tried {len(attempts)} fits: {attempts}")
            raise InferenceError(
                f"llama.cpp could not load '{os.path.basename(path)}' - "
                f"{detail} | last error: {last_err}")
        # Warmup: the first generation after load pays CUDA graph capture +
        # kernel autotuning. One throwaway 2-token run so the first REAL
        # request is already fast.
        try:
            llm.create_completion("Hi", max_tokens=2, temperature=0.0)
        except Exception:
            pass
        took = time.time() - t0
        meta = {"n_ctx": used_ctx, "n_gpu_layers": used_layers,
                "threads": threads.get("n_threads"),
                "fit_degraded": (used_ctx != req_ctx or used_layers != req_layers),
                "requested": {"n_ctx": req_ctx, "n_gpu_layers": req_layers},
                "kind": classify(path)}
        try:
            meta["vocab_size"] = int(llm.n_vocab())
            meta["context_params"] = int(llm.n_ctx())
        except Exception:
            pass
        with cls._lock:
            if key not in cls._cache:  # second thread may have won the race
                cls._cache[key] = {"llm": llm, "path": path, "meta": meta}
                cls._load_times[key] = round(took, 2)
            return cls._cache[key]["llm"], cls._load_times[key]

    @classmethod
    def unload(cls, key):
        with cls._lock:
            entry = cls._cache.pop(key, None)
            cls._load_times.pop(key, None)
        if entry is not None:
            try:
                entry["llm"].close()
            except Exception:
                pass
            return True
        return False

    @classmethod
    def unload_all(cls):
        with cls._lock:
            keys = list(cls._cache.keys())
        for k in keys:
            cls.unload(k)
        return keys

    @classmethod
    def metadata(cls, key):
        with cls._lock:
            entry = cls._cache.get(key)
            if not entry:
                return None
            return {"path": entry["path"], **dict(entry.get("meta", {})),
                    "load_time_s": cls._load_times.get(key)}


def _candidate_dirs():
    dirs = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # backend/inference/
        root = os.path.dirname(os.path.dirname(here))      # project root
        dirs.append(os.path.join(root, "models"))
    except Exception:
        pass
    extra = os.environ.get("JARVIS_LLAMACPP_DIR", "")
    if extra:
        dirs.append(extra)
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    if home:
        dirs.append(os.path.join(home, "models"))
        dirs.append(os.path.join(home, ".cache", "huggingface", "hub"))
    seen, out = set(), []
    for d in dirs:
        if d and d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return out


def _all_ggufs():
    """Every discoverable *.gguf as (path, kind). Search dirs + project root."""
    found = []
    dirs = list(_candidate_dirs())
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))
        if root and os.path.isdir(root):
            dirs.append(root)
    except Exception:
        pass
    seen = set()
    for d in dirs:
        try:
            if not d or not os.path.isdir(d):
                continue
            for fp in glob.glob(os.path.join(d, "*.gguf")) + \
                    glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
                n = _norm_path(fp)
                if n in seen:
                    continue
                seen.add(n)
                found.append((fp, classify(fp)))
        except Exception:
            continue
    return found


def _env_path(*names):
    """First existing absolute path from the given env vars."""
    for n in names:
        v = (os.environ.get(n, "") or "").strip().strip("\"'")
        if v and os.path.exists(v):
            return v
    return None


def _stem_of(v):
    return os.path.splitext(os.path.basename(v))[0].lower() if v else ""


def find_gguf(hint=None, want="text"):
    """Resolve a GGUF for ``want`` in {'text','embed','vision','mmproj'}.

    Returns a path or None. Resolution order:
      1. explicit env path for that role
      2. exact filename-stem match against the hint (deterministic)
      3. fuzzy scoring over discovered GGUFs of the right kind
    Never raises. Files of a different kind (embed vs vision vs mmproj) are
    never returned, which is what keeps a 4 GB vision GGUF from being loaded
    as the chat model.
    """
    env_map = {
        "text": ("JARVIS_LLAMACPP_MODEL", "JARVIS_LLAMACPP_TOOL_MODEL",
                 "JARVIS_LLAMACPP_CHAT_MODEL"),
        "embed": ("JARVIS_LLAMACPP_EMBED_MODEL",),
        "vision": ("JARVIS_LLAMACPP_VISION_MODEL",),
        "mmproj": ("JARVIS_LLAMACPP_MMPROJ",),
    }
    for env in env_map.get(want, ()):
        p = _env_path(env)
        if p and classify(p) == want:
            return p
        # An explicit path pointing at a differently-classified file is still
        # honoured - the operator asked for it by name.
        if p and want == "text":
            return p

    default_hint = {"text": DEFAULT_TOOL_HINT, "embed": DEFAULT_EMBED_HINT,
                    "vision": DEFAULT_VISION_HINT}.get(want, "")
    hn = " ".join([(hint or ""), default_hint]).lower()
    exact = set()
    for tok in hn.replace("/", " ").replace("\\", " ").split():
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok.endswith(".gguf"):
            exact.add(tok[:-5])
        elif "." not in tok and len(tok) > 3:
            exact.add(tok)

    best, best_score = None, -1
    for fp, kind in _all_ggufs():
        if kind != want:
            continue
        base = os.path.basename(fp).lower()
        stem = os.path.splitext(base)[0]
        score = 0
        if stem in exact:
            score += 50                      # configured model, deterministic win
        for tok in exact:
            if len(tok) > 4 and tok in base:
                score += 2
        if want == "text":
            if "qwen" in base:
                score += 3
            if "coder" in base:
                score += 3
            if "q4_k_m" in base:
                score += 2
            # Prefer fully-offloadable sizes; a 14B self-fits to a tiny
            # context on an 8GB card and is much slower, so it is never
            # auto-selected - it stays an explicit opt-in.
            try:
                size = os.path.getsize(fp)
            except Exception:
                size = 0
            if size and 2 * 1024 ** 3 < size < 6 * 1024 ** 3:
                score += 1
        if score > best_score:
            best, best_score = fp, score
    return best


def _iter_json_objects(text):
    """Yield every balanced {...} block, tolerating nested braces and strings."""
    if not text:
        return
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def _tool_call_candidates(obj):
    """(name, args) pairs from one parsed JSON object, or []. Never raises."""
    out = []
    if not isinstance(obj, dict):
        return out
    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        if fn.get("name"):
            out.append((fn["name"], fn.get("arguments", {}) or {}))
        return out
    if isinstance(obj.get("function"), str):
        out.append((obj["function"], obj.get("arguments", {}) or {}))
        return out
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if isinstance(name, str) and name:
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters", {}) or {}
        out.append((name, args))
    return out


def _normalize_tool_calls(native_calls, known=None):
    """Any tool-call shape -> the single shape server.py consumes.

    ``server.py`` reads ``tc["function"]["name"]`` and
    ``tc["function"]["arguments"]`` (arguments always a dict), so every input
    form - OpenAI ``{"function": {...}}``, bare ``{"name": ...}``, and
    JSON-encoded argument strings - is normalised here. Never raises.
    """
    out, seen = [], set()
    for name, args in (native_calls or []):
        if not name:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append({"function": {"name": name, "arguments": args}})
    return out


def _parse_toolcalls_from_text(content, known_tools=None):
    """Recover tool calls a model emitted as JSON text instead of native calls.

    Scans every balanced ``{...}`` block (so nested ``arguments`` objects
    parse correctly) and normalises each into the server's tool-call shape.
    Returns [] when nothing parses. Never raises.
    """
    if not content:
        return []
    known = set()
    try:
        for t in known_tools or []:
            fn = (t.get("function") or {}) if isinstance(t, dict) else {}
            if fn.get("name"):
                known.add(fn["name"])
    except Exception:
        pass
    raw = []
    for chunk in _iter_json_objects(content):
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        raw.extend(_tool_call_candidates(obj))
    if known:
        raw = [(n, a) for (n, a) in raw if n in known]
    return _normalize_tool_calls(raw)


class LlamaCppBackend(InferenceBackend):
    name = "llamacpp"

    def __init__(self, n_ctx=None, n_gpu_layers=None):
        super().__init__()
        try:
            self.n_ctx = int(n_ctx if n_ctx is not None else
                             os.environ.get("JARVIS_LLAMACPP_N_CTX", DEFAULT_N_CTX))
        except Exception:
            self.n_ctx = DEFAULT_N_CTX
        try:
            self.n_ctx_vision = int(os.environ.get(
                "JARVIS_LLAMACPP_N_CTX_VISION", DEFAULT_N_CTX_VISION))
        except Exception:
            self.n_ctx_vision = DEFAULT_N_CTX_VISION
        try:
            raw = (n_gpu_layers if n_gpu_layers is not None else
                   os.environ.get("JARVIS_LLAMACPP_GPU_LAYERS", DEFAULT_GPU_LAYERS))
            self.n_gpu_layers = int(raw)
        except Exception:
            self.n_gpu_layers = DEFAULT_GPU_LAYERS
        try:
            self.n_threads = int(os.environ.get("JARVIS_LLAMACPP_THREADS", "0") or 0) or None
        except Exception:
            self.n_threads = None
        # "" / "auto" keeps whatever template the GGUF embeds.
        self.chat_format = (os.environ.get(
            "JARVIS_LLAMACPP_CHAT_FORMAT", DEFAULT_CHAT_FORMAT) or "").strip()

    # -- role model resolution -------------------------------------------
    def resolve_model_file(self, role="tool"):
        role = (role or "tool").lower()
        if role == "chat":
            hint = (os.environ.get("JARVIS_LLAMACPP_CHAT_MODEL", "") or "").strip()
            return find_gguf(hint, "text")
        if role == "embed":
            return find_gguf(None, "embed")
        if role == "vision":
            return find_gguf(None, "vision")
        if role == "mmproj":
            return find_gguf(None, "mmproj")
        hint = (os.environ.get("JARVIS_LLAMACPP_TOOL_MODEL", "") or "").strip()
        return find_gguf(hint, "text")

    def label(self, role="tool"):
        """Short model label for health/metrics (file stem or hint)."""
        try:
            fp = self.resolve_model_file(role)
        except Exception:
            fp = None
        if fp:
            return os.path.splitext(os.path.basename(fp))[0][:80].lower().replace("_", "-")
        return {"chat": "chat model", "embed": "embedding model",
                "vision": "vision model"}.get(role, DEFAULT_TOOL_HINT)

    def _key_for(self, role):
        """chat and tool share ONE instance when they resolve to the same GGUF."""
        if role == "chat":
            try:
                if (self.resolve_model_file("chat") and
                        self.resolve_model_file("chat") == self.resolve_model_file("tool")):
                    return "tool"
            except Exception:
                pass
        return role

    @property
    def available(self):
        try:
            import llama_cpp  # noqa: F401
            has_pkg = True
        except Exception:
            has_pkg = False
        return has_pkg and bool(self.resolve_model_file("tool"))

    # -- chat -----------------------------------------------------------
    def chat(self, messages, model=None, tools=None, options=None, timeout=240,
             role="tool"):
        t0 = time.time()
        opts = options if isinstance(options, dict) else {}
        try:
            temperature = float(opts.get("temperature", 0.6))
        except Exception:
            temperature = 0.6
        try:
            max_tokens = int(opts.get("num_predict", 1024))
        except Exception:
            max_tokens = 1024

        key = self._key_for(role)
        model_file = self.resolve_model_file(role if key != "tool" else "tool")
        if not model_file:
            err = InferenceError(
                "Inference backend unavailable - no text GGUF configured for "
                f"role '{role}'. Set JARVIS_LLAMACPP_MODEL to a local .gguf "
                f"path (e.g. {DEFAULT_TOOL_HINT}).")
            self._record(time.time() - t0, model=self.label(role),
                         error=err, kind=role)
            raise err
        extra = {}
        if self.chat_format and self.chat_format.lower() not in ("auto", "none", ""):
            extra["chat_format"] = self.chat_format
        try:
            ModelManager._evict_conflicts(key)
            llm, load_took = ModelManager.load(
                key, model_file, n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers, n_threads=self.n_threads,
                llama_kwargs=extra)
        except InferenceError as e:
            self._record(time.time() - t0, model=self.label(role),
                         error=e, kind=role)
            raise
        except Exception as e:
            err = InferenceError(f"Inference backend unavailable ({e}).")
            self._record(time.time() - t0, model=self.label(role),
                         error=err, kind=role)
            raise err

        create_kwargs = dict(messages=messages, temperature=temperature,
                             max_tokens=max_tokens)
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        try:
            res = llm.create_chat_completion(**create_kwargs)
        except Exception as e:
            if tools:
                # A tool payload the template cannot express should degrade to
                # plain text, never fail the whole turn.
                try:
                    res = llm.create_chat_completion(
                        messages=messages, temperature=temperature,
                        max_tokens=max_tokens)
                except Exception as e2:
                    err = InferenceError(
                        f"Inference backend unavailable ({e2}).")
                    self._record(time.time() - t0, model=self.label(role),
                                 error=err, kind=role)
                    raise err
            else:
                err = InferenceError(f"Inference backend unavailable ({e}).")
                self._record(time.time() - t0, model=self.label(role),
                             error=err, kind=role)
                raise err

        latency = time.time() - t0
        try:
            choice = (res.get("choices") or [{}])[0]
            msg = (choice.get("message") or {})
            content = msg.get("content") or ""
            tool_calls = _normalize_tool_calls(
                _native_pairs(msg.get("tool_calls") or []))
            if not tool_calls and tools:
                tool_calls = _parse_toolcalls_from_text(content, tools)
            usage = res.get("usage") or {}
            tps = None
            try:
                n_tok = usage.get("completion_tokens")
                if n_tok and latency > 0:
                    tps = round(n_tok / latency, 2)
            except Exception:
                pass
            tps = tps or estimate_tps(content, latency)
            out = {"message": {"role": "assistant", "content": content,
                               "tool_calls": tool_calls},
                   "model": self.label(role), "backend": "llamacpp"}
            self._record(latency, model=self.label(role), tokens_per_s=tps,
                         load_time_s=load_took or None, kind=role)
            return out
        except InferenceError:
            raise
        except Exception as e:
            err = InferenceError(f"llama.cpp response parse failed: {e}")
            self._record(latency, model=self.label(role), error=err, kind=role)
            raise err

    # -- embeddings ------------------------------------------------------
    def embed(self, model, inputs, timeout=60, query_first=False):
        """Semantic embeddings from the nomic GGUF.

        ``query_first`` mirrors server.py's ``recall``, which sends
        ``[query, *documents]``. nomic-embed is trained with asymmetric
        prefixes, so the query gets ``search_query:`` and the documents
        ``search_document:``.
        """
        t0 = time.time()
        file = self.resolve_model_file("embed")
        if not file:
            err = InferenceError(
                "Inference backend unavailable - no embedding GGUF found. "
                "Drop nomic-embed-text-*.gguf into models\\ or set "
                "JARVIS_LLAMACPP_EMBED_MODEL.")
            self._record(time.time() - t0, model=self.label("embed"),
                         error=err, kind="embed")
            raise err
        texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
        if not texts:
            return {"embeddings": [], "model": self.label("embed"),
                    "backend": "llamacpp"}
        prefixed = []
        for i, t in enumerate(texts):
            s = str(t or "")
            if query_first and i == 0:
                s = "search_query: " + s
            elif "search_query:" not in s and "search_document:" not in s:
                s = "search_document: " + s
            prefixed.append(s)
        try:
            ModelManager.load("embed", file, n_ctx=512, n_gpu_layers=0,
                              llama_kwargs={"embedding": True})
        except InferenceError as e:
            self._record(time.time() - t0, model=self.label("embed"),
                         error=e, kind="embed")
            raise
        llm = ModelManager.get("embed")
        try:
            vecs = llm.embed(prefixed)
        except Exception as e:
            err = InferenceError(f"Embedding failed ({e}).")
            self._record(time.time() - t0, model=self.label("embed"),
                         error=err, kind="embed")
            raise err
        if vecs and not isinstance(vecs[0], (list, tuple)):
            vecs = [vecs]  # single input returns a flat vector
        out = {"embeddings": [list(v) for v in vecs],
               "model": self.label("embed"), "backend": "llamacpp"}
        self._record(time.time() - t0, model=self.label("embed"),
                     kind="embed")
        return out

    # -- vision -----------------------------------------------------------
    def vision(self, prompt, jpeg_bytes, model=None, timeout=150):
        """Describe/ground an image with the Qwen2.5-VL GGUF + mmproj."""
        t0 = time.time()
        import base64
        vl = self.resolve_model_file("vision")
        if not vl:
            err = InferenceError(
                "Inference backend unavailable - no vision GGUF found. Drop "
                "Qwen2.5-VL-*-Instruct-Q4_K_M.gguf into models\\ or set "
                "JARVIS_LLAMACPP_VISION_MODEL.")
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        mmproj = self.resolve_model_file("mmproj")
        if not mmproj:
            err = InferenceError(
                "Inference backend unavailable - vision projector (mmproj) "
                "GGUF missing. Drop mmproj-*.gguf next to the VL model.")
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        try:
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except Exception as e:
            err = InferenceError(
                f"llama-cpp-python build lacks Qwen2.5-VL support ({e}). "
                "Reinstall llama-cpp-python >= 0.3.30.")
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        try:
            # One handler for the whole fit ladder: building it loads the CLIP
            # projector, which is far too slow to repeat per attempt.
            handler = Qwen25VLChatHandler(clip_model_path=mmproj)
        except Exception as e:
            err = InferenceError(f"Vision projector failed to load ({e}).")
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        try:
            ModelManager._evict_conflicts("vision")
            llm, load_took = ModelManager.load(
                "vision", vl, n_ctx=self.n_ctx_vision,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                llama_kwargs={"chat_handler": handler})
        except InferenceError as e:
            # BUILD 18: the text model was just evicted, so if this load
            # failed the user needs the VRAM numbers, not a shrug.
            try:
                size_mb = round(os.path.getsize(vl) / (1024 ** 2))
                proj_mb = round(os.path.getsize(mmproj) / (1024 ** 2))
                free_mb = _vram_free_mb()
                need = size_mb + proj_mb
                err = InferenceError(
                    f"Vision model could not load: '{os.path.basename(vl)}' "
                    f"({size_mb}MB) + projector ({proj_mb}MB) = ~{need}MB "
                    f"needed, {free_mb:.0f}MB VRAM free. The chat model was "
                    f"evicted to make room. Detail: {e}")
            except Exception:
                err = e
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        except Exception as e:
            err = InferenceError(f"Vision backend unavailable ({e}).")
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        b64 = base64.b64encode(jpeg_bytes or b"").decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt or "Describe this image."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
        try:
            res = llm.create_chat_completion(messages=messages, max_tokens=512,
                                             temperature=0.2)
        except Exception as e:
            err = InferenceError(f"Vision inference failed ({e}).")
            self._record(time.time() - t0, model=self.label("vision"),
                         error=err, kind="vision")
            raise err
        latency = time.time() - t0
        try:
            content = ((res.get("choices") or [{}])[0].get("message") or {}).get(
                "content") or ""
        except Exception as e:
            err = InferenceError(f"Vision response parse failed ({e}).")
            self._record(latency, model=self.label("vision"), error=err,
                         kind="vision")
            raise err
        self._record(latency, model=self.label("vision"),
                     tokens_per_s=estimate_tps(content, latency),
                     load_time_s=load_took or None, kind="vision")
        return {"message": {"role": "assistant", "content": content},
                "model": self.label("vision"), "backend": "llamacpp"}

    # -- observability -----------------------------------------------------
    def health(self):
        try:
            import llama_cpp  # noqa: F401
            pkg, pkg_err = True, None
        except Exception as e:
            pkg, pkg_err = False, str(e)[:200]
        files = {r: self.resolve_model_file(r)
                 for r in ("tool", "chat", "embed", "vision", "mmproj")}
        loaded = ModelManager.loaded()
        model_loaded = bool(loaded)
        # Report ACTUAL fitted values once loaded (self-fit may shrink them).
        gpu_layers, context, fit_note = self.n_gpu_layers, self.n_ctx, None
        try:
            _meta = ModelManager.metadata("tool") or {}
            if _meta:
                gpu_layers = _meta.get("n_gpu_layers", gpu_layers)
                context = _meta.get("n_ctx", context)
                if _meta.get("fit_degraded"):
                    _req = _meta.get("requested") or {}
                    fit_note = (f"self-fit for VRAM: requested "
                                f"ctx={_req.get('n_ctx')}/layers={_req.get('n_gpu_layers')}"
                                f" -> actual ctx={context}/layers={gpu_layers}")
        except Exception:
            pass
        info = {
            "backend": "llamacpp",
            "model_loaded": model_loaded,
            "gpu_layers": gpu_layers,
            "context": context,
            "model": self.label("tool"),
            "files": files,
            "chat_model_file": files.get("chat"),
            "tool_model_file": files.get("tool"),
            "embed_model_file": files.get("embed"),
            "vision_model_file": files.get("vision"),
            "mmproj_file": files.get("mmproj"),
            "package": pkg,
            "chat_format": self.chat_format or "gguf-embedded",
            "loaded_roles": sorted(loaded.keys()),
            "vram_free_mb": _vram_free_mb(),
            "metrics": self.metric_snapshot(),
        }
        if fit_note:
            info["fit_note"] = fit_note
        if not pkg:
            info["reason"] = f"{INSTALL_HINT} ({pkg_err})"
        elif not files.get("tool"):
            info["reason"] = ("no text GGUF found - set JARVIS_LLAMACPP_MODEL "
                              f"to a local .gguf (e.g. {DEFAULT_TOOL_HINT})")
        return info

    def inventory(self):
        """Discovered GGUF inventory for /api/models and /api/health."""
        out = []
        for fp, kind in sorted(_all_ggufs(), key=lambda x: x[0].lower()):
            try:
                size = os.path.getsize(fp)
            except Exception:
                size = 0
            out.append({
                "name": os.path.splitext(os.path.basename(fp))[0],
                "file": os.path.basename(fp),
                "path": fp,
                "size": size,
                "kind": kind,
                "tools": kind == "text",
                "vision": kind == "vision",
                "embeddings": kind == "embed",
            })
        return out

    def list_models(self):
        """Model names (chat-model dropdown + tool selection)."""
        names = []
        for role in ("tool", "chat", "embed", "vision"):
            try:
                fp = self.resolve_model_file(role)
            except Exception:
                continue
            if fp:
                n = os.path.splitext(os.path.basename(fp))[0]
                if n not in names:
                    names.append(n)
        return names


def _native_pairs(native_calls):
    """OpenAI-style tool_calls -> [(name, args)] pairs. Never raises."""
    out = []
    try:
        for tc in native_calls or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = fn.get("name")
            if name:
                out.append((name, fn.get("arguments", {}) or {}))
    except Exception:
        pass
    return out
