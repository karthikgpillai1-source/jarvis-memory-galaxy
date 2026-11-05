"""Model router + Tool Brain (BUILD 18).

BUILD 18 removed Ollama entirely. There is exactly one backend -
``LlamaCppBackend`` - serving every call kind::

    chat     Qwen2.5-Coder-7B GGUF  (shared with the tool brain)
    tool     Qwen2.5-Coder-7B GGUF  (same instance, never 2x VRAM)
    embed    nomic-embed-text GGUF
    vision   Qwen2.5-VL GGUF + mmproj

``JARVIS_INFERENCE`` is retained only so a stale value in the environment
cannot break startup; it no longer selects anything. ``ollama`` and
``hybrid`` modes, the Ollama backend module, and the cross-backend
fallback contract are all gone.

Failure contract (BUILD 18, deliberately strict): if a request cannot be
served, the underlying ``InferenceError`` propagates with its real cause
- the rejected VRAM fits, the missing GGUF, the missing mmproj. There is
no silent second engine, because a silent fallback is what made the
migration invisible in the first place.
"""
import base64
import os
import threading
import time

from .inference_backend import InferenceError
from .llamacpp_backend import LlamaCppBackend

MODES = ("llamacpp",)
DEFAULT_MODE = "llamacpp"

_ROUTER = None
_ROUTER_LOCK = threading.Lock()


def current_mode():
    """Always 'llamacpp'. Kept for callers that report the mode."""
    return DEFAULT_MODE


def get_router(ollama_url=None, mode=None):
    """Process-wide router singleton (models stay loaded across requests).

    ``ollama_url`` / ``mode`` are accepted and ignored so existing call
    sites keep working after the Ollama removal.
    """
    global _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = ModelRouter()
        return _ROUTER


def reset_router():
    global _ROUTER
    with _ROUTER_LOCK:
        _ROUTER = None


class ModelRouter:
    """Single entry point for ALL model calls. Never call backends directly.

    BUILD 18: one backend, no modes. The ctor args are kept (and ignored)
    so old call sites do not break.
    """

    def __init__(self, ollama_url=None, mode=None):
        self.llamacpp = LlamaCppBackend()
        self.mode = DEFAULT_MODE
        self._lock = threading.Lock()
        self.errors = 0
        self.last_error = None
        self.last_backend = None
        self.last_kind = None
        self.last_latency_s = None
        self.last_tokens_per_s = None
        self.history = []  # last 20 routed calls (capped)

    # -- routing ---------------------------------------------------------
    def set_mode(self, mode):
        """No-op: there is only one mode now. Returns True if it was already."""
        return mode in MODES

    def _primary(self, kind):
        """The single backend. Retained as a helper so call sites read the same."""
        return self.llamacpp

    def _note(self, backend_name, kind, latency_s, tps, ok=True, err=None):
        with self._lock:
            self.last_backend = backend_name
            self.last_kind = kind
            self.last_latency_s = latency_s
            self.last_tokens_per_s = tps
            entry = {"t": time.strftime("%H:%M:%S"), "backend": backend_name,
                     "kind": kind, "latency_s": latency_s,
                     "tokens_per_s": tps, "ok": bool(ok),
                     "error": (str(err)[:200] if err else None)}
            self.history.append(entry)
            del self.history[:-20]
            if not ok:
                self.errors += 1
                self.last_error = str(err)[:300]
            return entry

    # -- unified calls ----------------------------------------------------
    def chat(self, messages, model=None, tools=None, options=None,
             timeout=240, kind="chat"):
        """Route a chat/completion call. Returns the backend response dict.

        ``kind``: "chat" (personality) or "tool" (Tool Brain reasoning).
        Raises only when EVERY backend fails.
        """
        kind = (kind or "chat").lower()
        if kind not in ("chat", "tool"):
            kind = "chat"
        t0 = time.time()
        try:
            res = self.llamacpp.chat(messages, model or "default", tools=tools,
                                     options=options, timeout=timeout,
                                     role="tool" if kind == "tool" else "chat")
            self._note(self.llamacpp.name, kind, round(time.time() - t0, 3),
                       self._tps_of(self.llamacpp), True)
            return res
        except Exception as e:
            # BUILD 18: no second backend. Record the real cause and let it
            # propagate so the user sees WHY, not a generic failure.
            self._note("llamacpp", kind, round(time.time() - t0, 3), None,
                       False, e)
            raise

    def embed(self, model, inputs, timeout=60, query_first=False):
        """Embeddings from the nomic GGUF via llama.cpp.

        ``query_first`` mirrors server.py's ``recall``, which sends
        ``[query, *documents]``; nomic-embed is trained with asymmetric
        prefixes so the backend needs to know which slot is the query.
        """
        t0 = time.time()
        try:
            res = self.llamacpp.embed(model, inputs, timeout=timeout,
                                      query_first=query_first)
            self._note("llamacpp", "embed", round(time.time() - t0, 3),
                       self._tps_of(self.llamacpp), True)
            return res
        except Exception as e:
            self._note("llamacpp", "embed", round(time.time() - t0, 3), None,
                       False, e)
            raise
            if primary is self.llamacpp:
                res = self.llamacpp.embed(model, inputs, timeout=timeout,
                                          query_first=query_first)
            self._note("llamacpp", "embed", round(time.time() - t0, 3),
                       self._tps_of(self.llamacpp), True)
            return res
        except Exception as e:
            self._note("llamacpp", "embed", round(time.time() - t0, 3), None,
                       False, e)
            raise

    def vision(self, prompt, jpeg_bytes, model=None, timeout=150):
        """Vision from the Qwen2.5-VL GGUF + mmproj via llama.cpp.

        BUILD 18: the text model is evicted from VRAM automatically, then
        the VL model loads. If every fit is refused, the InferenceError
        names the rejected fits and the free VRAM - it is surfaced to the
        user rather than silently retried elsewhere.
        """
        t0 = time.time()
        try:
            res = self.llamacpp.vision(prompt, jpeg_bytes, model=model,
                                       timeout=timeout)
            self._note("llamacpp", "vision", round(time.time() - t0, 3),
                       self._tps_of(self.llamacpp), True)
            return res
        except Exception as e:
            self._note("llamacpp", "vision", round(time.time() - t0, 3), None,
                       False, e)
            raise

    def chat_with_image(self, prompt, jpeg_bytes, model=None, tools=None,
                        timeout=150):
        """Image turn. BUILD 18: always the llama.cpp VL path."""
        return self.vision(prompt, jpeg_bytes, model=model, timeout=timeout)

    def list_models(self):
        """GGUF model names for the chat dropdown. BUILD 18: GGUF only."""
        try:
            return list(self.llamacpp.list_models() or [])
        except Exception:
            return []

    # -- observability ------------------------------------------------------
    @staticmethod
    def _tps_of(backend):
        try:
            return backend.metrics.get("last_tokens_per_s")
        except Exception:
            return None

    def take_fallback_notice(self):
        """BUILD 18: no fallback exists, so this is always None.

        Kept so server.py's call site does not need changing.
        """
        return None

    def health(self):
        try:
            cpp = self.llamacpp.health()
        except Exception as e:
            cpp = {"backend": "llamacpp", "model_loaded": False,
                   "error": str(e)[:200]}
        with self._lock:
            recent = list(self.history[-8:])
            errors = self.errors
            last_err = self.last_error
            mode = self.mode
            last = {"backend": self.last_backend, "kind": self.last_kind,
                    "latency_s": self.last_latency_s,
                    "tokens_per_s": self.last_tokens_per_s}
        # Spec §Health Endpoint top-level keys (llamacpp-flavoured view):
        spec_view = {"backend": cpp.get("backend", "llamacpp"),
                     "model_loaded": bool(cpp.get("model_loaded", False)),
                     "gpu_layers": cpp.get("gpu_layers",
                                           self.llamacpp.n_gpu_layers),
                     "context": cpp.get("context", self.llamacpp.n_ctx),
                     "model": cpp.get("model", self.llamacpp.label("tool"))}
        return {**spec_view,
                "inference_mode": mode,
                "llamacpp": cpp,
                "tool_brain": ToolBrain.status(self),
                "last_call": last,
                "errors": errors,
                "last_error": last_err,
                "recent_calls": recent,
                "llamacpp_metrics": self.llamacpp.metric_snapshot()}


class ToolBrain:
    """Dedicated tool-reasoning layer (BUILD 15, retargeted in 18).

    Pipeline::

        User -> Chat Model -> ToolBrain (Qwen2.5-Coder) -> OS Tool

    The chat/personality model never executes OS tools directly: every
    OS-capable turn is reasoned by the tool backend (the resident
    Qwen2.5-Coder GGUF), and every OS tool result is validated here —
    failures and contradictions produce a retry nudge instead of a
    narrated lie.
    """

    TOOL_SYSTEM_SUFFIX = (
        "TOOL BRAIN: you are the dedicated tool-reasoning layer "
        "(Qwen2.5-Coder). Responsibilities: (1) choose the MINIMUM tool calls "
        "that fulfil the order, one per turn; (2) validate each tool result "
        "before proceeding; (3) decide the next action from verified results "
        "only; (4) detect failures/contradictions and retry differently "
        "(never narrate an unexecuted step); (5) stop and report honestly "
        "when blocked (logins, ambiguity, hard failure). "
        "The chat model never touches OS tools — every OS action is yours."
    )

    # Result substrings that mean "this step HARD-failed" (soft notes like
    # "unverified / no new window" are routine verify language, not failures —
    # flagging those would spam retry nudges on healthy turns).
    _FAIL_MARKS = ("failed", "failure", "refused", "blocked",
                   "could not", "cannot", "error", "exception", "timed out",
                   "timeout", "verification failed", "could not ground")

    @classmethod
    def system_message(cls, tool_model_name):
        return {"role": "system",
                "content": (f"TOOL ROUTER: this turn needs computer/code tools "
                            f"— reasoning engine is {tool_model_name}. "
                            + cls.TOOL_SYSTEM_SUFFIX)}

    @classmethod
    def validate_tool_result(cls, name, result):
        """Return (ok: bool, note: str). Heuristic, never raises."""
        try:
            text = str(result or "").lower()
            if not text.strip():
                return False, "empty tool result — retry the call"
            for mark in cls._FAIL_MARKS:
                if mark in text:
                    return False, (f"tool '{name}' reported possible failure "
                                   f"({mark}) — verify state, then retry or "
                                   "use an alternative tool")
            return True, ""
        except Exception:
            return True, ""

    @classmethod
    def failure_nudge(cls, name, result):
        """System nudge appended after a failed tool result. Never raises."""
        try:
            ok, note = cls.validate_tool_result(name, result)
            if ok:
                return None
            return ("TOOL BRAIN VALIDATION: the last tool result indicates "
                    f"failure ({note}). Do NOT narrate success. Either retry "
                    f"'{name}' with corrected arguments, pick an alternative "
                    "tool, or report the blocker honestly. Emit ONE tool call "
                    "or a truthful status — no prose about unexecuted steps.")
        except Exception:
            return None

    @classmethod
    def status(cls, router=None):
        """Compact Tool Brain status for /api/health. Never raises."""
        try:
            mode = router.mode if router else current_mode()
            label = None
            if router is not None:
                try:
                    label = router.llamacpp.label("tool")
                except Exception:
                    label = None
            return {"backend": "llamacpp", "mode": mode,
                    "tool_model": label or "qwen2.5-coder-7b-instruct-q4_k_m",
                    "policy": "chat model never executes OS tools directly; "
                              "all OS turns reason via the tool backend with "
                              "result validation + retry"}
        except Exception as e:
            return {"backend": "llamacpp", "error": str(e)[:200]}
