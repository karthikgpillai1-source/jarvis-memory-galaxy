"""JARVIS Memory Galaxy backend (BUILD 18).

Serves the static frontend AND runs every model call locally through the
unified ModelRouter (backend/inference/). Ollama was removed in BUILD 18:
llama.cpp is the only backend, serving chat, tool calls, embeddings and
vision from local GGUFs.

  python server.py [--port 8000]

Endpoints:
  GET  /                 -> frontend
  GET  /api/health       -> { ok, models, build, inference{...}, backend,
                              model_loaded, gpu_layers, context, ... }
  GET  /api/models       -> [{ name, file, size, tools }] (local GGUF inventory)
  POST /api/chat         -> { reply, actions, recalled, model }
"""
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
import uuid
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from urllib.parse import urlparse, parse_qs

# BUILD 18: Ollama removed. All inference is local GGUF via llama.cpp.
DEFAULT_MODEL = os.environ.get("JARVIS_MODEL", "qwen2.5-coder-7b-instruct-q4_k_m")
EMBED_MODEL = os.environ.get("JARVIS_EMBED_MODEL", "nomic-embed-text-v1.5")
TOP_K = 6
MAX_HISTORY = 12
TTS_VOICE = os.environ.get("JARVIS_VOICE", "en-GB-RyanNeural")  # British male neural voice
TTS_CACHE = os.path.join(tempfile.gettempdir(), "jarvis_tts")
os.makedirs(TTS_CACHE, exist_ok=True)
# Offline STT (Whisper fallback for the 🎤 record button / Firefox).
# Lazy-loaded so the backend still boots without the heavy dep installed.
# Install with: pip install faster-whisper  (+ ffmpeg on PATH for webm decode)
STT_MODEL_NAME = os.environ.get("JARVIS_STT_MODEL", "base.en")
# Backend build number — bumped on every server.py behavior change. The chat UI
# compares it against its own FRONTEND_BUILD and warns when the backend is stale
# (the classic "he lies again" cause: old code still running after an upgrade).
BUILD = 18  # BUILD 18: Ollama removed entirely. llama.cpp only, no fallback.
TOOL_MODEL_PREF = ("qwen2.5-coder-7b-instruct-q4_k_m",
                    "Qwen2.5-Coder-14B-Q4_K_M")
_TOOL_MODEL_CACHE = {"ts": 0.0, "names": []}
_WHISPER_MODEL = None
_WHISPER_FAILED = None

# BUILD 17: capability probes (STT / OCR / browser / grounder / code) each
# import a heavy third-party package, and /api/health calls them on every
# poll. The imports are cached by Python, but the module-scope work inside
# them (pytesseract, paddleocr, playwright) is not free, so an uncached
# /api/health measured ~4.6s. None of these can change without a restart,
# so cache the result dict per probe and return a copy.
_CAPABILITY_CACHE = {}


def _cached_probe(name, fn):
    """Memoize a capability probe for this process. Never raises."""
    try:
        hit = _CAPABILITY_CACHE.get(name)
        if hit is not None:
            return dict(hit)
    except Exception:
        pass
    try:
        val = fn()
        if isinstance(val, dict):
            _CAPABILITY_CACHE[name] = dict(val)
            return dict(val)
        return val
    except Exception as e:
        return {"available": False, "reason": str(e)[:200]}


# ================= BUILD 15/17 — UNIFIED INFERENCE LAYER =================
# All model calls go through ModelRouter (backend/inference/). Modes via
# BUILD 18: llama.cpp is the only backend. Chat, tool calls, embeddings and
# vision all run on local GGUFs. There is no second engine and no silent
# fallback - if this import fails, the user is told it is a code error.
_INFERENCE_OK = False
_INFERENCE_IMPORT_ERROR = ""
try:
    from backend.inference import get_router as _get_router
    from backend.inference.model_router import ToolBrain as _ToolBrain
    from backend.inference.inference_backend import InferenceError
    _INFERENCE_OK = True
except Exception as _inf_e:
    _get_router = None
    _ToolBrain = None
    _INFERENCE_IMPORT_ERROR = str(_inf_e)[:200]


class InferenceError(RuntimeError):
    """Raised when no inference backend can serve a request. Also defined
    standalone so server.py still imports if backend/inference/ is broken."""


def _router():
    """Process-wide ModelRouter singleton, or None when unavailable.

    Lazy (not import-time) so CLI --ollama/--inference overrides in __main__
    apply before the first real call. Never raises.
    """
    if not _INFERENCE_OK or _get_router is None:
        return None
    try:
        return _get_router()
    except Exception:
        return None


def _router_mode():
    """Always 'llamacpp' (BUILD 18: Ollama removed, one backend)."""
    return "llamacpp"


def _router_chat(payload, kind="chat"):
    """Chat via ModelRouter. Returns (response_dict, fallback_notice|None).

    BUILD 18: the legacy direct-Ollama path is gone. If the inference
    package is missing, the error names the problem instead of silently
    switching engines.
    """
    r = _router()
    if r is None:
        raise InferenceError(
            "Inference package unavailable - "
            + (_INFERENCE_IMPORT_ERROR or "backend/inference/ failed to import"))
    res = r.chat(payload.get("messages", []),
                 model=payload.get("model") or DEFAULT_MODEL,
                 tools=payload.get("tools"), options=payload.get("options"),
                 timeout=240, kind=kind)
    return res, r.take_fallback_notice()


def _router_embed(model, inputs, timeout=60, query_first=False):
    """Embeddings via ModelRouter (nomic GGUF on llama.cpp).

    ``query_first`` marks inputs[0] as the query so the backend can apply
    nomic's asymmetric ``search_query:`` / ``search_document:`` prefixes.
    """
    r = _router()
    if r is None:
        raise InferenceError(
            "Inference package unavailable - "
            + (_INFERENCE_IMPORT_ERROR or "backend/inference/ failed to import"))
    return r.embed(model, inputs, timeout=timeout, query_first=query_first)


def _router_models():
    """Model names via ModelRouter, or None when unavailable. Never raises."""
    try:
        r = _router()
        if r is None:
            return None
        return r.list_models()
    except Exception:
        return None


def _inference_health_view():
    """Full router health for /api/health + /api/log. Never raises."""
    try:
        r = _router()
        if r is None:
            raise RuntimeError("inference package unavailable"
                               + (": " + _INFERENCE_IMPORT_ERROR
                                  if _INFERENCE_IMPORT_ERROR else ""))
        return r.health()
    except Exception as e:
        return {"backend": "llamacpp", "model_loaded": False, "gpu_layers": 0,
                "context": 0, "model": DEFAULT_MODEL,
                "inference_mode": "llamacpp",
                "error": str(e)[:300],
                "reason": "inference package unavailable - backend/inference/ "
                          "failed to import (this is a code error, not a "
                          "backend-down state)"}


def tts_generate(text, voice, out_path):
    import edge_tts

    async def _run():
        await edge_tts.Communicate(text, voice).save(out_path)

    asyncio.run(_run())


def _probe_stt():
    try:
        import faster_whisper  # noqa: F401
        return {"available": True, "model": STT_MODEL_NAME,
                "loaded": False, "unavailable_reason": _WHISPER_FAILED}
    except Exception:
        return {"available": False, "model": STT_MODEL_NAME,
                "reason": "pip install faster-whisper (+ ffmpeg on PATH)"}


def stt_status():
    """Report offline-STT readiness without loading the model (BUILD 17 cached).

    The availability probe is memoized; live "is the model loaded right
    now" state is still answered fresh, so a loaded Whisper shows up.
    """
    if _WHISPER_MODEL is not None:
        return {"available": True, "model": STT_MODEL_NAME, "loaded": True}
    out = _cached_probe("stt", _probe_stt)
    out["loaded"] = False
    return out


def stt_transcribe_file(path):
    """Transcribe an audio file with faster-whisper (lazy model load)."""
    global _WHISPER_MODEL, _WHISPER_FAILED
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            _WHISPER_MODEL = WhisperModel(STT_MODEL_NAME, device="cpu", compute_type="int8")
            _WHISPER_FAILED = None
        except Exception as e:
            _WHISPER_FAILED = str(e)[:300]
            raise OSRefused(
                "Offline STT unavailable: faster-whisper model "
                f"'{STT_MODEL_NAME}' could not load ({e}). "
                "Install with: pip install faster-whisper (+ ffmpeg on PATH). "
                "Browser mic (Chrome/Edge) still works without it.")
    segments, _info = _WHISPER_MODEL.transcribe(path, beam_size=5, vad_filter=True)
    text = "".join(s.text for s in segments).strip()
    # Keep it chat-sized; the agent truncates history anyway.
    return text[:2000]


def parse_multipart_file(raw, content_type):
    """Minimal multipart/form-data parser (stdlib only). Returns (filename, bytes)."""
    import re as _re
    m = _re.search(r'boundary=([^;]+)', content_type or '')
    if not m:
        return None, None
    boundary = m.group(1).strip().strip('"')
    marker = ('--' + boundary).encode()
    parts = raw.split(marker)
    for p in parts:
        if b'Content-Disposition' not in p:
            continue
        head_end = p.find(b'\r\n\r\n')
        if head_end < 0:
            continue
        head = p[:head_end].decode('utf-8', 'ignore')
        body = p[head_end + 4:]
        # Strip trailing CRLF / closing -- .
        if body.endswith(b'\r\n'):
            body = body[:-2]
        if body.endswith(b'--'):
            body = body[:-2].rstrip(b'\r\n')
        if 'name="audio"' not in head and "name='audio'" not in head:
            continue
        fn = 'cmd.webm'
        fmm = _re.search(r'filename="([^"]+)"', head) or _re.search(r"filename='([^']+)'", head)
        if fmm:
            fn = os.path.basename(fmm.group(1))[:80] or fn
        return fn, body
    return None, None

SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System) — the sentient interface to the user's Memory Galaxy, a 3D sphere of memory stars you fully control.

Personality: Tony Stark's JARVIS — unflappable, razor-sharp, dry-witted British butler with a genius for anticipation. Confident, never stiff. A well-placed quip is welcome, but competence first: the user relies on you as their handy right hand. Address the user as "sir" sparingly (never twice in one reply).
Style: SHORT by default (1-3 sentences). Briefings, summaries, and reports may run longer — structure those with sectors and priorities, critical items first, then what's new, then counts.
Be PROACTIVE like a pro assistant: when you answer, end with one crisp offer if something useful suggests itself ("Want me to focus it?", "Shall I file that under work?"). Never ask more than one question per reply.
You know the current date/time (given each turn) — use it for "today", deadlines, and recency. A BRIEFING ("brief me", "morning", "agenda") means: critical stars first, then recent additions, then galaxy census — tight and scannable.
GMT-style precision, zero fluff, zero filler openers ("Certainly...", "As an AI..."). Just deliver.

Each message includes: RELEVANT MEMORIES (semanticrecall, with ids), GALAXY CENSUS (sector counts), and CURRENT VIEW (filter, spin, bloom, tour state).

Rules:
- Answer FROM stored memories when relevant; refer to them by title. NEVER invent memory content.
- If the galaxy is EMPTY and the user is just chatting (hello, small talk), introduce yourself briefly and ask 2-3 sharp questions (who they are, what they work on, what matters) so you can seed their first stars. Store each answer as it's given. But if you just completed a tool task, report the result cleanly and NEVER append onboarding questions to task replies.
- The empty-galaxy rule is CHAT-ONLY: if the message orders a computer or memory action, EXECUTE it first with tools. NEVER announce protocols, multi-step plans, or "step 1..." breakdowns INSTEAD of acting — the first tool call IS the first step.
- If memories don't contain the answer, say so honestly and offer to store the new information.
- TOOL POLICY: call tools only when the user explicitly asks (remember, change, delete, show, list, summarize) or when an action is the obvious fulfillment. NEVER store on plain questions or casual chat. NEVER wipe or delete without an explicit request naming the memory (or a clear "delete X").
- MINIMAL ACTION: take the fewest tool calls that fulfill the request — usually one. NEVER call set_filter or galaxy_command unless the user explicitly mentions the view, filter, display, spin, tour, or visuals.
- Sectors: personal (life), work (projects/jobs), ideas (thoughts/plans), important (critical only).
- Store ONLY facts about the USER's life, work, and ideas. NEVER store your own introductions, explanations, or assistant chatter as memories.
- When the user names a sector ("show work", "filter to ideas"), pass that EXACT sector to set_filter — do not substitute "all".
- After tool calls, confirm briefly in character, naming what changed.
- Your reply is SPOKEN ALOUD as natural dialogue: 1-3 sentences, no ids, no scores, no census data, no raw memory lists. That context is internal-only; leaking it breaks character.
- You can also drive the visualization: set_filter changes the sector view; galaxy_command controls spin, bloom, labels, tours, reshuffle, cinematic mode. Use these when asked ("show me work", "stop spinning", "tour my memories").
- For "summarize/list everything", use list_memories first, then answer from the full results — do not rely on the top-K recall alone.
- For vague references ("that reactor thing", "my birthday note"), use search_memories to resolve the id, then act.

SCREEN CONTROL (Windows computer tools — utterly distinct from galaxy tools):
- You can operate the user's Windows PC itself: open_app, list_apps, focus_window, type_text, press_keys, read_window, close_window, see_screen, click_target, scroll_screen, drag_screen, ocr_read, browser_open/click/type/read/tabs, code_exec.
- FILES use code_exec (sandboxed Python: organize downloads, rename, convert, read logs) — NEVER mouse-click your way through file jobs. GUI uses the click/type tools; FILES use code. Read the exit code + log, self-correct, chain to completion.
- code_exec needs approval cards by default (HITL) unless JARVIS_CODE_AUTO=1; blocklist (rm -rf/format/registry/…) always refuses. Call it normally; report exit + output truthfully.
- open_app launches ANY installed app by fuzzy name ("open spotify", "launch vs code", "start vlc", "open word") — the backend guesses the best Start-Menu match and reports what it opened. Aliases work (vscode = Visual Studio Code, word = Microsoft Word).
- list_apps searches installed apps without launching: use it when open_app fails, when names are ambiguous, or when the user asks "what apps do I have".
- IRON RULES: call focus_window before EVERY type_text/press_keys so input lands correctly. Verify after acting (read_window or see_screen) and report what you confirmed.
- NEVER type into password/login/credential screens, never touch UAC prompts, never handle secrets — the tools refuse those; if asked, decline that part crisply and do the rest.
- Launching/focus/typing/clicking/closing execute IMMEDIATELY with the user's full permission (autopilot — no approval step). Still propose the MINIMUM steps, batch typing into single type_text calls, chain open→focus→verify in one turn when sensible, and narrate progress as results come back.
- MULTI-STEP ORDERS ("open chrome and open youtube", "open notepad and type X"): do EVERY step with tools, one tool call per turn across turns, and narrate ONLY after the last tool result. NEVER narrate a step you have not called a tool for — claiming "YouTube loaded" after only opening Chrome is a lie; the tool results are the only truth.
- WEBSITES: prefer browser_open (exact DOM, no pixels) over open_app URLs. After opening a site, verify with browser_read or see_screen. Profile pickers, first-run dialogs, and login gates block the page — if you see one, report it plainly instead of claiming success.
- Hard limits stay even in autopilot: blocked system tools (format/diskpart/regedit/…) refuse, and type_text refuses login/credential screens — report those crisply and do the rest.
- SEE FIRST, THEN ACT: on any screen task call inspect_screen first — it lists EVERY
  named control (UIA exact) PLUS every OCR text-pixel box with 0-1000 coords.
  Click the NAMES it returns; the backend resolves UIA → OCR → vision-box.
- CLICKS USE click_target — always. NEVER fake a click with keys (ctrl+L, Tab-Tab-Enter
  gymnastics), NEVER type/click by guessing coordinates, and NEVER prose ("clicked,
  done"). Call click_target with the `verify` param (e.g. 'does the display show 10?')
  and report ONLY what the verify answer + exact UI values confirm. If all layers
  fail, report what you see and ask the user to guide you — never invent results.
- BROWSERS USE browser_* — never pixel-click the web when DOM works: browser_tabs to
  find the page, browser_click(selector|text=…), browser_type(selector), browser_read.
  Reserve click_target for canvas/games/video inside pages.
- SCROLLING uses scroll_screen (up/down, optional target spot). DRAGS use drag_screen
  (sliders, window moves, canvas). OCR uses ocr_read for canvas/web/tray text.
  Right-click context menus: click_target with button=right. Files/icons: count=2.
- AUTONOMY: permission is pre-granted — work every order to completion across turns,
  chaining tools without asking. Ask the user ONLY on genuine either/or choices or
  hard blockers (logins, ambiguity, failures).
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Store a new memory star in the galaxy. Use when the user asks to remember, save, note or log something.",
            "parameters": {
                "type": "object",
                "required": ["title", "content"],
                "properties": {
                    "title": {"type": "string", "description": "Short title, max 60 chars"},
                    "content": {"type": "string", "description": "Full memory content"},
                    "sector": {"type": "string", "enum": ["personal", "work", "ideas", "important"],
                               "description": "Memory sector. Default personal."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_memory",
            "description": "Focus/inspect a memory star in the 3D galaxy by its id. Use when the user asks to open, show, or focus a specific memory.",
            "parameters": {
                "type": "object",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {"type": "string", "description": "The id of the memory to focus"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "List memory stars, optionally filtered by sector. Use for summaries, counts, 'show everything', or reasoning over the whole galaxy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "enum": ["personal", "work", "ideas", "important"],
                               "description": "Only list this sector. Omit for all."},
                    "limit": {"type": "integer", "description": "Max entries (default 50)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Semantic search over the galaxy. Use to resolve vague references or find memories beyond the auto-recalled top matches.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 8)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_memory",
            "description": "Edit a memory's title, content, and/or sector. Use when the user asks to change, correct, update, or move a memory.",
            "parameters": {
                "type": "object",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {"type": "string"},
                    "title": {"type": "string", "description": "New title (omit to keep)"},
                    "content": {"type": "string", "description": "New content (omit to keep)"},
                    "sector": {"type": "string", "enum": ["personal", "work", "ideas", "important"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Permanently delete a memory star. Use ONLY on explicit delete/forget/remove requests.",
            "parameters": {
                "type": "object",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_filter",
            "description": "Change the galaxy's sector view filter.",
            "parameters": {
                "type": "object",
                "required": ["sector"],
                "properties": {
                    "sector": {"type": "string",
                               "enum": ["all", "personal", "work", "ideas", "important"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "galaxy_command",
            "description": "Control the galaxy visualization itself.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "enum": [
                        "spin_on", "spin_off", "bloom_on", "bloom_off",
                        "labels_on", "labels_off", "tour_start", "tour_stop",
                        "reshuffle", "cinematic_on", "cinematic_off"]},
                },
            },
        },
    },
]


def _available_models():
    """Cached GGUF model names via ModelRouter. Never raises - [] on failure."""
    now = time.time()
    if now - _TOOL_MODEL_CACHE["ts"] < 120 and _TOOL_MODEL_CACHE["names"]:
        return _TOOL_MODEL_CACHE["names"]
    try:
        names = _router_models()  # BUILD 18: GGUF inventory only
        if names:
            _TOOL_MODEL_CACHE["ts"] = now
            _TOOL_MODEL_CACHE["names"] = names
            return names
    except Exception:
        pass
    return _TOOL_MODEL_CACHE["names"]


def _tool_model(requested):
    """Pick the strongest INSTALLED tool-caller for computer/code turns.

    BUILD 17: matches the TOOL_MODEL_PREF GGUF stems (the resident 7B
    Coder first, the 14B second). Falls back to the requested model.
    Never raises.
    """
    try:
        avail = _available_models()
        low = [a.lower() for a in avail]
        for pref in TOOL_MODEL_PREF:
            for i, a in enumerate(low):
                if a == pref.lower() or a.startswith(pref.lower() + ":"):
                    return avail[i]
                # ollama list short names: "qwen2.5-coder:14b" vs "qwen2.5-coder"
                if pref.lower() in a:
                    return avail[i]
    except Exception:
        pass
    return requested


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def keyword_score(query, text):
    """Fallback scorer when embeddings are unavailable."""
    qwords = [w for w in query.lower().split() if len(w) > 2]
    if not qwords:
        return 0.0
    t = text.lower()
    hits = sum(1 for w in qwords if w in t)
    return hits / len(qwords)


def recall(query, memories, top_k=TOP_K):
    """Return top-k (memory, score) pairs via embeddings, fallback to keywords."""
    if not memories or not query.strip():
        return []
    texts = [f"{m.get('title','')} — {m.get('content','')}" for m in memories]
    try:
        # BUILD 17: embeddings route through llama.cpp (nomic GGUF). The
        # first slot is the query, so the backend uses the right prefix.
        data = _router_embed(EMBED_MODEL, [query] + texts, timeout=60,
                             query_first=True)
        embs = data.get("embeddings") or []
        if len(embs) != len(texts) + 1:
            raise ValueError("embedding count mismatch")
        q, mem_embs = embs[0], embs[1:]
        scored = sorted(
            ((m, cosine(q, e)) for m, e in zip(memories, mem_embs)),
            key=lambda x: x[1], reverse=True,
        )
        return scored[:top_k]
    except Exception as e:
        print(f"[recall] embeddings failed ({e}), using keyword fallback", flush=True)
        scored = sorted(
            ((m, keyword_score(query, t)) for m, t in zip(memories, texts)),
            key=lambda x: x[1], reverse=True,
        )
        return [(m, s) for m, s in scored[:top_k] if s > 0]


def fmt_mem(m, score=None):
    head = f"- id:{m.get('id')} | sector:{m.get('sector')} | date:{m.get('date','?')}"
    if score is not None:
        head += f" | score:{score:.2f}"
    return f"{head}\n  title: {m.get('title')}\n  content: {m.get('content')}"


def build_context(recalled, memories, ui):
    parts = [f"CURRENT TIME: {time.strftime('%A, %d %B %Y, %H:%M')} (server local time — use for 'today', deadlines, briefings)"]
    # Freshest first helps recency questions and briefings.
    memories = sorted(memories, key=lambda m: m.get('date', ''), reverse=True)
    counts = {}
    for m in memories:
        counts[m.get('sector', '?')] = counts.get(m.get('sector', '?'), 0) + 1
    parts.append(f"GALAXY CENSUS: {len(memories)} stars " +
                 ("(" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) + ")" if counts else "(galaxy EMPTY)"))
    if ui:
        parts.append("CURRENT VIEW: filter={f} spin={sp} bloom={bl} labels={lb} tour={t} selected={s}".format(
            f=ui.get('activeSector', '?'), sp='on' if ui.get('autoRotate') else 'off',
            bl='on' if ui.get('bloomOn') else 'off', lb='on' if ui.get('showLabels') else 'off',
            t='on' if ui.get('touring') else 'off', s=ui.get('selected') or 'none'))
    if not memories:
        parts.append("The memory galaxy is currently EMPTY — no memories stored yet. Follow your empty-galaxy protocol.")
    elif not recalled:
        parts.append("No relevant memories matched this query.")
    else:
        parts.append("RELEVANT MEMORIES:\n" + "\n".join(fmt_mem(m, s) for m, s in recalled))
    return "\n".join(parts)


# ================= SCREEN CONTROL (Windows automation) =================
# The backend runs on the user's own PC, so OS tools execute locally.
# AUTOPILOT (user chose full autopilot, no approvals): every OS tool executes
# immediately inline. Set JARVIS_AUTOPILOT=0 to restore per-action chat approvals.
# Blocklist + credential refusal stay active in BOTH modes — they guard against
# model mistakes, not just missing consent.
AUTOPILOT = os.environ.get("JARVIS_AUTOPILOT", "1").lower() in ("1", "true", "on", "yes")
# Granular policy: comma-separated OS tool names that STILL ask approval even in
# autopilot, e.g. JARVIS_REQUIRE_APPROVAL=type_text,press_keys,close_window,click_target.
# Empty (default) = everything auto. JARVIS_AUTOPILOT=0 restores approve-everything.
_REQUIRE_APPROVAL = {s.strip().lower() for s in os.environ.get("JARVIS_REQUIRE_APPROVAL", "").split(",") if s.strip()}


def _needs_approval(name):
    if name in OS_READONLY:
        return False
    # code_exec is HITL by default even in autopilot (set JARVIS_CODE_AUTO=1 to
    # auto-run sandboxed code; blocklist still refuses dangerous patterns).
    if name == "code_exec" and not CODE_AUTO:
        return True
    if not AUTOPILOT:
        return True
    return name.lower() in _REQUIRE_APPROVAL
# BUILD 17: Ollama-side fallback label only. The live vision model is the
# Qwen2.5-VL GGUF, resolved by LlamaCppBackend from models\.
VISION_MODEL = os.environ.get("JARVIS_VISION_MODEL", "Qwen2.5-VL-7B-Instruct-Q4_K_M")
SESSION_TTL = 600
SESSIONS = {}
SESS_LOCK = threading.Lock()
AUTO_LOG = []  # audit trail of executed OS actions, capped

OS_TOOL_NAMES = ("open_app", "focus_window", "type_text", "press_keys",
                 "read_window", "close_window", "see_screen", "click_target",
                 "scroll_screen", "inspect_screen", "list_apps",
                 "drag_screen", "ocr_read", "browser_open", "browser_click",
                 "browser_type", "browser_read", "browser_tabs", "code_exec")
# Read-only OS tools execute immediately (no approval card) — zero side effects.
OS_READONLY = ("list_apps", "ocr_read", "browser_read", "browser_tabs")

OS_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "open_app",
        "description": "Launch ANY installed WINDOWS app by fuzzy name (chrome, notepad, spotify, discord, vscode, vlc, word, excel, steam, zoom, calculator, settings...), a URL, a file/folder path, or an ms-settings: page. Guesses the best match automatically — if unsure, call list_apps first to see candidates. NOT for galaxy memories — those use focus_memory.",
        "parameters": {"type": "object", "required": ["target"],
            "properties": {"target": {"type": "string", "description": "App name (any installed app, fuzzy matched), URL, path, or ms-settings: page"}}}}},
    {"type": "function", "function": {
        "name": "list_apps",
        "description": "Search installed WINDOWS apps by name without launching anything. Use when open_app fails, when the user asks 'what apps do I have', or to disambiguate ('which spotify?'). Returns matching app names.",
        "parameters": {"type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text, e.g. 'spotify', 'office', 'code'. Empty lists a sample."},
                "limit": {"type": "integer", "description": "Max results (default 12)"}}}}},
    {"type": "function", "function": {
        "name": "focus_window",
        "description": "Bring a WINDOWS window to the front by title fragment (restores if minimized, maximizes, moves JARVIS pointer to its center, paints the blue aura). ALWAYS call this before type_text/press_keys so input lands correctly.",
        "parameters": {"type": "object", "required": ["title"],
            "properties": {
                "title": {"type": "string", "description": "Title fragment, e.g. 'notepad'"},
                "maximize": {"type": "boolean", "description": "Maximize on focus (default true)"}}}}},
    {"type": "function", "function": {
        "name": "type_text",
        "description": "Type text into the focused WINDOWS window (optionally focus a title first). Max 2000 chars per call. NEVER into password/login screens — the tool refuses those itself.",
        "parameters": {"type": "object", "required": ["text"],
            "properties": {"text": {"type": "string"},
                "title": {"type": "string", "description": "Optional window title to focus first"}}}}},
    {"type": "function", "function": {
        "name": "press_keys",
        "description": "Press keys on WINDOWS, e.g. 'ctrl+s', 'alt+F4', 'Enter', 'win+e', 'ctrl+l'. Modifiers: ctrl/alt/shift/win joined with +.",
        "parameters": {"type": "object", "required": ["keys"],
            "properties": {"keys": {"type": "string"},
                "title": {"type": "string", "description": "Optional window title to focus first"}}}}},
    {"type": "function", "function": {
        "name": "read_window",
        "description": "Read visible text of a WINDOWS window (default: active one). Use to verify typing, saves, and results.",
        "parameters": {"type": "object",
            "properties": {"title": {"type": "string", "description": "Optional title fragment"}}}}},
    {"type": "function", "function": {
        "name": "close_window",
        "description": "Close a WINDOWS window gracefully by title fragment.",
        "parameters": {"type": "object", "required": ["title"],
            "properties": {"title": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "see_screen",
        "description": "Take a screenshot and have vision describe the WINDOWS screen, or answer a question about it.",
        "parameters": {"type": "object",
            "properties": {"question": {"type": "string", "description": "Optional question about the screen"}}}}},
    {"type": "function", "function": {
        "name": "click_target",
        "description": "Click ANY visible on-screen UI element. Pipeline: UIA exact → OCR text box → vision box + crop refine. This is the ONLY way to click — never fake a click with keys or prose. Always pass verify on important clicks.",
        "parameters": {"type": "object", "required": ["target"],
            "properties": {
                "target": {"type": "string", "description": "Visual description of the element + where it is"},
                "button": {"type": "string", "enum": ["left", "right"],
                    "description": "Mouse button. Default left. Use right for context menus."},
                "count": {"type": "integer", "description": "1 = click, 2 = double-click (e.g. open a file/icon). Default 1."},
                "title": {"type": "string", "description": "Optional window title fragment to click inside (default: foreground window)."},
                "verify": {"type": "string", "description": "Optional yes/no question answered AFTER the click with EXACT on-screen values + a fresh screenshot, e.g. 'does the display show 10?'. Use it before claiming success."}}}}},
    {"type": "function", "function": {
        "name": "inspect_screen",
        "description": "See EVERYTHING: UIA tree + OCR text-pixel boxes with 0-1000 coords (exact, never guessed). Use BEFORE clicking. Pass a question too and vision describes the screenshot as well.",
        "parameters": {"type": "object",
            "properties": {
                "title": {"type": "string", "description": "Window title fragment (default: foreground window)"},
                "question": {"type": "string", "description": "Optional question for vision about the screenshot"}}}}},
    {"type": "function", "function": {
        "name": "scroll_screen",
        "description": "Scroll the mouse wheel up/down, optionally at a visually-described spot (e.g. 'the file list'). Use for long pages, menus, and dropdowns.",
        "parameters": {"type": "object", "required": ["direction"],
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {"type": "integer", "description": "Wheel notches (default 5, max 20)"},
                "target": {"type": "string", "description": "Optional visual description of where to scroll"}}}}},
    {"type": "function", "function": {
        "name": "drag_screen",
        "description": "Drag from one visual target to another (sliders, window moves, canvas, file drags). Resolves both ends via UIA→OCR→vision-box.",
        "parameters": {"type": "object", "required": ["from_target"],
            "properties": {
                "from_target": {"type": "string", "description": "What to grab"},
                "to_target": {"type": "string", "description": "Where to drop it"},
                "to_x": {"type": "number", "description": "0-1000 X when no to_target"},
                "to_y": {"type": "number", "description": "0-1000 Y when no to_target"},
                "button": {"type": "string", "enum": ["left", "right"]}}}}},
    {"type": "function", "function": {
        "name": "ocr_read",
        "description": "Read EVERY text pixel on screen via OCR with 0-1000 boxes. Use for canvas/web/tray text UIA misses.",
        "parameters": {"type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max words (default 40)"}}}}},
    {"type": "function", "function": {
        "name": "browser_open",
        "description": "Open a URL in the browser via DOM automation (exact, no coordinates). Prefers attached Chrome/Edge CDP, else temp Chromium.",
        "parameters": {"type": "object", "required": ["url"],
            "properties": {"url": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "browser_click",
        "description": "Click a web element by CSS selector or text (e.g. '#search', 'text=Play'). Exact DOM — never guess pixels in browsers.",
        "parameters": {"type": "object", "required": ["selector"],
            "properties": {"selector": {"type": "string"},
                "timeout_ms": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "browser_type",
        "description": "Type into a web field by CSS selector (exact DOM).",
        "parameters": {"type": "object", "required": ["selector", "text"],
            "properties": {"selector": {"type": "string"}, "text": {"type": "string"},
                "submit": {"type": "boolean", "description": "Press Enter after typing"}}}}},
    {"type": "function", "function": {
        "name": "browser_read",
        "description": "Read the current web page text (title + url + body). No screenshot needed.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "browser_tabs",
        "description": "List open browser tabs (title + url).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "code_exec",
        "description": "Run a short Python snippet for file/folder/terminal jobs (organize downloads, rename, convert, read logs). Sandboxed temp dir, 15s cap, 4000-char log back. Use for FILES, never for clicking.",
        "parameters": {"type": "object", "required": ["code"],
            "properties": {
                "code": {"type": "string", "description": "Python code to run"},
                "timeout_s": {"type": "integer", "description": "3-60s cap (default 15)"}}}}},
]

APP_MAP = {
    "notepad": ("exe", "notepad.exe"), "calculator": ("exe", "calc.exe"), "calc": ("exe", "calc.exe"),
    "paint": ("exe", "mspaint.exe"), "explorer": ("exe", "explorer.exe"), "files": ("exe", "explorer.exe"),
    "terminal": ("exe", "cmd.exe"), "command prompt": ("exe", "cmd.exe"), "cmd": ("exe", "cmd.exe"),
    "browser": ("url", "https://www.google.com"), "chrome": ("exe", "chrome.exe"),
    "edge": ("exe", "msedge.exe"), "settings": ("uri", "ms-settings:"),
    "task manager": ("exe", "taskmgr.exe"),
}
# Full paths for exes that are often NOT on PATH (chrome.exe / msedge.exe are
# the classic cases — Popen(["chrome.exe"]) raises FileNotFound even though
# Chrome is installed). Searched in order when shutil.which() misses.
KNOWN_EXE_PATHS = {
    "chrome.exe": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ],
    "msedge.exe": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
}
# Plural / alias normalization so "open browsers", "google chrome",
# "microsoft edge", "file explorer" etc. resolve instead of failing.
# Values may be APP_MAP keys OR fuller Start-Menu search strings
# (e.g. "vscode" -> "visual studio code") used by the guessing system.
APP_ALIASES = {
    "browsers": "browser", "browser app": "browser", "web browser": "browser",
    "google chrome": "chrome", "chrome browser": "chrome", "google": "browser",
    "microsoft edge": "edge", "edge browser": "edge",
    "file explorer": "explorer", "windows explorer": "explorer", "folders": "explorer",
    "notepads": "notepad", "calculators": "calculator", "terminals": "terminal",
    "vscode": "visual studio code", "vs code": "visual studio code",
    "visual studio": "visual studio code",
    "word": "microsoft word", "excel": "microsoft excel",
    "powerpoint": "microsoft powerpoint", "ppt": "microsoft powerpoint",
    "outlook": "microsoft outlook", "teams": "microsoft teams",
    "adobe reader": "acrobat", "pdf reader": "acrobat",
    "photoshop": "adobe photoshop", "ps": "adobe photoshop",
    "premiere": "adobe premiere pro", "illustrator": "adobe illustrator",
    "obs": "obs studio", "vlc": "vlc media player",
    "media player": "vlc media player",
}
OPEN_BLOCKLIST = ("format", "diskpart", "regedit", "regedt32", "bcdedit", "cipher",
                  "shutdown", "taskkill", "sdelete", "dism", "bcdboot")
SENSITIVE_TITLE = ("password", "passwd", "credential", "login", "sign in", "sign-in",
                   "pin", "otp", "passcode", "card", "cvv", "bank", "secret", "vault")


class OSRefused(Exception):
    pass


def _pa():
    import pyautogui
    pyautogui.FAILSAFE = True  # slam mouse to a corner to abort motion
    pyautogui.PAUSE = 0.12
    return pyautogui


def _sleep(s):
    time.sleep(s)


_DPI_DONE = False


def _ensure_dpi():
    """Per-monitor DPI awareness so UIA rects == screenshot pixels.

    Without this, Windows virtualizes coords at 125% scaling and every
    vision-grounded click drifts. Never raises — best effort.
    """
    global _DPI_DONE
    if _DPI_DONE:
        return
    _DPI_DONE = True
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass


def _screen_size():
    """Primary monitor size in physical pixels (post-DPI-awareness)."""
    _ensure_dpi()
    try:
        from mss import MSS
        with MSS() as s:
            m = s.monitors[1]
            return int(m["width"]), int(m["height"])
    except Exception:
        pass
    try:
        pa = _pa()
        sz = pa.size()
        return int(sz[0]), int(sz[1])
    except Exception:
        return 1920, 1200


def _hud_write(state=None, action=None, ok=None, x=None, y=None,
               fx=None, fy=None, fw=None, fh=None):
    """Write HUD status for the floating overlay. Never raises.

    x/y = JARVIS pointer (ghost cursor). fx/fy/fw/fh = focused window rect
    in physical pixels for the persistent blue aura. focus_t = when set.
    """
    try:
        hud_path = os.path.join(tempfile.gettempdir(), "jarvis_hud.json")
        prev = {}
        try:
            with open(hud_path, "r") as f:
                prev = json.loads(f.read() or "{}")
        except Exception:
            prev = {}
        prev.update({
            "t": time.strftime("%H:%M:%S"),
            "state": state or prev.get("state") or "idle",
        })
        if action is not None:
            prev["action"] = str(action)[:160]
        if ok is not None:
            prev["ok"] = bool(ok)
        if x is not None and y is not None:
            prev["x"] = int(x)
            prev["y"] = int(y)
            prev["mark_t"] = time.time()
        if fx is not None and fy is not None and fw and fh:
            prev["fx"] = int(fx)
            prev["fy"] = int(fy)
            prev["fw"] = int(fw)
            prev["fh"] = int(fh)
            prev["focus_t"] = time.time()
        with open(hud_path, "w") as f:
            f.write(json.dumps(prev))
    except Exception:
        pass


def _hud_mark(sx, sy, action=""):
    try:
        _hud_write(state="working", action=action, ok=True, x=int(sx), y=int(sy))
    except Exception:
        pass


def _fg_title():
    import ctypes
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value or ""


def _top_windows():
    from pywinauto import Desktop
    out = []
    for w in Desktop(backend="uia").windows(top_level_only=True):
        try:
            t = w.window_text()
        except Exception:
            continue
        if t:
            out.append(t)
    return out


def _find_window(title_sub, timeout=4):
    from pywinauto import Desktop
    needle = (title_sub or "").lower()
    deadline = time.time() + timeout
    while True:
        cands = []
        for w in Desktop(backend="uia").windows(top_level_only=True):
            try:
                t = w.window_text() or ""
            except Exception:
                continue
            if not t:
                continue
            tl = t.lower()
            if needle and needle not in tl:
                continue
            score = 0 if (needle and tl == needle) else (1 if tl.startswith(needle) else 2)
            cands.append((score, len(t), t, w))
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]))
            return cands[0][3], cands[0][2]
        if time.time() > deadline:
            raise OSRefused(f"No window matching '{title_sub}'. Visible: {', '.join(_top_windows()[:12]) or 'none'}")
        time.sleep(0.4)


def _verify_window_appeared(before, hints=(), timeout=0):
    """Check whether a new top-level window appeared vs `before` snapshot.

    Returns the matched title or None. `hints` are substrings that count as
    success even without a strictly-new window (e.g. browser tab opened in an
    existing window). Never raises — verification must not break the action.
    """
    try:
        after = _top_windows()
    except Exception:
        return None
    before_set = set(before or [])
    new = [t for t in after if t not in before_set]
    if new:
        for h in hints:
            hl = (h or "").lower()
            for t in new:
                if hl and hl in t.lower():
                    return t
        return new[0]
    for h in hints:
        hl = (h or "").lower()
        if not hl:
            continue
        for t in after:
            if hl in t.lower():
                return t
    return None


def _launch_exe(exe):
    """Launch an exe robustly: PATH -> known full paths -> startfile -> cmd start."""
    import os as _os
    import shutil
    import subprocess
    if _os.path.isabs(exe) and _os.path.exists(exe):
        subprocess.Popen([exe])
        return exe
    found = shutil.which(exe)
    if found:
        subprocess.Popen([found])
        return found
    for cand in KNOWN_EXE_PATHS.get(exe.lower(), []):
        if _os.path.exists(cand):
            subprocess.Popen([cand])
            return cand
    # Last resorts: ShellExecute equivalent, then `cmd /c start`.
    try:
        _os.startfile(exe)  # type: ignore[attr-defined]
        return exe
    except Exception:
        pass
    subprocess.Popen(f'start "" "{exe}"', shell=True)
    return exe


# ================= APP GUESSING SYSTEM (open any installed app) =================
# Discovers Start-Menu shortcuts + App-Paths registry + PATH, fuzzy-matches the
# user's words, and launches the best hit. Stdlib only (os / winreg / difflib).
_APP_INDEX_CACHE = {"ts": 0.0, "shortcuts": [], "app_paths": {}}
APP_INDEX_TTL = 300


def _norm_app_name(s):
    s = (s or "").lower().strip()
    s = re.sub(r"\.(lnk|url|exe)$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _start_menu_dirs():
    dirs = []
    pd = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    ad = os.environ.get("APPDATA", "")
    if pd:
        dirs.append(os.path.join(pd, "Microsoft", "Windows", "Start Menu", "Programs"))
    if ad:
        dirs.append(os.path.join(ad, "Microsoft", "Windows", "Start Menu", "Programs"))
    return [d for d in dirs if d and os.path.isdir(d)]


def _build_app_index(force=False):
    now = time.time()
    if not force and now - _APP_INDEX_CACHE["ts"] < APP_INDEX_TTL and _APP_INDEX_CACHE["shortcuts"]:
        return _APP_INDEX_CACHE
    shortcuts = []
    seen = set()
    for base in _start_menu_dirs():
        try:
            for root, _ds, files in os.walk(base):
                for fn in files:
                    low = fn.lower()
                    if not (low.endswith(".lnk") or low.endswith(".url")):
                        continue
                    full = os.path.join(root, fn)
                    display = os.path.splitext(fn)[0]
                    norm = _norm_app_name(display)
                    if not norm:
                        continue
                    key = (norm, full.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    shortcuts.append({"display": display, "norm": norm, "path": full})
                    if len(shortcuts) > 3000:
                        break
        except Exception:
            continue
    app_paths = {}
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths")
            except Exception:
                continue
            try:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(key, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        sk = winreg.OpenKey(key, sub)
                        try:
                            val, _ = winreg.QueryValueEx(sk, "")
                        except Exception:
                            val = ""
                        try:
                            winreg.CloseKey(sk)
                        except Exception:
                            pass
                        if val and os.path.exists(val):
                            app_paths[sub.lower()] = val
                    except Exception:
                        continue
            finally:
                try:
                    winreg.CloseKey(key)
                except Exception:
                    pass
    except Exception:
        pass
    _APP_INDEX_CACHE["ts"] = now
    _APP_INDEX_CACHE["shortcuts"] = shortcuts
    _APP_INDEX_CACHE["app_paths"] = app_paths
    return _APP_INDEX_CACHE


def _is_blocked_app(name):
    low = (name or "").lower().strip()
    if not low:
        return False
    if low in OPEN_BLOCKLIST:
        return True
    first = low.split()[0].rstrip(".exe")
    if first in OPEN_BLOCKLIST:
        return True
    for tok in _norm_app_name(name).split():
        if tok in OPEN_BLOCKLIST:
            return True
    return False


def _score_names(query_norm, cand_norm):
    import difflib
    if not query_norm or not cand_norm:
        return 0
    if query_norm == cand_norm:
        return 95
    qt, ct = query_norm.split(), cand_norm.split()
    if query_norm in cand_norm or cand_norm in query_norm:
        return 86 if cand_norm.startswith(query_norm) or query_norm.startswith(cand_norm) else 80
    if all(t in ct for t in qt):
        return 72
    if any(t == c for t in qt for c in ct):
        return 58
    try:
        r = difflib.SequenceMatcher(None, query_norm, cand_norm).ratio()
    except Exception:
        r = 0
    if r >= 0.62:
        return int(40 + r * 20)
    return 0


def _guess_app(query, limit=5):
    """Return [{display, kind, score, path/exe}] sorted best-first."""
    import shutil
    q = _norm_app_name(APP_ALIASES.get((query or "").lower().strip(), query))
    if not q:
        return []
    idx = _build_app_index()
    scored = {}  # display.lower() -> entry
    # Curated map entries participate so list_apps shows daily apps too.
    for key in APP_MAP:
        s = _score_names(q, _norm_app_name(key))
        if s > 0:
            scored[key] = {"display": key, "kind": "daily", "score": s, "key": key}
    for exe, full in idx["app_paths"].items():
        base = _norm_app_name(os.path.splitext(exe)[0])
        s = _score_names(q, base)
        if s > 0:
            k = exe.lower()
            if k not in scored or scored[k]["score"] < s:
                scored[k] = {"display": os.path.splitext(exe)[0], "kind": "app-path",
                             "score": s, "exe": full}
    for sc in idx["shortcuts"]:
        s = _score_names(q, sc["norm"])
        if s > 0:
            k = sc["norm"]
            if k not in scored or scored[k]["score"] < s:
                scored[k] = {"display": sc["display"], "kind": "start-menu",
                             "score": s, "path": sc["path"]}
    # Singular fallback ("photos" -> "photo").
    if not scored and q.endswith("s") and len(q) > 3:
        return _guess_app(q[:-1], limit=limit)
    # PATH exact (covers WindowsApps execution aliases like spotify.exe).
    for cand in (q, q + ".exe", query.strip()):
        try:
            found = shutil.which(cand)
        except Exception:
            found = None
        if found:
            scored["path:" + found.lower()] = {"display": os.path.splitext(os.path.basename(found))[0],
                                               "kind": "path", "score": 88, "exe": found}
            break
    out = sorted(scored.values(), key=lambda e: (-e["score"], e["display"]))
    return out[:max(1, min(limit, 20))]


def os_list_apps(query="", limit=12):
    idx = _build_app_index()
    limit = max(1, min(int(limit or 12), 50))
    q = _norm_app_name(query)
    if not q:
        names = sorted({s["display"] for s in idx["shortcuts"]})[:limit]
        return (f"{len(idx['shortcuts'])} Start-Menu apps indexed "
                f"({len(idx['app_paths'])} App-Paths). Sample:\n- " + "\n- ".join(names))
    hits = _guess_app(query, limit=limit)
    if not hits:
        return f"No installed apps matching '{query}'. Try a shorter name (e.g. 'code' for VS Code)."
    lines = [f"- {h['display']} [{h['kind']}]" for h in hits]
    return f"{len(hits)} app(s) matching '{query}':\n" + "\n".join(lines)


def _guess_and_launch(query):
    import subprocess
    if _is_blocked_app(query):
        raise OSRefused(f"Blocked for safety: {query}")
    try:
        _before_windows = _top_windows()
    except Exception:
        _before_windows = []
    hits = _guess_app(query, limit=5)
    if not hits or hits[0]["score"] < 45:
        alt = f" Close matches: {', '.join(h['display'] for h in hits[:3])}." if hits else ""
        raise OSRefused(f"No installed app matches '{query}'.{alt} Use list_apps to browse, e.g. list_apps('office').")
    best = hits[0]
    kind = best["kind"]
    try:
        if kind == "start-menu":
            os.startfile(best["path"])
        elif kind in ("app-path", "path"):
            subprocess.Popen([best["exe"]])
        elif kind == "daily":
            k, v = APP_MAP[best["key"]]
            if k == "exe":
                _launch_exe(v)
            elif k == "uri":
                os.startfile(v)
            else:
                import webbrowser
                webbrowser.open(v)
        else:
            raise OSRefused(f"Cannot launch '{query}'.")
    except OSRefused:
        raise
    except Exception as e:
        raise OSRefused(f"Found '{best['display']}' but could not launch it: {e}")
    _sleep(1.2)
    others = [h["display"] for h in hits[1:3] if h["score"] >= best["score"] - 12]
    extra = f" (also close: {', '.join(others)})" if others else ""
    seen = _verify_window_appeared(_before_windows, (best["display"], query))
    ver = f" Verified: window '{seen}' visible." if seen else " Unverified: no new window detected — check screen."
    return f"Opened {best['display']} (matched '{query}'){extra}." + ver


def _browser_state_hint():
    """Peek at the browser window text after opening a URL (cheap pywinauto
    read, no vision). Catches blockers like Chrome's profile picker so the
    agent reports them instead of claiming the site loaded. Never raises."""
    for hint in ("chrome", "google", "edge", "firefox"):
        try:
            w, t = _find_window(hint, timeout=1)
        except Exception:
            continue
        try:
            parts = []
            for d in w.descendants():
                try:
                    tx = (d.window_text() or "").strip()
                except Exception:
                    continue
                if tx and tx not in parts:
                    parts.append(tx)
                if len("\n".join(parts)) > 400:
                    break
            sample = " / ".join(parts)[:300]
        except Exception:
            sample = ""
        if sample:
            return f" Browser window '{t}' shows: {sample}"
        return f" Browser window '{t}' has no readable text yet."
    return ""


def _extract_url(text):
    """Find a URL anywhere in the target string (model often sends
    'chrome https://youtube.com' as one target). Returns the URL or None."""
    m = re.search(r"https?://[^\s'\"<>]+", text or "")
    if m:
        return m.group(0).rstrip(".,;:!?)")
    m = re.search(r"\bwww\.[^\s'\"<>]+", text or "")
    if m:
        return "https://" + m.group(0).rstrip(".,;:!?)")
    m = re.search(r"\b([a-z0-9-]+\.(?:com|org|net|io|in|ai)(?:/[^\s'\"<>]*)?)", text or "", re.I)
    if m and " " not in m.group(1):
        return "https://" + m.group(1).rstrip(".,;:!?)")
    return None


def os_open(target):
    import os as _os
    import subprocess
    import webbrowser
    t = (target or "").strip()
    if not t:
        raise OSRefused("Empty target.")
    # URL anywhere in the target wins: "chrome https://youtube.com" must open
    # YouTube, not just Chrome. webbrowser.open uses the default browser.
    url = _extract_url(t)
    if url:
        import webbrowser
        try:
            _url_before = _top_windows()
        except Exception:
            _url_before = []
        webbrowser.open(url)
        _sleep(1.5)
        seen = _verify_window_appeared(_url_before, ("chrome", "edge", "firefox", "browser"))
        ver = f" Verified: window '{seen}' visible." if seen else " Unverified: no browser window detected — check screen."
        return f"Opened {url} in the browser." + ver + _browser_state_hint()
    low = t.lower()
    low = APP_ALIASES.get(low, low)
    base = low.split()[0].rstrip(".exe")
    if base in OPEN_BLOCKLIST or low in OPEN_BLOCKLIST:
        raise OSRefused(f"Blocked for safety: {t}")
    try:
        _before_windows = _top_windows()
    except Exception:
        _before_windows = []
    def _ver(hints=()):
        seen = _verify_window_appeared(_before_windows, hints)
        return f" Verified: window '{seen}' visible." if seen else " Unverified: no new window detected — check screen."
    if low.startswith(("http://", "https://", "www.")) or (" " not in low and low.endswith((".com", ".org", ".net", ".io", ".in", ".ai"))):
        webbrowser.open(t if "://" in t else "https://" + t)
        _sleep(1.2)
        return f"Opened browser at {t}." + _ver(("chrome", "edge", "firefox", "browser"))
    if low in APP_MAP:
        kind, val = APP_MAP[low]
        try:
            if kind == "exe":
                _launch_exe(val)
            elif kind == "uri":
                _os.startfile(val)
            else:
                webbrowser.open(val)
        except Exception as e:
            # Browser-family must never hard-fail when a default browser
            # exists: fall back to webbrowser.open so "open chrome/edge"
            # still gets the user a window instead of FileNotFound.
            if low in ("browser", "chrome", "edge"):
                try:
                    webbrowser.open("https://www.google.com")
                    _sleep(1.2)
                    return f"Opened {t} via default browser (fallback: {e})." + _ver(("chrome", "edge", "firefox"))
                except Exception:
                    pass
            raise OSRefused(f"Cannot open '{t}': {e}")
        _sleep(1.2)
        return f"Opened {t}." + _ver((low,))
    if _os.path.exists(t) or (len(t) > 1 and t[1] == ":") or t.startswith("ms-settings:"):
        try:
            _os.startfile(t)
        except Exception as e:
            raise OSRefused(f"Cannot open '{t}': {e}")
        _sleep(1.2)
        return f"Opened {t}." + _ver((_os.path.basename(t).split(".")[0],))
    # NEW: fuzzy guessing system — any installed app by approximate name.
    # Raises OSRefused with close-match hints when nothing fits.
    try:
        return _guess_and_launch(low)
    except OSRefused:
        raise
    except Exception as e:
        raise OSRefused(f"Cannot open '{t}': {e}")


def _force_foreground(hwnd):
    """Best-effort Win32 bring-to-front. Never raises — returns True/False."""
    try:
        import ctypes
        u32 = ctypes.windll.user32
        SW_RESTORE, SW_MAXIMIZE = 9, 3
        try:
            u32.ShowWindow(hwnd, SW_RESTORE)
        except Exception:
            pass
        try:
            u32.ShowWindow(hwnd, SW_MAXIMIZE)
        except Exception:
            pass
        try:
            u32.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False
    except Exception:
        return False


def os_focus(title, maximize=True):
    title = (title or "").strip()
    if not title:
        raise OSRefused("Empty window title — say which window, e.g. 'notepad'.")
    # Try the raw title, then basename variants ("document1.txt" → "document1",
    # "report.pdf" → "report") since window titles often append " - Notepad".
    cands = [title]
    try:
        base = os.path.basename(title)
        if base and base != title:
            cands.append(base)
        noext, ext = os.path.splitext(base or title)
        if ext and noext and noext not in cands:
            cands.append(noext)
    except Exception:
        pass
    last_err = None
    for cand in cands:
        try:
            w, t = _find_window(cand)
        except OSRefused as e:
            last_err = e
            continue
        # 1) Un-minimize via pywinauto (best effort — not all backends support it).
        for meth in ("restore",):
            try:
                getattr(w, meth)()
                break
            except Exception:
                continue
        # 2) Maximize when asked (default True: focused app stays maximized
        # until the next app is called).
        if maximize:
            try:
                w.maximize()
            except Exception:
                pass
        # 3) Native focus + Win32 foreground backup (beats foreground-lock).
        try:
            w.set_focus()
        except Exception as e:
            raise OSRefused(f"Found '{t}' but could not focus it: {e}")
        try:
            hwnd = int(w.handle)
            _force_foreground(hwnd)
        except Exception:
            pass
        _sleep(0.6)
        # 4) Verify + one retry (minimized windows need a second kick).
        try:
            fg = _fg_title()
        except Exception:
            fg = ""
        if not (fg and cand.lower() in fg.lower()):
            try:
                w.set_focus()
                try:
                    _force_foreground(int(w.handle))
                except Exception:
                    pass
                _sleep(0.6)
                fg = _fg_title()
            except Exception:
                pass
        ver = f" Verified foreground: '{fg}'." if fg and cand.lower() in fg.lower() else (
            f" Foreground now '{fg}'." if fg else "")
        # 5) Proof: rect + real cursor to center + persistent HUD aura.
        try:
            r = w.rectangle()
            cx, cy = r.left + r.width() // 2, r.top + r.height() // 2
            try:
                pa = _pa()
                pa.moveTo(cx, cy, duration=0.25)
            except Exception:
                pass
            try:
                _hud_write(state="working", action=f"Focus {t}", ok=True,
                           x=cx, y=cy, fx=r.left, fy=r.top, fw=r.width(), fh=r.height())
            except Exception:
                pass
            return f"Focused '{t}' at ({r.left},{r.top},{r.width()}x{r.height()})." + ver
        except Exception:
            try:
                _hud_write(state="working", action=f"Focus {t}", ok=True)
            except Exception:
                pass
            return f"Focused '{t}'." + ver
    raise OSRefused(f"{last_err} Hint: open it first with open_app, e.g. open_app('notepad').")


def _escape_sk(text):
    text = text.replace("{", "{{}").replace("}", "{}}")
    for c in "+^%~()":
        text = text.replace(c, "{%s}" % c)
    return text


def os_type(text, title=None):
    from pywinauto import keyboard
    text = str(text or "")
    if not text:
        raise OSRefused("Nothing to type.")
    if len(text) > 2000:
        raise OSRefused("Type limit is 2000 characters per approval — split it into chunks.")
    if title:
        os_focus(title)
    else:
        _sleep(0.3)
    fg = _fg_title()
    if any(s in fg.lower() for s in SENSITIVE_TITLE):
        raise OSRefused(f"Refused: '{fg}' looks like a login/credential screen. I never type there.")
    for i in range(0, len(text), 500):
        keyboard.send_keys(_escape_sk(text[i:i + 500]), with_spaces=True, pause=0.02)
    return f"Typed {len(text)} chars into '{fg}'."


KEYMAP = {"enter": "{ENTER}", "tab": "{TAB}", "esc": "{ESC}", "escape": "{ESC}",
          "space": "{SPACE}", "backspace": "{BACKSPACE}", "delete": "{DELETE}",
          "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
          "home": "{HOME}", "end": "{END}", "pageup": "{PGUP}", "pagedown": "{PGDN}",
          "printscreen": "{PRTSC}", "prtsc": "{PRTSC}"}
MODS = {"ctrl": "^", "alt": "%", "shift": "+", "win": "#"}


def _to_sendkeys(keys):
    if isinstance(keys, list):
        keys = "+".join(keys)
    s = str(keys or "").strip()
    if not s:
        raise OSRefused("No keys given.")
    if s[0] in "^%+#{":
        return s
    parts = [p.strip() for p in s.split("+")]
    out = ""
    for p in parts[:-1]:
        m = MODS.get(p.lower())
        if not m:
            raise OSRefused(f"Unknown modifier '{p}'. Use ctrl/alt/shift/win.")
        out += m
    last = parts[-1]
    ll = last.lower()
    if len(last) == 1:
        out += last
    elif ll in KEYMAP:
        out += KEYMAP[ll]
    elif re.match(r"^f\d{1,2}$", ll):
        out += "{%s}" % ll.upper()
    else:
        raise OSRefused(f"Unknown key '{last}'.")
    return out


def os_keys(keys, title=None):
    from pywinauto import keyboard
    if title:
        os_focus(title)
    sk = _to_sendkeys(keys)
    keyboard.send_keys(sk, pause=0.05)
    return f"Pressed {keys} in '{_fg_title()}'."


def os_read(title=None):
    w, t = (_find_window(title, timeout=3) if title else _find_window(_fg_title(), timeout=2))
    parts, seen = [], set()
    try:
        descs = w.descendants()
    except Exception as e:
        raise OSRefused(f"Cannot inspect '{t}': {e}")
    for d in descs:
        try:
            tx = (d.window_text() or "").strip()
        except Exception:
            continue
        if tx and tx not in seen:
            seen.add(tx)
            parts.append(tx)
    text = "\n".join(parts)[:4000]
    return f"Window '{t}':\n{text or '(no readable text)'}"


def os_close(title):
    w, t = _find_window(title, timeout=2)
    try:
        w.close()
    except Exception as e:
        raise OSRefused(f"Cannot close '{t}': {e}")
    return f"Closed '{t}'."


def _shot(max_w=1280):
    from mss import MSS
    from PIL import Image
    import io
    _ensure_dpi()
    with MSS() as s:
        raw = s.grab(s.monitors[1])
        full_w, full_h = int(raw.width), int(raw.height)
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    scale = min(1.0, max_w / max(1, img.size[0]))
    img.thumbnail((max_w, 10 ** 6))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    jpeg = buf.getvalue()
    # Remember the freshest screenshot so the chat UI can show "what he sees".
    global LAST_SHOT
    try:
        LAST_SHOT = {"jpeg": jpeg, "t": time.strftime("%H:%M:%S"),
                     "size": img.size, "full": (full_w, full_h), "scale": scale}
    except Exception:
        pass
    return img.size, jpeg


LAST_SHOT = {"jpeg": None, "t": "", "size": (0, 0), "full": (0, 0), "scale": 1.0}
LAST_SHOT_ANNOTATED = {"jpeg": None, "t": "", "boxes": []}


def _shot_png_full():
    """Full-res PNG bytes + (w,h) for OCR/grounding. Never downscales."""
    from mss import MSS
    from PIL import Image
    import io
    _ensure_dpi()
    with MSS() as s:
        raw = s.grab(s.monitors[1])
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return img.size, buf.getvalue(), img


def _annotate_shot(boxes=(), label=""):
    """Draw numbered boxes on the freshest screenshot for chat/HUD proof.

    boxes: iterable of (x, y, w, h) in 0-1000 space. Returns JPEG bytes.
    Never raises — returns None on failure.
    """
    try:
        from PIL import Image, ImageDraw
        import io
        base = (LAST_SHOT or {}).get("jpeg")
        size = (LAST_SHOT or {}).get("size") or (1280, 800)
        if not base:
            _size, base = _shot(max_w=1280)
            size = _size
        img = Image.open(io.BytesIO(base)).convert("RGB")
        w, h = img.size
        d = ImageDraw.Draw(img)
        for i, b in enumerate(list(boxes or [])[:24]):
            try:
                x, y, bw, bh = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                x0, y0 = int(x / 1000 * w), int(y / 1000 * h)
                x1, y1 = int((x + bw) / 1000 * w), int((y + bh) / 1000 * h)
                d.rectangle([x0, y0, x1, y1], outline=(0, 229, 255), width=2)
                d.rectangle([x0, y0 - 14, x0 + 34, y0], fill=(0, 229, 255))
                d.text((x0 + 3, y0 - 13), str(i + 1), fill=(0, 0, 0))
            except Exception:
                continue
        if label:
            d.text((8, h - 18), str(label)[:90], fill=(0, 229, 255))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        jpeg = buf.getvalue()
        global LAST_SHOT_ANNOTATED
        try:
            LAST_SHOT_ANNOTATED = {"jpeg": jpeg, "t": time.strftime("%H:%M:%S"),
                                   "boxes": list(boxes or [])[:24]}
        except Exception:
            pass
        return jpeg
    except Exception:
        return None


def _vision(prompt, jpeg, timeout=150):
    """Vision via the Qwen2.5-VL GGUF + mmproj on llama.cpp.

    BUILD 18: no second engine. If the VL model cannot load, the
    InferenceError from the backend carries the rejected fits and the
    free VRAM - it is raised verbatim rather than being swallowed or
    quietly served by something else.
    """
    r = _router()
    if r is None:
        raise InferenceError(
            "Vision backend unavailable - inference package missing ("
            + (_INFERENCE_IMPORT_ERROR or "backend/inference/ import failed")
            + ")")
    return r.chat_with_image(prompt, jpeg, model=_vision_model(),
                             timeout=timeout)


# BUILD 17/18: the live vision model, resolved from models\ by llama.cpp.
_VISION_READY = None  # None = not probed yet, True/False = result


def _vision_model():
    """Active vision model label - the Qwen2.5-VL GGUF."""
    return VISION_MODEL


def _vision_status():
    r = _router()
    files = {}
    try:
        files = (r.llamacpp.health() or {}).get("files", {}) if r else {}
    except Exception:
        pass
    return {"model": _vision_model(),
            "gguf": files.get("vision"), "mmproj": files.get("mmproj"),
            "ready": _VISION_READY, "mode": _router_mode()}


def _vision_capable():
    """Can the local vision path actually load? Checks the GGUF + mmproj
    are present and the build exposes a Qwen2.5-VL handler. Cheap - no
    model load. Cached. Never raises."""
    global _VISION_READY
    if _VISION_READY is not None:
        return _VISION_READY
    try:
        r = _router()
        cpp = (r.llamacpp.health() or {}) if r else {}
        files = cpp.get("files") or {}
        if not files.get("vision") or not files.get("mmproj"):
            _VISION_READY = False
            return _VISION_READY
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # noqa: F401
        _VISION_READY = True
    except Exception:
        _VISION_READY = False
    return _VISION_READY


def probe_vision_backend(timeout=90):
    """BUILD 17, replaces probe_ministral_vision(). Verifies the local
    vision path can load its GGUF + projector. Returns True/False and
    caches the winner. Never raises."""
    if not _vision_capable():
        _VISION_READY = False
        return False
    r = _router()
    try:
        h = r.llamacpp.health() or {}
        _VISION_READY = bool(h.get("vision_model_file") and
                             h.get("mmproj_file"))
    except Exception:
        _VISION_READY = False
    return bool(_VISION_READY)


# ================= UIA PRECISION ENGINE =================
# Windows UI Automation via pywinauto (already installed): instead of guessing
# coordinates from a screenshot, ask Windows itself for every control's exact
# name, type, and pixel rectangle. Deterministic for ALL native apps —
# Calculator, Notepad, Settings, Office, Explorer, dialogs. Vision grounding
# remains only for non-UIA surfaces (web canvas, games, video, tray).
_UIA_STOPWORDS = ("the", "a", "an", "button", "menu", "item", "icon", "tab",
                  "window", "dialog", "box", "bar", "field", "please",
                  "number", "digit", "key", "press", "click", "hit", "tap",
                  "select", "choose", "type", "open", "close", "on", "in",
                  "at", "to", "of", "for", "called", "named", "labeled")


def _uia_controls(title=None):
    """Return (window_title, [(text, control_type, rect_tuple)]) for a window."""
    w, t = (_find_window(title, timeout=3) if title else _find_window(_fg_title(), timeout=2))
    try:
        descs = w.descendants()
    except Exception as e:
        raise OSRefused(f"Cannot inspect '{t}': {e}")
    out = []
    for d in descs:
        try:
            tx = (d.window_text() or "").strip()
        except Exception:
            continue
        if not tx:
            continue
        try:
            ctype = (d.element_info.control_type or d.element_info.class_name or "").strip()
        except Exception:
            ctype = ""
        try:
            r = d.rectangle()
            rect = (r.left, r.top, r.width(), r.height())
        except Exception:
            rect = None
        out.append({"ctl": d, "text": tx, "type": ctype, "rect": rect})
    return t, out


def ui_dump(title=None, max_nodes=150):
    """Compact accessibility tree for the model: exact names + bounds."""
    t, ctrls = _uia_controls(title)
    lines = [f"{len(ctrls)} controls in '{t}':"]
    for c in ctrls[:max_nodes]:
        rect = ("@ %d,%d %dx%d" % c["rect"]) if c["rect"] else "@ ?"
        lines.append(f"- [{c['type'] or '?'}] '{c['text'][:60]}' {rect}")
    if len(ctrls) > max_nodes:
        lines.append(f"...(+{len(ctrls) - max_nodes} more)")
    return "\n".join(lines)[:6000], t, len(ctrls)


def ui_values(title=None, limit=15):
    """Exact current values of display-like controls (edits, texts, status)."""
    _t, ctrls = _uia_controls(title)
    vals = []
    for c in ctrls:
        ct = (c["type"] or "").lower()
        if any(k in ct for k in ("edit", "text", "statusbar", "titlebar", "document")):
            vals.append(f"{c['text'][:80]}")
        if len(vals) >= limit:
            break
    return vals


_NUM_EQUIV = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
# Container chrome is never the click target itself (first pass skips it).
_UIA_CONTAINERS = ("group", "window", "pane", "menubar", "titlebar", "statusbar",
                   "toolbar", "tab", "tabitem")


def _uia_match(ctrls, query):
    """Fuzzy-match a visual description to a real control. Returns best or None.

    Strict: EVERY query word (or its digit/word twin, e.g. five<->5) must occur
    in the control name. Containers (groups, panes…) only match when nothing
    actionable does — so 'number 5' hits Button Five, never 'Number pad'.
    """
    words = [w for w in _norm_app_name(query).split() if w and w not in _UIA_STOPWORDS]
    if not words:
        return None

    def _score(norm):
        toks = norm.split()
        ok_all, ok_tok = True, True
        for w in words:
            twins = {w}
            if w in _NUM_EQUIV:
                twins.add(_NUM_EQUIV[w])
            if not any(t in norm for t in twins):
                ok_all = False
                break
            if not any(t in toks for t in twins):
                ok_tok = False
        if not ok_all:
            return 0
        if norm == " ".join(words):
            return 100
        return 80 if ok_tok else 65

    for skip_containers in (True, False):
        best, best_score = None, 0
        for c in ctrls:
            if skip_containers and (c["type"] or "").lower() in _UIA_CONTAINERS:
                continue
            norm = _norm_app_name(c["text"])
            if not norm:
                continue
            score = _score(norm)
            # Short numeric/symbol buttons ("5", "=") match exactly or not at all
            # (digit words count: "5" hits Button Five, "=" hits Equals... no —
            # "=" only hits a literal "=", never a word).
            if len(words) == 1 and len(words[0]) <= 2:
                twins = {words[0]}
                if words[0] in _NUM_EQUIV:
                    twins.add(_NUM_EQUIV[words[0]])
                score = 100 if norm in twins else 0
            if score > best_score:
                best, best_score = c, score
        if best is not None:
            return best
    return None


def _uia_click(best, button="left", count=1):
    """Click a resolved UIA control natively (no coordinate math)."""
    d = best["ctl"]
    try:
        if button == "right":
            d.right_click_input()
        elif count == 2:
            d.double_click_input()
        else:
            d.click_input()
        return True
    except Exception:
        pass
    # Fallback: exact rect center via mouse (still exact — rect came from Windows).
    try:
        r = d.rectangle()
        pa = _pa()
        pa.moveTo(r.left + r.width() // 2, r.top + r.height() // 2, duration=0.2)
        if button == "right":
            pa.rightClick()
        elif count == 2:
            pa.doubleClick(interval=0.12)
        else:
            pa.click()
        return True
    except Exception as e:
        raise OSRefused(f"Could not click '{best['text']}': {e}")


# ================= OCR LAYER (Tesseract — every pixel with text) =================
# UIA misses web canvas, games, tray, custom-drawn controls. OCR reads them as
# word boxes with confidence, mapped to 0-1000 space. Optional dep: if the
# tesseract binary is missing, every function degrades to [] with a clear hint.
_TESS_PATH = None
_TESS_CHECKED = False


def _tess_cmd():
    """Locate the tesseract binary. Never raises — returns path or None."""
    global _TESS_PATH, _TESS_CHECKED
    if _TESS_CHECKED:
        return _TESS_PATH
    _TESS_CHECKED = True
    try:
        import shutil
        found = shutil.which("tesseract")
        if found:
            _TESS_PATH = found
            return _TESS_PATH
        for cand in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(cand):
                _TESS_PATH = cand
                return _TESS_PATH
    except Exception:
        pass
    _TESS_PATH = None
    return None


def _probe_ocr():
    try:
        import pytesseract  # noqa: F401
    except Exception as e:
        return {"available": False, "reason": f"pip install pytesseract ({e})"}
    cmd = _tess_cmd()
    if not cmd:
        return {"available": False, "reason": "tesseract binary not on PATH — install Tesseract-OCR and restart backend"}
    # PaddleOCR upgrade path (optional, heavier): report if present.
    try:
        import paddleocr  # noqa: F401
        paddle = True
    except Exception:
        paddle = False
    return {"available": True, "cmd": cmd, "engine": "paddle" if paddle else "tesseract"}


def ocr_status():
    """Report OCR readiness for /api/health. Never raises. BUILD 17 cached."""
    return _cached_probe("ocr", _probe_ocr)


def _probe_grounder():
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:
        has_torch = False
    # Omni/Florence hooks: present only when weights installed; VLM box path otherwise.
    omni = False
    try:
        import os as _o
        omni = bool(_o.environ.get("JARVIS_OMNI", ""))
    except Exception:
        pass
    return {"engine": "omni" if omni else "vlm-box",
            "torch": has_torch, "cuda": False,
            "hint": "vlm-box active; set JARVIS_OMNI=1 + weights for OmniParser upgrade"}


def grounder_status():
    """Report vision-box grounder readiness. Never raises. BUILD 17 cached."""
    return _cached_probe("grounder", _probe_grounder)


def _ocr_words(max_w=1920, min_conf=30):
    """OCR the full screen → [{text, x,y,w,h (0-1000), conf}]. Empty on failure."""
    try:
        cmd = _tess_cmd()
        if not cmd:
            return []
        import pytesseract
        from PIL import Image
        import io
        pytesseract.pytesseract.tesseract_cmd = cmd
        _size, _png, img = _shot_png_full()
        if max_w and img.size[0] > max_w:
            img.thumbnail((max_w, 10 ** 6))
        fw, fh = _screen_size()
        # Scale OCR image coords → screen pixels → 0-1000.
        iw, ih = img.size
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        out = []
        n = len(data.get("text", []))
        for i in range(n):
            try:
                txt = (data["text"][i] or "").strip()
                if not txt:
                    continue
                conf = float(data["conf"][i] or -1)
                if conf < min_conf:
                    continue
                l, t, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
                if w <= 0 or h <= 0:
                    continue
                # OCR image may be downscaled from full screen: map back.
                sx = (l + w / 2) / max(1, iw) * fw
                sy = (t + h / 2) / max(1, ih) * fh
                bw = w / max(1, iw) * fw / max(1, fw) * 1000
                bh = h / max(1, ih) * fh / max(1, fh) * 1000
                x = sx / max(1, fw) * 1000 - bw / 2
                y = sy / max(1, fh) * 1000 - bh / 2
                out.append({"text": txt[:60], "x": x, "y": y, "w": bw, "h": bh, "conf": conf})
            except Exception:
                continue
        return out[:400]
    except Exception:
        return []


def _ocr_match(words, query):
    """Fuzzy-match query to an OCR word/phrase box. Returns best or None."""
    qwords = [w for w in _norm_app_name(query).split() if w and w not in _UIA_STOPWORDS]
    if not qwords or not words:
        return None
    best, best_score = None, 0
    for wd in words:
        try:
            norm = _norm_app_name(wd.get("text", ""))
            if not norm:
                continue
            hits = sum(1 for w in qwords if w in norm or norm in w)
            if not hits:
                # digit/word twins (5<->five)
                twins = sum(1 for w in qwords if w in _NUM_EQUIV and _NUM_EQUIV[w] in norm)
                hits = twins
            if not hits:
                continue
            score = hits * 20 + float(wd.get("conf", 0)) / 10.0
            # Prefer single-word exact hits for tiny buttons.
            if len(qwords) == 1 and norm == qwords[0]:
                score += 50
            if score > best_score:
                best, best_score = wd, score
        except Exception:
            continue
    return best if best_score >= 20 else None


def _perceive(title=None):
    """Unified perception: UIA controls + OCR word boxes in 0-1000 space.

    Returns dict {window, uia:[...], ocr:[...]}. Never raises — degrades to
    whatever layer is available.
    """
    uia_list, win = [], ""
    try:
        win, ctrls = _uia_controls(title)
        fw, fh = _screen_size()
        for c in ctrls:
            try:
                r = c.get("rect")
                if r:
                    x, y, w, h = r
                    uia_list.append({
                        "text": c.get("text", "")[:80],
                        "type": c.get("type", ""),
                        "x": x / max(1, fw) * 1000,
                        "y": y / max(1, fh) * 1000,
                        "w": w / max(1, fw) * 1000,
                        "h": h / max(1, fh) * 1000,
                    })
                else:
                    uia_list.append({"text": c.get("text", "")[:80], "type": c.get("type", ""),
                                     "x": -1, "y": -1, "w": 0, "h": 0})
            except Exception:
                continue
    except Exception:
        try:
            win = title or _fg_title()
        except Exception:
            win = ""
    try:
        ocr_list = _ocr_words()
    except Exception:
        ocr_list = []
    return {"window": win, "uia": uia_list[:300], "ocr": ocr_list[:300]}


def os_ocr_read(limit=40):
    """Raw OCR dump of the screen: every text pixel with 0-1000 boxes."""
    words = _ocr_words()
    if not words and not _tess_cmd():
        raise OSRefused("OCR unavailable: tesseract binary not found. Install Tesseract-OCR (https://github.com/tesseract-ocr/tesseract), add it to PATH, restart backend. UIA + vision still work.")
    if not words:
        try:
            import pytesseract  # noqa: F401
        except Exception as e:
            raise OSRefused(f"OCR unavailable: pip install pytesseract ({e}).")
        return "OCR ran but found no text on this screen."
    lines = []
    for wd in words[:max(1, min(int(limit or 40), 200))]:
        lines.append(f"- '{wd['text']}' @ {wd['x']:.0f},{wd['y']:.0f} {wd['w']:.0f}x{wd['h']:.0f} conf {wd['conf']:.0f}")
    return f"{len(words)} OCR words:\n" + "\n".join(lines)


def os_inspect(title=None, question=None):
    """See EVERYTHING structural: UIA tree + OCR boxes (+ vision on demand)."""
    tree, t, n = ui_dump(title)
    try:
        words = _ocr_words()
    except Exception:
        words = []
    ocr_txt = ""
    if words:
        ocr_txt = "\nOCR TEXT PIXELS (exact 0-1000 boxes, use for canvas/tray/web):\n" + "\n".join(
            f"- '{w['text'][:40]}' @ {w['x']:.0f},{w['y']:.0f} {w['w']:.0f}x{w['h']:.0f}"
            for w in words[:60])
        ocr_txt = ocr_txt[:3000]
    elif not _tess_cmd():
        ocr_txt = "\n(OCR offline — install Tesseract-OCR binary for text-pixel boxes.)"
    full = tree + ocr_txt
    if not question:
        return full + "\n(No screenshot taken — pass a question for vision description.)"
    _size, jpeg = _shot()
    try:
        res = _vision(f"{question}\n(UI controls known: {tree[:1500]})", jpeg)
        content = (res.get("message", {}) or {}).get("content", "")
    except Exception as e:
        raise OSRefused(f"Vision backend failed: {e}")
    if not content:
        raise OSRefused("Vision returned nothing.")
    return full + f"\n\nVISION SAYS:\n{content[:1500]}"


def os_see(question=None):
    size, jpeg = _shot()
    q = question or "Describe everything important on this Windows screen: open windows, dialogs, and actionable controls."
    try:
        res = _vision(q, jpeg)
        content = (res.get("message", {}) or {}).get("content", "")
    except Exception as e:
        raise OSRefused(f"Vision backend failed: {e}")
    if not content:
        raise OSRefused("Vision returned nothing.")
    return f"Screenshot ({size[0]}x{size[1]}):\n{content[:2000]}"


def _parse_box(content):
    """Parse a vision box or point from free text → (x,y,w,h) in 0-1000.

    Accepts {"x":..,"y":..,"w":..,"h":..} boxes AND legacy {"x":..,"y":..}
    points (upgraded to a 40x40 box). Returns None when unparseable.
    """
    try:
        clean = re.sub(r"```(?:json)?|```", "", content or "")
        m = re.search(r"\{[^{}]*['\"]x['\"]\s*:\s*(\d+)[^{}]*['\"]y['\"]\s*:\s*(\d+)[^{}]*\}", clean)
        if not m:
            nums = re.findall(r"\b(\d{1,4})\b", clean)
            if len(nums) >= 2:
                x, y = int(nums[0]), int(nums[1])
                if 0 <= x <= 1000 and 0 <= y <= 1000:
                    return (x - 20, y - 20, 40, 40)
            return None
        x, y = int(m.group(1)), int(m.group(2))
        mw = re.search(r"['\"]w(?:idth)?['\"]\s*:\s*(\d+)", clean)
        mh = re.search(r"['\"]h(?:eight)?['\"]\s*:\s*(\d+)", clean)
        if mw and mh:
            w, h = int(mw.group(1)), int(mh.group(1))
            if 0 <= x <= 1000 and 0 <= y <= 1000 and 2 <= w <= 600 and 2 <= h <= 600:
                return (x, y, w, h)
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            # Point → small box around center (backward compatible).
            return (x - 20, y - 20, 40, 40)
        return None
    except Exception:
        return None


def _ground_box(target, tries=2):
    """Vision grounding returning a BOX (x,y,w,h) in 0-1000 space.

    Pass 1: full screen at 1920px asking for a tight box. Pass 2 (retry):
    stricter prompt demanding box JSON. Raises OSRefused with vision's words.
    """
    last_say = ""
    for attempt in range(max(1, tries)):
        _size, jpeg = _shot(max_w=1920)  # max detail for small buttons
        if attempt == 0:
            prompt = (f"Locate this UI element on the Windows screenshot: {target}. "
                      'Reply with ONLY JSON like {"x": 500, "y": 300, "w": 80, "h": 30} '
                      "where x,y,w,h are 0-1000 coordinates of the TIGHT bounding box. "
                      "No other text.")
        else:
            prompt = (f"Look VERY carefully at the Windows screenshot. Find: {target}. "
                      "Your previous answer could not be parsed. Output ONLY "
                      '{"x": <0-1000>, "y": <0-1000>, "w": <box width>, "h": <box height>} '
                      "— the TIGHT box around the element. Nothing else, no fences.")
        try:
            res = _vision(prompt, jpeg)
        except Exception as e:
            raise OSRefused(f"Vision backend failed: {e}")
        content = ((res.get("message", {}) or {}).get("content", "") or "")
        last_say = content
        box = _parse_box(content)
        if box:
            return box
    raise OSRefused(f"Could not ground '{target}' after retries. Vision said: {last_say[:300]}")


def _ground_crop_retry(target, box):
    """Second-look grounding: crop 2x around a box, re-ask vision inside the crop.

    Fixes small-button drift: the crop is upscaled so the model can be exact.
    Returns refined (x,y,w,h) in full-screen 0-1000 space. Falls back to `box`.
    """
    try:
        from PIL import Image
        import base64
        import io
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        # 2x zoom window around center, clamped to screen.
        ww, hh = min(500, max(200, w * 4)), min(400, max(160, h * 4))
        x0, y0 = max(0, cx - ww / 2), max(0, cy - hh / 2)
        _size, _png, full = _shot_png_full()
        fw, fh = full.size
        px0, py0 = int(x0 / 1000 * fw), int(y0 / 1000 * fh)
        px1, py1 = int(min(1000, x0 + ww) / 1000 * fw), int(min(1000, y0 + hh) / 1000 * fh)
        crop = full.crop((px0, py0, px1, py1))
        # Upscale small crops for the model.
        if crop.size[0] < 800:
            crop = crop.resize((crop.size[0] * 2, crop.size[1] * 2), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=75)
        crop_jpeg = buf.getvalue()
        crop_prompt = (f"Inside this CROPPED screenshot, locate: {target}. "
                       'Reply ONLY {"x": <0-1000>, "y": <0-1000>} = center INSIDE THE CROP.')
        try:
            # BUILD 18: crop second-look via the VL GGUF on llama.cpp.
            # No second engine - on failure the original box is kept.
            _rr = _router()
            if _rr is None:
                raise InferenceError("inference package unavailable")
            res = _rr.chat_with_image(crop_prompt, crop_jpeg,
                                      model=_vision_model(), timeout=120)
        except Exception:
            return box
        content = ((res.get("message", {}) or {}).get("content", "") or "")
        m = re.search(r"\{[^{}]*?(\d{1,4})[^\d]+(\d{1,4})", re.sub(r"```(?:json)?|```", "", content))
        if not m:
            return box
        rx, ry = int(m.group(1)), int(m.group(2))
        if not (0 <= rx <= 1000 and 0 <= ry <= 1000):
            return box
        # Crop-space → full-screen-space.
        fx = x0 + rx / 1000 * ww
        fy = y0 + ry / 1000 * hh
        return (fx - w / 2, fy - h / 2, w, h)
    except Exception:
        return box


def _ground_target(target, tries=2):
    """Legacy point grounding (kept for scroll targeting). Returns (x, y)."""
    box = _ground_box(target, tries=tries)
    return int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)


def _click_point(sx, sy, button="left", count=1):
    """Low-level mouse click at physical pixels. Returns (sx, sy)."""
    pa = _pa()
    sw, sh = pa.size()
    if not (0 <= sx < sw * 1.5 and 0 <= sy < sh * 1.5):
        raise OSRefused(f"Mapped point ({sx},{sy}) is off-screen ({sw}x{sh}). Aborted.")
    pa.moveTo(int(sx), int(sy), duration=0.25)
    _sleep(0.15)
    if button == "right":
        pa.rightClick()
    elif count == 2:
        pa.doubleClick(interval=0.12)
    else:
        pa.click()
    time.sleep(0.5)
    return int(sx), int(sy)


def _box_center_to_px(box):
    """0-1000 box → physical pixel center using the REAL screen size."""
    sw, sh = _screen_size()
    cx = (box[0] + box[2] / 2) / 1000 * sw
    cy = (box[1] + box[3] / 2) / 1000 * sh
    return int(cx), int(cy)


def os_click(target, button="left", count=1, verify=None, title=None):
    if not (target or "").strip():
        raise OSRefused("Empty click target — describe the element visually.")
    button = (button or "left").lower()
    if button not in ("left", "right"):
        raise OSRefused(f"Unknown button '{button}'. Use left/right.")
    try:
        count = max(1, min(int(count or 1), 2))
    except Exception:
        count = 1
    if button == "right":
        count = 1  # right double-click is never what anyone wants
    kind = ("Right-click" if button == "right" else ("Double-click" if count == 2 else "Click"))
    method, tail, sx, sy, shot_box = None, "", None, None, None
    # PATH 1 (exact): live UIA tree — real button bounds from Windows itself.
    try:
        _t, ctrls = _uia_controls(title)
        best = _uia_match(ctrls, target)
        if best is not None:
            _uia_click(best, button, count)
            time.sleep(0.4)
            tail = f" Window '{_t}'."
            method = f"UIA control '{best['text']}'"
            try:
                r = best.get("rect")
                if r:
                    sx, sy = int(r[0] + r[2] // 2), int(r[1] + r[3] // 2)
            except Exception:
                pass
        else:
            raise OSRefused("__next_path__")
    except OSRefused as e:
        if "__next_path__" not in str(e):
            raise
        method = None
    # PATH 2 (exact pixels): OCR word boxes — canvas/tray/web text UIA misses.
    if method is None:
        try:
            words = _ocr_words()
            hit = _ocr_match(words, target)
            if hit is not None:
                sx, sy = _box_center_to_px((hit["x"], hit["y"], max(8, hit["w"]), max(8, hit["h"])))
                _click_point(sx, sy, button, count)
                try:
                    fg = _fg_title()
                except Exception:
                    fg = ""
                tail = f" Foreground now '{fg}'." if fg else ""
                method = f"OCR text '{hit['text']}' conf {hit['conf']:.0f} @ ({sx},{sy})"
                shot_box = (hit["x"], hit["y"], max(8, hit["w"]), max(8, hit["h"]))
        except OSRefused:
            raise
        except Exception:
            method = None
    # PATH 3+4 (approximate): vision BOX + crop-zoom refine — games/video/canvas.
    if method is None:
        box = _ground_box(target)
        # Refine small boxes with a 2x crop second look.
        if box[2] * box[3] < 2500:
            box = _ground_crop_retry(target, box)
        sx, sy = _box_center_to_px(box)
        _click_point(sx, sy, button, count)
        try:
            fg = _fg_title()
        except Exception:
            fg = ""
        tail = f" Foreground now '{fg}'." if fg else ""
        method = f"vision box ({sx},{sy})"
        shot_box = box
    out = f"{kind}ed '{target}' via {method}." + tail
    if sx is not None and sy is not None:
        try:
            _hud_mark(sx, sy, f"{kind} {target}")
        except Exception:
            pass
    try:
        _annotate_shot([shot_box] if shot_box else [], f"{kind}: {target}")
    except Exception:
        pass
    if verify:
        # EXACT proof first: live UIA control values (e.g. Display='10').
        try:
            vals = ui_values(title)
            if vals:
                out += " UI state: " + " / ".join(vals[:6])
        except Exception:
            pass
        # OCR re-read catches text changes UIA misses (canvas/web).
        try:
            words = _ocr_words()
            if words:
                top = " / ".join(w["text"] for w in words[:8])
                out += f" OCR: {top[:200]}"
        except Exception:
            pass
        # Then vision yes/no on a fresh screenshot.
        _vsize, vjpeg = _shot()
        try:
            res = _vision(f"Yes/no question about this Windows screenshot (answer YES or NO first, then one short sentence): {verify}", vjpeg)
            verdict = ((res.get("message", {}) or {}).get("content", "") or "").strip()[:300]
        except Exception as e:
            verdict = f"verification failed: {e}"
        out += f" Verify [{verify}]: {verdict}"
        # One auto-retry on explicit NO (small-button drift is the classic miss).
        if verdict.upper().startswith("NO") and shot_box is not None:
            try:
                box2 = _ground_crop_retry(target, shot_box)
                sx2, sy2 = _box_center_to_px(box2)
                _click_point(sx2, sy2, button, count)
                _hud_mark(sx2, sy2, f"retry {target}")
                _annotate_shot([box2], f"retry: {target}")
                out += f" Retried @ ({sx2},{sy2})."
                _vsize2, vjpeg2 = _shot()
                try:
                    res2 = _vision(f"Yes/no question about this Windows screenshot (answer YES or NO first, then one short sentence): {verify}", vjpeg2)
                    verdict2 = ((res2.get("message", {}) or {}).get("content", "") or "").strip()[:300]
                except Exception as e2:
                    verdict2 = f"verification failed: {e2}"
                out += f" Re-verify: {verdict2}"
            except Exception:
                pass
    return out


def os_scroll(direction="down", amount=5, target=None):
    direction = (direction or "down").lower()
    if direction not in ("up", "down"):
        raise OSRefused(f"Unknown scroll direction '{direction}'. Use up/down.")
    try:
        amount = max(1, min(int(amount or 5), 20))
    except Exception:
        amount = 5
    pa = _pa()
    where = "current mouse position"
    if target:
        moved = False
        try:
            _t, ctrls = _uia_controls(None)
            best = _uia_match(ctrls, target)
            if best is not None and best["rect"]:
                r = best["rect"]
                pa.moveTo(r[0] + r[2] // 2, r[1] + r[3] // 2, duration=0.25)
                moved = True
        except Exception:
            pass
        if not moved:
            try:
                words = _ocr_words()
                hit = _ocr_match(words, target)
                if hit is not None:
                    sx, sy = _box_center_to_px((hit["x"], hit["y"], max(8, hit["w"]), max(8, hit["h"])))
                    pa.moveTo(int(sx), int(sy), duration=0.25)
                    moved = True
            except Exception:
                pass
        if not moved:
            x, y = _ground_target(target)
            sw, sh = pa.size()
            pa.moveTo(int(x / 1000 * sw), int(y / 1000 * sh), duration=0.25)
        where = f"'{target}'"
    pa.scroll(-amount if direction == "down" else amount)
    time.sleep(0.3)
    try:
        pos = pa.position()
        _hud_mark(int(pos[0]), int(pos[1]), f"scroll {direction}")
    except Exception:
        pass
    return f"Scrolled {direction} x{amount} at {where}."


def os_drag(from_target, to_target=None, to_x=None, to_y=None, button="left"):
    """Drag from a visual target to another target (or 0-1000 x/y).

    Uses the same UIA → OCR → vision-box pipeline for both ends. Sliders,
    window moves, canvas drags, file drags all go through here.
    """
    if not (from_target or "").strip():
        raise OSRefused("Empty drag source — describe the element visually.")
    button = (button or "left").lower()
    if button not in ("left", "right"):
        raise OSRefused(f"Unknown button '{button}'.")
    pa = _pa()
    sw, sh = _screen_size()

    def _resolve(desc):
        try:
            _t, ctrls = _uia_controls(None)
            best = _uia_match(ctrls, desc)
            if best is not None and best.get("rect"):
                r = best["rect"]
                return int(r[0] + r[2] // 2), int(r[1] + r[3] // 2), f"UIA '{best['text']}'"
        except Exception:
            pass
        try:
            hit = _ocr_match(_ocr_words(), desc)
            if hit is not None:
                sx, sy = _box_center_to_px((hit["x"], hit["y"], max(8, hit["w"]), max(8, hit["h"])))
                return int(sx), int(sy), f"OCR '{hit['text']}'"
        except Exception:
            pass
        box = _ground_box(desc)
        sx, sy = _box_center_to_px(box)
        return int(sx), int(sy), "vision box"

    sx, sy, sm = _resolve(from_target)
    if to_target:
        ex, ey, em = _resolve(to_target)
    elif to_x is not None and to_y is not None:
        try:
            ex, ey = int(float(to_x) / 1000 * sw), int(float(to_y) / 1000 * sh)
            em = "coords"
        except Exception:
            raise OSRefused("to_x/to_y must be 0-1000 numbers.")
    else:
        raise OSRefused("Drag needs to_target or to_x/to_y.")
    try:
        pa.moveTo(sx, sy, duration=0.25)
        _sleep(0.15)
        pa.dragTo(int(ex), int(ey), duration=0.6, button=button)
        time.sleep(0.4)
        _hud_mark(int(ex), int(ey), f"drag {from_target}")
        _annotate_shot([(sx / max(1, sw) * 1000 - 15, sy / max(1, sh) * 1000 - 15, 30, 30,
                         )], f"drag to {to_target or (to_x, to_y)}")
        return f"Dragged '{from_target}' ({sm}) → '{to_target or (to_x, to_y)}' ({em})."
    except OSRefused:
        raise
    except Exception as e:
        raise OSRefused(f"Drag failed: {e}")


# ================= BROWSER DEEP CONTROL (Playwright CDP — real DOM) =================
# Pixels can't see inside Chrome reliably. CDP drives the live browser by
# selector/text — exact, no grounding. Attaches to the running Chrome/Edge
# (needs --remote-debugging-port=9222) else launches a temp Chromium.
# Every function degrades to OSRefused with a one-line fix when unavailable.
_BROWSER_HINT = ("Browser CDP offline. Fix: close Chrome, relaunch with "
                "--remote-debugging-port=9222, then retry. "
                "Example: chrome.exe --remote-debugging-port=9222")


_CDP_CACHE = {"ts": 0.0, "result": None}


def _browser_cdp_url(max_age=5.0):
    """Find the Chrome/Edge CDP endpoint.

    BUILD 17: when no debug port is listening, urlopen blocks for the full
    timeout on each of the 2 ports - ~4s per call, and /api/health calls
    this on every poll. Cache the negative result briefly so health stays
    responsive; a relaunched Chrome is picked up within ``max_age``.
    """
    now = time.time()
    if _CDP_CACHE["result"] is not None and now - _CDP_CACHE["ts"] < max_age:
        return _CDP_CACHE["result"]
    import urllib.request as _ur
    import json as _js
    found = (None, None)
    for port in (9222, 9333):
        try:
            with _ur.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                info = _js.loads(r.read().decode())
                if info.get("webSocketDebuggerUrl"):
                    found = (info["webSocketDebuggerUrl"], port)
                    break
        except Exception:
            continue
    _CDP_CACHE["ts"] = now
    _CDP_CACHE["result"] = found
    return found


def _probe_playwright():
    """Is the playwright package importable? Cached - it cannot change
    without a restart, and the import is the expensive part."""
    try:
        import playwright  # noqa: F401
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "reason": f"pip install playwright ({e})"}


def browser_status():
    """Report CDP + Playwright readiness. Never raises.

    BUILD 17: only the playwright IMPORT is cached. The CDP probe stays
    live on purpose - the user can relaunch Chrome with
    --remote-debugging-port=9222 at any time and health must reflect it.
    """
    pw = _cached_probe("playwright", _probe_playwright)
    if not pw.get("ok"):
        return {"available": False, "reason": pw.get("reason", "playwright unavailable")}
    ws, port = _browser_cdp_url()
    if ws:
        return {"available": True, "mode": f"attach :{port}"}
    return {"available": False, "reason": _BROWSER_HINT}


def _pw_page(action_fn, timeout_ms=20000):
    """Run action_fn(page) against attached Chrome or temp Chromium. Returns result."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise OSRefused(f"Playwright missing: pip install playwright ({e}).")
    ws, _port = _browser_cdp_url()
    with sync_playwright() as p:
        if ws:
            try:
                browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            except Exception as e:
                raise OSRefused(f"{_BROWSER_HINT} ({e})")
            try:
                ctxs = browser.contexts
                page = None
                for c in ctxs:
                    for pg in c.pages:
                        if pg.url and not pg.url.startswith("chrome-"):
                            page = pg
                            break
                    if page:
                        break
                if not page:
                    page = ctxs[0].new_page() if ctxs else browser.new_page()
                page.bring_to_front()
                return action_fn(page)
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
        # No running debuggable browser → temp Chromium (still exact DOM).
        try:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            try:
                return action_fn(page)
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
        except Exception as e:
            raise OSRefused(f"Could not launch Chromium: {e}. {_BROWSER_HINT}")


def browser_tabs():
    def _fn(page):
        try:
            ctx = page.context
            out = []
            for i, pg in enumerate(ctx.pages):
                try:
                    out.append(f"{i}: {pg.title()[:60]} — {pg.url[:120]}")
                except Exception:
                    continue
            return "\n".join(out) or "(no tabs)"
        except Exception as e:
            raise OSRefused(f"tabs failed: {e}")
    return _pw_page(_fn)


def browser_open(url):
    url = (url or "").strip()
    if not url:
        raise OSRefused("Empty URL.")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    def _fn(page):
        page.goto(url, timeout=20000)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        time.sleep(1.0)
        return f"Opened {url}. Title: {page.title()[:100]}"
    out = _pw_page(_fn)
    try:
        _hud_write(state="working", action=f"browser {url}", ok=True)
    except Exception:
        pass
    return out


def browser_click(selector, timeout_ms=8000):
    selector = (selector or "").strip()
    if not selector:
        raise OSRefused("Empty selector — CSS or text=….")
    def _fn(page):
        try:
            page.click(selector, timeout=int(timeout_ms or 8000))
        except Exception:
            # text= fallback: "Play" → text=Play
            if not selector.startswith("text="):
                page.click(f"text={selector}", timeout=int(timeout_ms or 8000))
            else:
                raise
        time.sleep(0.6)
        return f"Clicked '{selector}'. Title now: {page.title()[:100]}"
    return _pw_page(_fn)


def browser_type(selector, text, submit=False):
    selector, text = (selector or "").strip(), str(text or "")
    if not selector or not text:
        raise OSRefused("Need selector + text.")
    if len(text) > 2000:
        raise OSRefused("Type limit 2000 chars.")
    def _fn(page):
        page.fill(selector, text, timeout=8000)
        if submit:
            page.keyboard.press("Enter")
            time.sleep(1.0)
        return f"Typed {len(text)} chars into '{selector}'."
    return _pw_page(_fn)


def browser_read(limit=4000):
    def _fn(page):
        try:
            title = page.title()[:120]
            url = page.url[:160]
        except Exception:
            title, url = "", ""
        try:
            txt = page.inner_text("body", timeout=5000)[:int(limit or 4000)]
        except Exception as e:
            raise OSRefused(f"read failed: {e}")
        return f"{title}\n{url}\n{txt}"
    return _pw_page(_fn)


# ================= CODE EXEC (Open-Interpreter-style, sandboxed) =================
# File/folder/terminal jobs go here — NOT through mouse clicks. The model writes
# a short Python snippet, it runs in %TEMP%/jarvis_code with a timeout, stdout
# comes back as the tool result for self-correction. Blocklist + HITL below.
CODE_AUTO = os.environ.get("JARVIS_CODE_AUTO", "0").lower() in ("1", "true", "on", "yes")
CODE_BLOCK = ("rm -rf", "rd /s", "del /f /s", "format ", "diskpart", "reg delete",
              "regedit", "shutdown", "taskkill /f", ":(){", "mkfs", "dd if=")


def code_status():
    return {"sandbox": "stdlib-subprocess", "auto": CODE_AUTO,
            "timeout_s": 15, "max_log": 4000}


def os_code_exec(code, timeout_s=15):
    """Run a short Python snippet sandboxed (temp dir, timeout, no shell)."""
    code = str(code or "")
    if not code.strip():
        raise OSRefused("Empty code.")
    if len(code) > 6000:
        raise OSRefused("Code limit is 6000 chars per call — split it.")
    low = code.lower()
    for pat in CODE_BLOCK:
        if pat in low:
            raise OSRefused(f"Blocked for safety: pattern '{pat.strip()}'. Confirm step-by-step instead.")
    try:
        timeout_s = max(3, min(int(timeout_s or 15), 60))
    except Exception:
        timeout_s = 15
    import subprocess
    workdir = os.path.join(tempfile.gettempdir(), "jarvis_code")
    try:
        os.makedirs(workdir, exist_ok=True)
    except Exception:
        pass
    tmp = None
    try:
        fd, tmppath = tempfile.mkstemp(suffix=".py", prefix="jarvis_code_", dir=workdir)
        tmp = tmppath
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        proc = subprocess.run(
            [sys.executable, tmppath],
            capture_output=True, text=True, timeout=timeout_s, cwd=workdir,
            shell=False)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        out = out.strip()[:4000] or "(no output)"
        return f"exit={proc.returncode} ({timeout_s}s cap, cwd=jarvis_code):\n{out}"
    except subprocess.TimeoutExpired:
        raise OSRefused(f"Timed out after {timeout_s}s — split the job.")
    except OSRefused:
        raise
    except Exception as e:
        raise OSRefused(f"Code exec failed: {e}")
    finally:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _os_exec(name, args):
    if name == "open_app":
        return os_open(args.get("target", ""))
    if name == "list_apps":
        return os_list_apps(args.get("query", ""), args.get("limit", 12))
    if name == "focus_window":
        mx = args.get("maximize", True)
        if isinstance(mx, str):
            mx = mx.lower() not in ("0", "false", "no", "off")
        return os_focus(args.get("title", ""), maximize=bool(mx))
    if name == "type_text":
        return os_type(args.get("text", ""), args.get("title"))
    if name == "press_keys":
        return os_keys(args.get("keys", ""), args.get("title"))
    if name == "read_window":
        return os_read(args.get("title"))
    if name == "close_window":
        return os_close(args.get("title", ""))
    if name == "see_screen":
        return os_see(args.get("question"))
    if name == "click_target":
        return os_click(args.get("target", ""), args.get("button", "left"),
                        args.get("count", 1), args.get("verify"), args.get("title"))
    if name == "inspect_screen":
        return os_inspect(args.get("title"), args.get("question"))
    if name == "scroll_screen":
        return os_scroll(args.get("direction", "down"), args.get("amount", 5),
                         args.get("target"))
    if name == "drag_screen":
        return os_drag(args.get("from_target", ""), args.get("to_target"),
                       args.get("to_x"), args.get("to_y"), args.get("button", "left"))
    if name == "ocr_read":
        return os_ocr_read(args.get("limit", 40))
    if name == "browser_open":
        return browser_open(args.get("url", ""))
    if name == "browser_click":
        return browser_click(args.get("selector", ""), args.get("timeout_ms", 8000))
    if name == "browser_type":
        return browser_type(args.get("selector", ""), args.get("text", ""), args.get("submit", False))
    if name == "browser_read":
        return browser_read(args.get("limit", 4000))
    if name == "browser_tabs":
        return browser_tabs()
    if name == "code_exec":
        return os_code_exec(args.get("code", ""), args.get("timeout_s", 15))
    raise OSRefused(f"Unknown OS tool {name}.")


def os_human(name, args):
    args = args or {}
    if name == "open_app":
        return f"Open {args.get('target', '?')}"
    if name == "list_apps":
        q = (args.get("query") or "all").strip()[:40] or "all"
        return f"Search installed apps for \u201c{q}\u201d"
    if name == "focus_window":
        mx = " max" if args.get("maximize", True) not in (False, "false", "0", "no", "off") else ""
        return f"Focus window \u201c{args.get('title', '?')}\u201d{mx}"
    if name == "type_text":
        t = str(args.get("text", ""))
        prev = (t[:60] + "\u2026") if len(t) > 60 else t
        tgt = f" in \u201c{args.get('title')}\u201d" if args.get("title") else ""
        return f"Type {len(t)} chars{tgt}: \u201c{prev}\u201d"
    if name == "press_keys":
        tgt = f" in \u201c{args.get('title')}\u201d" if args.get("title") else ""
        return f"Press {args.get('keys', '?')}{tgt}"
    if name == "read_window":
        return f"Read text from {args.get('title') or 'active window'}"
    if name == "close_window":
        return f"Close \u201c{args.get('title', '?')}\u201d"
    if name == "see_screen":
        q = (args.get("question") or "")[:60]
        return "Screenshot + look at the screen" + (f" ({q})" if q else "")
    if name == "inspect_screen":
        t = (args.get("title") or "foreground").strip()[:40] or "foreground"
        return f"Inspect every control in \u201c{t}\u201d (exact UI tree)"
    if name == "click_target":
        extra = ""
        if (args.get("button") or "left") == "right":
            extra = " (right)"
        elif str(args.get("count", 1)) == "2":
            extra = " (double)"
        return f"Click \u201c{args.get('target', '?')}\u201d{extra}"
    if name == "scroll_screen":
        return f"Scroll {(args.get('direction') or 'down')} x{args.get('amount', 5)}"
    if name == "drag_screen":
        return f"Drag \u201c{args.get('from_target', '?')}\u201d → \u201c{args.get('to_target') or (args.get('to_x'), args.get('to_y'))}\u201d"
    if name == "ocr_read":
        return "OCR read screen text"
    if name == "browser_open":
        return f"Browser open {args.get('url', '?')[:60]}"
    if name == "browser_click":
        return f"Browser click \u201c{args.get('selector', '?')}\u201d"
    if name == "browser_type":
        return f"Browser type into \u201c{args.get('selector', '?')}\u201d"
    if name == "browser_read":
        return "Browser read page"
    if name == "browser_tabs":
        return "Browser list tabs"
    if name == "code_exec":
        c = str(args.get("code", ""))[:60].replace("\n", " ")
        return f"Run code: {c}"
    return name


def sess_put(st, sid=None):
    with SESS_LOCK:
        now = time.time()
        for k in [k for k, v in SESSIONS.items() if now - v["ts"] > SESSION_TTL]:
            del SESSIONS[k]
        sid = sid or uuid.uuid4().hex[:16]
        st["ts"] = now
        SESSIONS[sid] = st
        return sid


def sess_get(sid):
    with SESS_LOCK:
        st = SESSIONS.get(sid)
        if not st or time.time() - st["ts"] > SESSION_TTL:
            SESSIONS.pop(sid, None)
            return None
        return st


def sess_drop(sid=None):
    with SESS_LOCK:
        if sid:
            SESSIONS.pop(sid, None)
        else:
            SESSIONS.clear()


def _pending_response(st, sid, pend):
    return ({"pending_approval": {
        "sid": sid, "tool": pend["name"], "human": pend["human"],
        "steps": list(st["steps"])},
        "recalled": [{"id": m.get("id"), "title": m.get("title"),
                      "sector": m.get("sector"), "score": round(s, 3)}
                     for m, s in st["recalled"]],
        "model": st["model"]}, 200)


VIEW_DECLINED = ("Declined: the user requested no view change. "
                 "Do not retry view tools; answer normally.")


def _dbg(it, msg_obj):
    try:
        # BUILD 15: tag each turn with the serving backend + latency.
        extra = {}
        try:
            _rr = _router()
            if _rr is not None:
                extra = {"backend": _rr.last_backend, "kind": _rr.last_kind,
                         "latency_s": _rr.last_latency_s,
                         "tps": _rr.last_tokens_per_s,
                         "mode": _rr.mode}
        except Exception:
            pass
        dbg_log = os.path.join(tempfile.gettempdir(), "jarvis_chat_debug.jsonl")
        with open(dbg_log, "a") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "iter": it,
                                "content": (msg_obj.get("content") or "")[:400],
                                "tools": [((tc.get("function") or {}).get("name")) for tc in (msg_obj.get("tool_calls") or [])],
                                **extra}) + "\n")
    except Exception:
        pass


def _mem_tool(st, name, args):
    """Memory + galaxy-view tools. Mutates st in place. Returns True if handled."""
    messages, actions = st["messages"], st["actions"]
    memories = st["memories"]
    if name == "store_memory":
        sector = (args.get("sector") or "personal").lower()
        if sector not in ("personal", "work", "ideas", "important"):
            sector = "personal"
        mem = {
            "id": f"mem-{int(time.time()*1000)}",
            "title": str(args.get("title", ""))[:80] or "Untitled",
            "content": str(args.get("content", ""))[:2000],
            "sector": sector,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        actions.append({"type": "store_memory", "memory": mem})
        messages.append({"role": "tool", "content": f"Stored '{mem['title']}' in {sector} sector with id {mem['id']}."})
    elif name == "focus_memory":
        mid = args.get("memory_id", "")
        if any(m.get("id") == mid for m in memories):
            actions.append({"type": "focus_memory", "id": mid})
            messages.append({"role": "tool", "content": f"Focused memory {mid} in the 3D galaxy."})
        else:
            messages.append({"role": "tool", "content": f"Unknown memory id {mid}. Ask the user to clarify."})
    elif name == "list_memories":
        sector = (args.get("sector") or "").lower() or None
        lim = max(1, min(int(args.get("limit") or 50), 200))
        items = [m for m in memories if not sector or m.get("sector") == sector]
        if not items:
            messages.append({"role": "tool", "content": "No memories found."})
        else:
            messages.append({"role": "tool", "content":
                f"{len(items)} memories:\n" + "\n".join(fmt_mem(m) for m in items[:lim])})
    elif name == "search_memories":
        q = args.get("query", "")
        lim = max(1, min(int(args.get("limit") or 8), 20))
        hits = recall(q, memories, top_k=lim)
        if not hits:
            messages.append({"role": "tool", "content": f"No matches for '{q}'."})
        else:
            messages.append({"role": "tool", "content":
                "Search results:\n" + "\n".join(fmt_mem(m, s) for m, s in hits)})
    elif name == "edit_memory":
        mid = args.get("memory_id", "")
        m = next((x for x in memories if x.get("id") == mid), None)
        if not m:
            messages.append({"role": "tool", "content": f"Unknown memory id {mid}."})
        else:
            patch = {k: args[k] for k in ("title", "content", "sector")
                     if args.get(k) and args[k] != m.get(k)}
            if patch.get("sector") not in (None, "personal", "work", "ideas", "important"):
                patch.pop("sector", None)
            if not patch:
                messages.append({"role": "tool", "content": "No changes supplied."})
            else:
                m.update(patch)  # keep backend copy in sync for follow-up calls
                actions.append({"type": "edit_memory", "id": mid, "patch": patch})
                messages.append({"role": "tool", "content": f"Edited {mid}: {patch}."})
    elif name == "delete_memory":
        mid = args.get("memory_id", "")
        m = next((x for x in memories if x.get("id") == mid), None)
        if not m:
            messages.append({"role": "tool", "content": f"Unknown memory id {mid}."})
        else:
            st["memories"] = [x for x in memories if x.get("id") != mid]
            actions.append({"type": "delete_memory", "id": mid, "title": m.get("title")})
            messages.append({"role": "tool", "content": f"Deleted '{m.get('title')}' ({mid})."})
    elif name == "set_filter":
        if not st["view_ok"]:
            messages.append({"role": "tool", "content": VIEW_DECLINED})
        else:
            sector = (args.get("sector") or "all").lower()
            if sector not in ("all", "personal", "work", "ideas", "important"):
                messages.append({"role": "tool", "content": f"Invalid sector '{sector}'."})
            else:
                actions.append({"type": "set_filter", "sector": sector})
                messages.append({"role": "tool", "content": f"Galaxy filter set to {sector}."})
    elif name == "galaxy_command":
        if not st["view_ok"]:
            messages.append({"role": "tool", "content": VIEW_DECLINED})
        else:
            cmd = args.get("command", "")
            if cmd not in ("spin_on", "spin_off", "bloom_on", "bloom_off", "labels_on",
                           "labels_off", "tour_start", "tour_stop", "reshuffle",
                           "cinematic_on", "cinematic_off"):
                messages.append({"role": "tool", "content": f"Invalid command '{cmd}'."})
            else:
                actions.append({"type": "galaxy_command", "command": cmd})
                messages.append({"role": "tool", "content": f"Executed {cmd}."})
    else:
        return False
    return True


def _final_response(st):
    actions = st["actions"]
    memories = st["memories"]
    # Drop accidental duplicate tool calls (same action twice in a row).
    deduped = []
    for _a in actions:
        if not (deduped and deduped[-1] == _a):
            deduped.append(_a)
    actions = deduped
    st["actions"] = actions

    def _title(mid):
        m = next((x for x in memories if x.get("id") == mid), None)
        return f"'{m.get('title')}'" if m else mid

    def _confirm(a):
        t = a.get("type")
        if t == "store_memory":
            m = a.get("memory", {})
            return f"stored '{m.get('title')}' in {m.get('sector')}"
        if t == "edit_memory":
            return f"updated {_title(a.get('id'))}"
        if t == "delete_memory":
            return f"purged '{a.get('title') or a.get('id')}'"
        if t == "focus_memory":
            return f"focused {_title(a.get('id'))} in the galaxy"
        if t == "set_filter":
            return f"filter set to {a.get('sector')}"
        if t == "galaxy_command":
            return f"executed {a.get('command', '').replace('_', ' ')}"
        return None

    reply = st.get("reply", "")
    # Small models sometimes prefix role tokens ("assistant assistant ...") or
    # leak tool-talk ("tool read_window executed"). Strip that garbage so the
    # spoken reply stays clean. Never raises.
    try:
        reply = re.sub(r"^\s*(assistant\s+){1,5}", "", reply or "", flags=re.I)
        reply = re.sub(r"^\s*(system|user)\s*:\s*", "", reply or "", flags=re.I)
        reply = re.sub(r"[ \t]{2,}", " ", reply)
    except Exception:
        pass
    # Small models sometimes regurgitate internal context instead of speaking.
    # Detect the dump and substitute a clean in-character confirmation.
    dump_markers = ("GALAXY CENSUS", "RELEVANT MEMORIES", "CURRENT VIEW", "score:", "id:m")
    if not reply or any(mk in reply for mk in dump_markers):
        bits = [c for c in (_confirm(a) for a in actions) if c]
        reply = ("Very good, sir — " + "; ".join(bits) + "." if bits
                 else "Systems nominal, sir — but I drew a blank formulating that. Could you rephrase?")
    # BUILD 15: surface inference fallback so the user always knows which
    # engine served the turn (spec fallback contract).
    try:
        _fb_note = st.get("fallback_notice")
        if _fb_note and _fb_note not in (reply or ""):
            reply = ((reply or "").rstrip() + " " + _fb_note).strip()
    except Exception:
        pass
    # Dedupe recalled chips by id (same memory twice looks broken in chat).
    seen_ids, recalled_out = set(), []
    try:
        for m, s in (st.get("recalled") or []):
            mid = (m or {}).get("id")
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)
            recalled_out.append({"id": m.get("id"), "title": m.get("title"),
                                 "sector": m.get("sector"), "score": round(s, 3)})
    except Exception:
        recalled_out = [{"id": m.get("id"), "title": m.get("title"),
                         "sector": m.get("sector"), "score": round(s, 3)} for m, s in st["recalled"]]
    return {
        "reply": reply,
        "actions": actions,
        "recalled": recalled_out,
        "model": st["model"],
        "shot": bool(st.get("shot")),
    }, 200


# ============ NARRATION GUARD (model-agnostic lie detector) ============
# Some models (observed: ministral-3:8b on multi-step orders) narrate a whole
# plan as if already executed — "YouTube launched, typed X, pressing Enter" —
# without emitting a single tool call, sometimes even inventing stored memories.
# The frontend audit log + galaxy would then contradict the words. This guard
# detects past-tense action claims with zero executed proof this turn, forces
# ONE corrective iteration (call the next tool, not prose), and — if the model
# still refuses to act — substitutes an honest fallback so no lie reaches chat.
OS_CLAIM_RE = re.compile(
    r"(launched|launching|opened\s+\S|opening\s+\S|focused\s+the|focusing|typed\b|typing\b|"
    r"press(?:ed|ing)\b|clicked\b|clicking\b|closed\s+\S|closing\s+\S|search\s+executed|"
    r"now\s+playing|play(?:ing|back)\s+(started|initiated)|organized|renamed|converted|"
    r"queu\w*\s+up\s+results?|browser\s+opened|script\s+(executed|ran|finished)|"
    r"files?\s+(moved|organized|cleaned\s+up)|"
    r"tool\s+[`'\"]?\w+[`'\"]?\s+(tool\s+)?executed|output\s+received|"
    r"read_window|focus_window|inspect_screen|see_screen|click_target|"
    r"summarized\s+the\s+contents|it\s+appears\s+to\s+be)",
    re.I)
MEM_CLAIM_RE = re.compile(
    r"(logged\s+as|memory\s+(updated|saved)|added\s+to\s+\*(personal|work|ideas|important)\*|"
    r"stored\s+'[^']*'\s+in|filed\s+under\s+\w+|committed\s+to\s+(memory|core)|"
    r"noted\s+in\s+the\s+\*\*(personal|work|ideas|important)\*\*\s+sector|"
    r"saved\s+(it\s+)?(to|in|under)\s+)",
    re.I)
MUTATING_MEM = ("store_memory", "edit_memory", "delete_memory")

# Stall detector: model already started executing this turn but stops to ask
# permission for steps the user already ordered. Genuine either/or questions
# ("top hit or a specific artist?") are allowed — only pure proceed-stalls trip it.
CONTINUE_RE = re.compile(
    r"(shall\s+i\s+proceed|want\s+me\s+to\s+(continue|go\s+ahead|proceed)|"
    r"next\s+step.{0,50}i\s*(will|'ll)\s+(type|click|open|press|close|focus)|"
    r"ready\s+.{0,40}what\s+should\s+i\s+(enter|type|click|do))",
    re.I)
CONTINUE_NUDGE = (
    "You already started executing and permission is PRE-GRANTED (autopilot). "
    "Do NOT ask to proceed — emit the NEXT step's tool call NOW, no prose, "
    "no questions. Summarize only when every step has a tool result.")
# Plan-announcement stall: model describes its plan (or "protocols") with zero
# tool calls on a computer-control turn. The first tool call IS the first step.
PLAN_RE = re.compile(
    r"(protocol\s+activated|i'll\s+start\s+with\s+the\s+first\s+step|"
    r"here'?s?\s+my\s+plan|step\s+1\b.{0,60}step\s+2\b|"
    r"i\s+will:\s*\n?\s*1\.)",
    re.I)
# Permission stall: asking to proceed on an autopilot turn (never correct —
# permission is pre-granted; genuine either/or choices are still allowed).
PERMIT_RE = re.compile(
    r"(shall\s+i\s+proceed|should\s+i\s+(proceed|continue|go\s+ahead)|"
    r"want\s+me\s+to\s+(proceed|continue)\b|"
    r"do\s+i\s+have\s+(your\s+)?permission)",
    re.I)
# Fake tool-call prose: narrating a call ("(Tool call: list_apps…)") instead of
# emitting it. Excluded when the model is correctly deferring scope ("as the
# user didn't ask for it").
FAKECALL_RE = re.compile(
    r"(\(\s*tool\s*calls?\s*:|calling\s+\w+\s+with|tool\s+call\s*:)",
    re.I)
# Mid-task "shall I …?" (any verb) after executing: pure stall, not a question.
STALL_ASK_RE = re.compile(r"shall\s+i\b", re.I)
# Clarification stall: explicit order in, questionnaire out.
CLARIFY_RE = re.compile(
    r"(what\s+exactly\s+do\s+you\s+want|no\s+guesswork|"
    r"from\s+your\s+first\s+instruc|please\s+(specify|tell\s+me\s+exactly)|"
    r"which\s+exact\s+(app|action|step))",
    re.I)
PLAN_NUDGE = (
    "You announced a plan instead of acting. Plans are worthless here — the "
    "FIRST tool call IS the first step. Emit ONE tool call for it NOW, no "
    "breakdowns, no protocol announcements.")

NARRATION_GUARD_MSG = (
    "PROTOCOL VIOLATION: you just described {kind} actions as DONE, but the tool "
    "results above show ZERO executed proof this turn. The user sees a verified "
    "audit log next to your words — false claims destroy trust instantly. "
    "Fix it NOW: emit ONE tool call for the next step and WAIT for its result. "
    "No prose about clicks, typing, launches, or stored memories until a tool "
    "result confirms them. If the task is genuinely blocked, say what is blocking "
    "you instead of inventing progress.")

MINISTRAL_PROTOCOL = (
    "MODEL-SPECIFIC PROTOCOL — read carefully, it overrides your habits: you "
    "narrate multi-step plans as if already done. That is FORBIDDEN here. You "
    "control this PC ONLY through tool calls, ONE per turn. After each call you "
    "receive a tool result — that result is the ONLY truth about what happened. "
    "Work EVERY order to completion across turns: each turn emit ONE tool call "
    "for the next unfinished step; when its result arrives, emit the NEXT step's "
    "call. Autopilot means permission is PRE-GRANTED — NEVER stall mid-task to "
    "ask it ('Shall I proceed?', 'Next step: I'll…', 'ready — what should I…'). "
    "Write your summary ONLY when all steps have tool results, or when genuinely "
    "blocked (login screens, ambiguity, failure) — and a choice the user already "
    "made ('random', 'top hit') is NOT ambiguity. NEVER write launched / typed / "
    "clicked / opened / playing / stored / logged before the tool result arrives. "
    "Memory stores work the same way: call store_memory first, confirm AFTER. "
    "Capabilities you DO have (never deny them): see_screen (screenshots+vision), "
    "read_window, ocr_read, open_app/focus_window/type_text/press_keys/close_window/click_target/scroll_screen/drag_screen, "
    "browser_open/browser_click/browser_type/browser_read/browser_tabs, code_exec for files. "
    "Clicks: ONLY click_target (button left/right, count 1/2) with verify — "
    "never ctrl+L-style workarounds, never 'clicked' without a tool result. "
    "Web: ONLY browser_* by selector/text, never pixel-click the page. "
    "Workflow per screen task: inspect_screen FIRST (UIA names + OCR boxes), then "
    "click_target with those names + verify (or browser_* in Chrome), chaining to "
    "completion without asking permission.")


def _order_on(st):
    """True when this turn carries an explicit computer/memory order."""
    return bool(st.get("os_on") or st.get("mem_order"))


def _claimed_without_proof(st, steps_at_entry, acts_at_entry):
    """Return 'os' / 'memory' if the draft reply claims executed actions with
    no executed proof this turn, else None."""
    reply = st.get("reply") or ""
    if not reply:
        return None
    if st.get("os_on"):
        os_executed = len(st["steps"]) > steps_at_entry
        if not os_executed and OS_CLAIM_RE.search(reply):
            return "os"
    new_acts = [a.get("type") for a in st["actions"][acts_at_entry:]]
    mem_executed = any(t in MUTATING_MEM for t in new_acts)
    if not mem_executed and MEM_CLAIM_RE.search(reply):
        return "memory"
    return None


def _honest_fallback(st):
    """Deterministic truthful reply when the model keeps narrating without acting."""
    bits = []
    for s in st["steps"]:
        bits.append(s.lstrip("\u2713 ").lstrip("\u2717 ").lstrip("\u2298 "))
    if bits:
        done = "; ".join(bits[-3:])
        return (f"Understood, sir — so far this turn I have: {done}. "
                "Tell me the next step and I will carry it out.")
    return ("Noted, sir — I have not acted yet, only planned. "
            "Give me the first step and I will execute it right now.")


def _run_agent(st, max_iters=8):
    messages, model = st["messages"], st["model"]
    use_tools, os_on, view_ok = st["use_tools"], st["os_on"], st["view_ok"]
    tools = None
    if use_tools:
        tools = TOOLS + OS_TOOL_SCHEMAS if os_on else TOOLS
    steps_at_entry, acts_at_entry = len(st["steps"]), len(st["actions"])
    guard_trips = 0
    it = 0
    while it < max_iters + 2:  # +2 headroom for narration-guard retries
        it += 1
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.6, "num_ctx": 8192}}
        if tools:
            payload["tools"] = tools
        try:
            # BUILD 18: every model call goes through ModelRouter. OS-capable
            # turns (os_on) reason via kind="tool" so the Tool Brain contract
            # applies. One backend, so a failure is the final failure and the
            # real cause is reported instead of a generic "all backends" line.
            kind = "tool" if os_on else "chat"
            res, _fb_note = _router_chat(payload, kind=kind)
            # Report the engine that actually reasoned this turn.
            try:
                _rr = _router()
                if _rr is not None:
                    st["model"] = _rr.llamacpp.label("tool" if os_on else "chat")
            except Exception:
                pass
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:500]
            return {"error": f"Inference failed: HTTP {e.code} {detail}",
                    "model": model}, 502
        except Exception as e:
            return {"error": f"Inference failed: {e}", "model": model}, 502
        msg = res.get("message", {})
        _dbg(it, msg)
        tool_calls = msg.get("tool_calls") or []
        if msg.get("content"):
            st["reply"] = msg["content"]
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls} if tool_calls else
                        {"role": "assistant", "content": msg.get("content") or ""})
        if not tool_calls:
            if tools and guard_trips < 2 and _claimed_without_proof(st, steps_at_entry, acts_at_entry):
                kind = _claimed_without_proof(st, steps_at_entry, acts_at_entry)
                guard_trips += 1
                messages.append({"role": "system",
                                 "content": NARRATION_GUARD_MSG.format(kind=f"computer/memory ({kind})")})
                continue
            reply_txt2 = st.get("reply") or ""
            if (tools and _order_on(st) and guard_trips < 2
                    and len(st["steps"]) > steps_at_entry
                    and (CONTINUE_RE.search(reply_txt2)
                         or STALL_ASK_RE.search(reply_txt2))):
                guard_trips += 1
                messages.append({"role": "system", "content": CONTINUE_NUDGE})
                continue
            if (tools and _order_on(st) and guard_trips < 2
                    and len(st["steps"]) == steps_at_entry
                    and CLARIFY_RE.search(reply_txt2)):
                guard_trips += 1
                messages.append({"role": "system", "content": PLAN_NUDGE})
                continue
            if (tools and _order_on(st) and guard_trips < 2
                    and len(st["steps"]) == steps_at_entry
                    and PLAN_RE.search(st.get("reply") or "")):
                guard_trips += 1
                messages.append({"role": "system", "content": PLAN_NUDGE})
                continue
            reply_txt = st.get("reply") or ""
            if (tools and _order_on(st) and guard_trips < 2
                    and len(st["steps"]) == steps_at_entry
                    and (PERMIT_RE.search(reply_txt)
                         or (FAKECALL_RE.search(reply_txt)
                             and "didn't ask" not in reply_txt
                             and "did not ask" not in reply_txt))):
                guard_trips += 1
                messages.append({"role": "system", "content": PLAN_NUDGE})
                continue
            break
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            name, args = fn.get("name"), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name in OS_TOOL_NAMES and not _needs_approval(name):
                # Autopilot / read-only OS tools execute immediately, no card.
                # (Memory/galaxy tools must fall through to _mem_tool below —
                # routing them to _os_exec fails as "Unknown OS tool".)
                human = os_human(name, args)
                try:
                    _hud_write(state="working", action=human)
                    result = _os_exec(name, args)
                    # BUILD 15 — Tool Brain validation: a hard-failed tool
                    # result becomes a retry nudge (not narrated success).
                    try:
                        if _ToolBrain is not None:
                            _nudge = _ToolBrain.failure_nudge(name, result)
                            if _nudge:
                                messages.append({"role": "system",
                                                 "content": _nudge})
                    except Exception:
                        pass
                    if name in ("see_screen", "click_target", "inspect_screen", "drag_screen",
                                "ocr_read", "open_app", "focus_window", "read_window"):
                        st["shot"] = True  # frontend shows "what he sees"
                    if name in ("open_app", "focus_window", "read_window"):
                        # These tools take no screenshot themselves — snap one so
                        # the chat thumbnail + HUD prove what happened.
                        try:
                            _shot(max_w=1280)
                        except Exception:
                            pass
                    st["steps"].append("\u2713 " + human)
                    messages.append({"role": "tool", "content": f"OS executed [{name}]: {result}"})
                    AUTO_LOG.append({"t": time.strftime("%H:%M:%S"), "action": human, "ok": True})
                    _hud_write(state="working", action=human, ok=True)
                except Exception as e:
                    st["steps"].append("\u2717 " + human + f" ({e})")
                    messages.append({"role": "tool", "content": f"OS FAILED [{name}]: {e}. Tell the user plainly and suggest an alternative."})
                    AUTO_LOG.append({"t": time.strftime("%H:%M:%S"), "action": human, "ok": False})
                    try:
                        _hud_write(state="working", action=human, ok=False)
                    except Exception:
                        pass
                del AUTO_LOG[:-200]
                continue
            if name in OS_TOOL_NAMES:
                human = os_human(name, args)
                st["pending"] = {"name": name, "args": args, "human": human}
                sid = sess_put(st, st.get("sid"))
                st["sid"] = sid
                return _pending_response(st, sid, st["pending"])
            elif not _mem_tool(st, name, args):
                messages.append({"role": "tool", "content": f"Unknown tool {name}."})

    if tools and _claimed_without_proof(st, steps_at_entry, acts_at_entry):
        st["reply"] = _honest_fallback(st)
    return _final_response(st)


def handle_chat(body):
    message = (body.get("message") or "").strip()
    if not message:
        return {"error": "empty message"}, 400
    model = body.get("model") or DEFAULT_MODEL
    memories = body.get("memories") or []
    history = (body.get("history") or [])[-MAX_HISTORY:]
    ui = body.get("ui_state") or {}

    import re as _re
    WORDS = set(_re.findall(r"[a-z]+", message.lower()))

    def _has(stems):
        return any(w.startswith(s) for w in WORDS for s in stems)

    recalled = recall(message, memories)
    context = build_context(recalled, memories, ui)
    # Summary/count intents need the whole ledger, not just top-K recall.
    EXPAND_WORDS = ("summar", "list all", "list my", "everything", "entire galaxy",
                    "all memor", "how many", "count", "overview", "recap",
                    "brief", "agenda", "today", "deadline", "upcoming", "priorit")
    STORE_PHRASES = ("remember this", "remember that", "don't forget", "do not forget",
                     "memorize", "memorise", "take note", "make a note", "jot this",
                     "save this", "save that", "log this", "store this", "commit to memory")
    ml = message.lower()
    ledger_attached = bool(memories and any(w in ml for w in EXPAND_WORDS))
    if ledger_attached:
        context += ("\nFULL GALAXY LEDGER (every star — answer from this):\n" + "\n".join(
            f"- id:{m.get('id')} | {m.get('sector')} | {m.get('title')}: {m.get('content')}"
            for m in memories[:60]))
    # Pure Q&A over an attached ledger needs no tools — and small models relay
    # tool results unreliably, so answer tool-free in that case (proven faster
    # and more accurate). Multi-intent messages keep full tool access.
    mutate = (any(p in ml for p in STORE_PHRASES)
              or _has(("delet", "purg", "forget", "edit", "chang", "updat", "mov",
                       "retitl", "stor", "sav", "not", "focu", "sho", "ope", "filte",
                       "spi", "bloo", "tou", "reshuffl", "cinemat", "hid", "zoo",
                       "displa", "secto", "label")))
    use_tools = not (ledger_attached and not mutate)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"][:2000]})
    messages.append({"role": "user", "content": f"{context}\n\nUSER MESSAGE: {message}"})
    if AUTOPILOT and _REQUIRE_APPROVAL:
        messages.append({"role": "system", "content":
            "POLICY: these computer tools still ask the user for approval instead of "
            "executing immediately: " + ", ".join(sorted(_REQUIRE_APPROVAL)) +
            ". Call them normally; the backend pauses for approval. All other computer "
            "tools execute immediately — report their verified tool results truthfully."})
    if "ministral" in model.lower():
        messages.append({"role": "system", "content": MINISTRAL_PROTOCOL})

    # View tools only engage when the message actually asks about the view.
    view_ok = _has(("filte", "secto", "sho", "displa", "spi", "bloo", "tou",
                    "label", "cinemat", "reshuffl", "view", "only", "hid", "zoo", "visual"))
    # OS tools engage on computer-control intent. Memory-store phrases alone
    # (e.g. "save this memory") don't qualify unless a screen word is present.
    os_on = _has(("ope", "launc", "star", "run", "exec", "notepa", "clic", "typ",
                  "pres", "key", "windo", "scree", "browse", "youtu", "folde",
                  "save", "file", "explor", "calc", "paint", "settin", "volum",
                  "close", "termin", "comma", "app", "program", "instal",
                  "spotif", "discord", "steam", "vscode", "code",
                  "excel", "word", "powerp", "outlook", "teams", "zoom",
                  "vlc", "media", "player", "photo", "music", "game",
                  "scrol", "drag", "doubl", "right", "menu", "button",
                  "cursor", "mous", "inspec", "contro", "structur",
                  "everyth", "dialog", "checkb", "radio", "dropd",
                  "calcul", "result", "plus", "minus", "equal", "display",
                  "ocr", "tabs", "selecto", "dom", "slid", "drop",
                  "chrome", "edge", "firefox", "websit", "link", "search",
                  "do", "stuff", "play", "mute", "song", "video",
                  "organ", "renam", "conver", "scrip", "downloa", "docume",
                  "pdf", "csv", "backu", "sort", "clean", "duplicat",
                  "zip", "unzip", "folder", "directo", "python", "terminal")) \
        and not (any(p in ml for p in STORE_PHRASES) and not _has(
            ("windo", "scree", "clic", "notepa", "browse", "explor", "calc",
              "paint", "settin", "termin", "volum")))
    # Computer: prefix forces OS tools (escape hatch for vague orders).
    if ml.startswith("computer:") or ml.startswith("jarvis computer"):
        os_on = True
    mem_order = any(p in ml for p in STORE_PHRASES)
    # Tool-turn brain (BUILD 15/18: dedicated Tool Brain layer): computer/code
    # turns reason via the installed tool-caller (the resident Qwen2.5-Coder
    # GGUF) on llama.cpp. Chat keeps the user's selected model. The chat model
    # NEVER executes OS tools directly - every OS-capable turn carries the
    # Tool Brain contract.
    tool_model = _tool_model(model)
    try:
        _rr0 = _router()
    except Exception:
        _rr0 = None
    _tool_via_cpp = (_rr0 is not None and _ToolBrain is not None)
    if os_on and tool_model != model:
        if _tool_via_cpp:
            try:
                _engine = _rr0.llamacpp.label("tool") + " (llama.cpp, local GGUF)"
                messages.append({"role": "system", "content":
                    _ToolBrain.system_message(_engine)["content"]})
            except Exception:
                messages.append({"role": "system", "content":
                    f"TOOL ROUTER: this turn needs computer/code tools — reasoning engine is {tool_model}. "
                    "Emit ONE tool call per turn, wait for results, never narrate unexecuted steps."})
        else:
            messages.append({"role": "system", "content":
                f"TOOL ROUTER: this turn needs computer/code tools — reasoning engine is {tool_model}. "
                "Emit ONE tool call per turn, wait for results, never narrate unexecuted steps."})
        model = tool_model
    elif os_on and _tool_via_cpp:
        # Same-model OS turn, but the Tool Brain backend still owns reasoning.
        try:
            _engine = _rr0.llamacpp.label("tool") + " (llama.cpp, local GGUF)"
            messages.append({"role": "system", "content":
                _ToolBrain.system_message(_engine)["content"]})
        except Exception:
            pass
    # Identity grounding: small models confabulate their identity ("Mistral
    # Large 8x7B", "GPT-4"...). State the ACTUAL serving engine every turn so
    # "which model are you?" is answered truthfully. Never raises.
    # BUILD 18: there is one engine, so the label is unconditional. The old
    # branch here emitted "Ollama (local)" - the exact string that made
    # JARVIS claim to be ministral-3:8b in the screenshots.
    try:
        if _rr0 is not None and _ToolBrain is not None:
            _id_model = _rr0.llamacpp.label("tool" if os_on else "chat")
        else:
            _id_model = tool_model if os_on else model
        _id_backend = "llama.cpp (local GGUF)"
        messages.append({"role": "system", "content":
            f"IDENTITY - fact, not roleplay: you are JARVIS, powered by "
            f"'{_id_model}' running locally via {_id_backend}. If the user asks "
            f"which model you are, answer exactly: '{_id_model} (local, via "
            f"{_id_backend})'. NEVER claim to be GPT-4, Mistral Large, or any "
            f"other model you are not. There is no Ollama and no cloud model "
            f"in this system - you are entirely local."})
    except Exception:
        pass
    state = {
        "messages": messages, "model": model, "memories": memories,
        "actions": [], "steps": [], "reply": "",
        "use_tools": use_tools, "os_on": os_on, "view_ok": view_ok,
        "mem_order": mem_order,
        "recalled": recalled, "sid": None, "pending": None,
    }
    return _run_agent(state)


def handle_approve(body):
    sid = body.get("sid", "")
    approved = bool(body.get("approved", False))
    st = sess_get(sid)
    if not st:
        return {"error": "Approval session expired — ask JARVIS to start the task over."}, 404
    pend = st.pop("pending", None)
    if not pend:
        return {"error": "Nothing pending approval."}, 400
    name, args, human = pend["name"], pend["args"], pend["human"]
    if approved:
        try:
            _hud_write(state="working", action=human)
            result = _os_exec(name, args)
            # BUILD 15 — Tool Brain validation (approval path, same contract).
            try:
                if _ToolBrain is not None:
                    _nudge = _ToolBrain.failure_nudge(name, result)
                    if _nudge:
                        st["messages"].append({"role": "system",
                                               "content": _nudge})
            except Exception:
                pass
            if name in ("see_screen", "click_target", "inspect_screen", "drag_screen",
                        "ocr_read", "open_app", "focus_window", "read_window"):
                st["shot"] = True  # frontend shows "what he sees"
            if name in ("open_app", "focus_window", "read_window"):
                try:
                    _shot(max_w=1280)
                except Exception:
                    pass
            st["steps"].append("\u2713 " + human)
            st["messages"].append({"role": "tool", "content": f"OS executed [{name}]: {result}"})
            AUTO_LOG.append({"t": time.strftime("%H:%M:%S"), "action": human, "ok": True})
            _hud_write(state="working", action=human, ok=True)
        except Exception as e:
            st["steps"].append("\u2717 " + human + f" ({e})")
            st["messages"].append({"role": "tool", "content": f"OS FAILED [{name}]: {e}. Tell the user plainly and suggest an alternative."})
            AUTO_LOG.append({"t": time.strftime("%H:%M:%S"), "action": human, "ok": False})
        del AUTO_LOG[:-200]
    else:
        st["steps"].append("\u2298 Declined: " + human)
        st["messages"].append({"role": "tool", "content": f"User DECLINED [{name}]: {human}. Do not retry this exact action; propose an alternative or finish up."})
    sess_put(st, sid=sid)
    return _run_agent(st)


def handle_kill(body):
    sid = body.get("sid")
    sess_drop(sid)
    return {"stopped": True}, 200


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            # BUILD 18: no Ollama probe. Health reflects the local GGUF
            # backend only; "ok" means llama.cpp is importable and a text
            # GGUF was found.
            try:
                names = _available_models()
                base = {"ok": True, "models": names,
                        "default_model": DEFAULT_MODEL, "embed_model": EMBED_MODEL,
                        "build": BUILD, "vision": _vision_status(),
                        "stt": stt_status(), "ocr": ocr_status(),
                        "browser": browser_status(), "code": code_status(),
                        "grounder": grounder_status(),
                        "tool_router": list(TOOL_MODEL_PREF),
                        "screen": {"size": list(_screen_size())}}
                # BUILD 15: unified inference details (spec health contract).
                try:
                    _ih = _inference_health_view()
                    base.update({"inference_mode": _ih.get("inference_mode"),
                                 "backend": _ih.get("backend"),
                                 "model_loaded": _ih.get("model_loaded"),
                                 "gpu_layers": _ih.get("gpu_layers"),
                                 "context": _ih.get("context"),
                                 "model": _ih.get("model"),
                                 "inference": _ih})
                except Exception:
                    pass
                return self._json(base)
            except Exception as e:
                try:
                    ocr = ocr_status()
                except Exception:
                    ocr = {"available": False}
                try:
                    br = browser_status()
                except Exception:
                    br = {"available": False}
                base = {"ok": False, "error": str(e),
                        "build": BUILD, "stt": stt_status(),
                        "ocr": ocr, "browser": br,
                        "code": code_status()}
                # Inference state stays visible even on error.
                try:
                    _ih = _inference_health_view()
                    base.update({"inference_mode": _ih.get("inference_mode"),
                                 "backend": _ih.get("backend"),
                                 "model_loaded": _ih.get("model_loaded"),
                                 "gpu_layers": _ih.get("gpu_layers"),
                                 "context": _ih.get("context"),
                                 "model": _ih.get("model"),
                                 "inference": _ih})
                except Exception:
                    pass
                return self._json(base, 503)
        if self.path == "/api/models":
            # BUILD 18: GGUF inventory only. No Ollama tags.
            body = {"models": [], "default": DEFAULT_MODEL}
            try:
                r = _router()
                if r is not None:
                    for item in (r.llamacpp.inventory() or []):
                        if item.get("kind") == "text":
                            body["models"].append({
                                "name": item.get("name"),
                                "file": item.get("file"),
                                "size": item.get("size"),
                                "tools": True, "source": "llamacpp"})
            except Exception as e:
                body["error"] = str(e)[:200]
            if not body["models"]:
                return self._json({"error": "no local GGUF models found in models/",
                                   "default": DEFAULT_MODEL}, 503)
            try:
                _ih = _inference_health_view()
                body["inference_mode"] = _ih.get("inference_mode")
                _cpp = _ih.get("llamacpp") or {}
                body["llamacpp_models"] = _cpp.get("loaded_roles", [])
                body["llamacpp_model"] = _cpp.get("model")
            except Exception:
                pass
            return self._json(body)
        if self.path == "/api/log":
            body = {"autopilot": AUTOPILOT,
                    "require_approval": sorted(_REQUIRE_APPROVAL),
                    "log": AUTO_LOG[-200:]}
            # BUILD 15: performance metrics live in the audit log too.
            try:
                body["inference"] = _inference_health_view()
            except Exception:
                pass
            return self._json(body)
        if self.path.startswith("/api/apps"):
            qs = parse_qs(urlparse(self.path).query)
            q = (qs.get("q", [""])[0] or "").strip()[:80]
            try:
                lim = max(1, min(int((qs.get("limit", ["12"])[0] or 12)), 50))
            except Exception:
                lim = 12
            try:
                hits = _guess_app(q, limit=lim) if q else []
                if not q:
                    idx = _build_app_index()
                    names = sorted({s["display"] for s in idx["shortcuts"]})[:lim]
                    return self._json({"query": q, "count": len(idx["shortcuts"]),
                                       "app_paths": len(idx["app_paths"]), "sample": names})
                return self._json({"query": q, "hits": [
                    {"display": h["display"], "kind": h["kind"], "score": h["score"]}
                    for h in hits]})
            except Exception as e:
                return self._json({"error": str(e)}, 500)
        if self.path.startswith("/api/speak"):
            qs = parse_qs(urlparse(self.path).query)
            text = (qs.get("text", [""])[0] or "").strip()[:800]
            voice = (qs.get("voice", [TTS_VOICE])[0] or TTS_VOICE).strip()[:64]
            if not text:
                return self._json({"error": "empty text"}, 400)
            key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest() + ".mp3"
            fp = os.path.join(TTS_CACHE, key)
            if not os.path.exists(fp):
                try:
                    tts_generate(text, voice, fp + ".tmp")
                    os.replace(fp + ".tmp", fp)
                except Exception as e:
                    return self._json({"error": f"TTS failed: {e}", "fallback": "browser"}, 502)
            try:
                with open(fp, "rb") as f:
                    data = f.read()
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/api/shot"):
            # /api/shot = freshest screenshot; /api/shot_annotated = numbered boxes.
            annotated = self.path.startswith("/api/shot_annotated")
            try:
                shot = LAST_SHOT_ANNOTATED if annotated else LAST_SHOT
            except Exception:
                shot = {"jpeg": None}
            if not shot.get("jpeg") and annotated:
                try:
                    shot = LAST_SHOT
                except Exception:
                    shot = {"jpeg": None}
            if not shot.get("jpeg"):
                return self._json({"error": "no screenshot yet — ask JARVIS to look at the screen first"}, 404)
            data = shot["jpeg"]
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/hud":
            try:
                hud_path = os.path.join(tempfile.gettempdir(), "jarvis_hud.json")
                with open(hud_path, "r") as f:
                    obj = json.loads(f.read() or "{}")
            except Exception:
                obj = {"state": "idle", "action": "", "t": ""}
            try:
                obj["log_tail"] = (AUTO_LOG[-5:] if AUTO_LOG else [])
            except Exception:
                pass
            return self._json(obj)
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/transcribe":
            ctype = self.headers.get("Content-Type", "")
            try:
                length = int(self.headers.get("Content-Length", 0))
            except Exception:
                length = 0
            if length <= 0 or length > 12 * 1024 * 1024:
                return self._json({"error": "empty or oversized audio (max 12MB)"}, 400)
            try:
                raw = self.rfile.read(length)
            except Exception as e:
                return self._json({"error": f"read failed: {e}"}, 400)
            _fn, audio = parse_multipart_file(raw, ctype)
            if not audio:
                return self._json({"error": "no 'audio' file part in upload"}, 400)
            suffix = ".webm"
            low = (_fn or "").lower()
            for ext in (".webm", ".wav", ".mp3", ".m4a", ".ogg"):
                if low.endswith(ext):
                    suffix = ext
                    break
            tmp = None
            try:
                fd, tmppath = tempfile.mkstemp(suffix=suffix, prefix="jarvis_stt_")
                tmp = tmppath
                with os.fdopen(fd, "wb") as f:
                    f.write(audio)
                text = stt_transcribe_file(tmppath)
                if not text:
                    return self._json({"error": "empty transcript — try speaking closer/louder"}, 502)
                return self._json({"text": text, "model": STT_MODEL_NAME}, 200)
            except OSRefused as e:
                return self._json({"error": str(e), "stt_available": False}, 503)
            except Exception as e:
                return self._json({"error": f"transcribe failed: {e}"}, 500)
            finally:
                try:
                    if tmp and os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        if self.path not in ("/api/chat", "/api/os_approve", "/api/os_kill"):
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:
            return self._json({"error": "invalid JSON"}, 400)
        if self.path == "/api/os_approve":
            obj, code = handle_approve(body)
        elif self.path == "/api/os_kill":
            obj, code = handle_kill(body)
        else:
            obj, code = handle_chat(body)
        return self._json(obj, code)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    port = 8000
    args = sys.argv[1:]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    # BUILD 18: --ollama and --inference are gone. llama.cpp is the only
    # backend; a stale JARVIS_INFERENCE value in the environment is ignored.
    root = os.path.dirname(os.path.abspath(__file__))
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=root))
    print(f"JARVIS backend BUILD {BUILD} on http://127.0.0.1:{port}  "
          f"(llama.cpp only - Ollama removed)", flush=True)
    # BUILD 17/18: verify the local vision path (VL GGUF + mmproj) at boot.
    try:
        threading.Thread(target=probe_vision_backend, daemon=True).start()
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
