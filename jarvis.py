"""J.A.R.V.I.S. single entry — backend + floating HUD + browser, one console.

Replaces opening two terminals + browser by hand (old JARVIS.bat did three
separate `start` calls). Includes a watchdog: if the backend dies (the known
between-sessions death, cause still unknown), the tail is logged to
%TEMP%\\jarvis_crash.log and the backend is restarted (max 5 times).

  python jarvis.py [--port 8000] [--ollama http://127.0.0.1:11434]
                   [--inference ollama|llamacpp|hybrid]
                   [--no-hud] [--no-browser]
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or "python"
CRASH_LOG = os.path.join(tempfile.gettempdir(), "jarvis_crash.log")


def parse_args(argv):
    port, ollama = 8000, None
    no_hud, no_browser = False, False
    inference = None
    if "--port" in argv:
        try:
            port = int(argv[argv.index("--port") + 1])
        except Exception:
            pass
    if "--ollama" in argv:
        try:
            ollama = argv[argv.index("--ollama") + 1].rstrip("/")
        except Exception:
            pass
    if "--inference" in argv:
        # BUILD 15: ollama|llamacpp|hybrid (default hybrid via server.py).
        try:
            cand = argv[argv.index("--inference") + 1].strip().lower()
            if cand in ("ollama", "llamacpp", "hybrid"):
                inference = cand
        except Exception:
            pass
    if "--no-hud" in argv:
        no_hud = True
    if "--no-browser" in argv:
        no_browser = True
    return port, ollama, no_hud, no_browser, inference


def health_ok(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def pump(name, proc):
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            print(f"[{name}] {line.rstrip()}", flush=True)
    except Exception:
        pass


def start_backend(port, ollama, inference=None):
    cmd = [PY, os.path.join(HERE, "server.py"), "--port", str(port)]
    if ollama:
        cmd += ["--ollama", ollama]
    if inference:
        cmd += ["--inference", inference]
    proc = subprocess.Popen(
        cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    threading.Thread(target=pump, args=("backend", proc), daemon=True).start()
    return proc


def start_hud(port):
    proc = subprocess.Popen(
        [PY, os.path.join(HERE, "hud.py"), "--port", str(port)],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    threading.Thread(target=pump, args=("hud", proc), daemon=True).start()
    return proc


def log_crash(text):
    try:
        with open(CRASH_LOG, "a") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{text[:3000]}\n")
    except Exception:
        pass


def main():
    port, ollama, no_hud, no_browser, inference = parse_args(sys.argv[1:])
    print("============================================", flush=True)
    print(" J.A.R.V.I.S. MEMORY GALAXY — single entry", flush=True)
    print(f" port {port}  (hud={'off' if no_hud else 'on'}, "
          f"browser={'off' if no_browser else 'on'})", flush=True)
    print("============================================", flush=True)

    backend = start_backend(port, ollama, inference)
    hud = None

    # Wait for backend warm-up (~20s max).
    for _ in range(40):
        time.sleep(0.5)
        if backend.poll() is not None:
            break
        if health_ok(port):
            break
    if backend.poll() is not None:
        print("[jarvis] backend exited during warm-up — see log above.", flush=True)
    elif not health_ok(port):
        print("[jarvis] backend did not answer /api/health in 20s — continuing anyway.", flush=True)
    else:
        print(f"[jarvis] backend up on http://127.0.0.1:{port}", flush=True)

    if not no_hud and os.path.exists(os.path.join(HERE, "hud.py")):
        hud = start_hud(port)
        print("[jarvis] HUD overlay starting (ghost cursor + blue aura).", flush=True)

    if not no_browser:
        try:
            time.sleep(1.0)
            webbrowser.open(f"http://127.0.0.1:{port}")
            print("[jarvis] browser opening.", flush=True)
        except Exception as e:
            print(f"[jarvis] browser open failed: {e}", flush=True)

    print("[jarvis] running — close this window (or Ctrl+C) to stop all.", flush=True)

    # Watchdog: restart a dead backend, max 5 times (known intermittent death).
    restarts = 0
    try:
        while True:
            time.sleep(10)
            if backend.poll() is not None:
                rc = backend.poll()
                msg = f"backend died (exit={rc}, restarts={restarts})"
                print(f"[jarvis] {msg} — logging + restarting.", flush=True)
                log_crash(msg)
                if restarts >= 5:
                    print("[jarvis] too many restarts — parking. Restart JARVIS manually.", flush=True)
                    break
                restarts += 1
                backend = start_backend(port, ollama, inference)
                continue
            if hud is not None and hud.poll() is not None:
                print("[jarvis] HUD exited — relaunching.", flush=True)
                hud = start_hud(port)
    except KeyboardInterrupt:
        print("\n[jarvis] stopping…", flush=True)
    finally:
        for p, name in ((hud, "hud"), (backend, "backend")):
            try:
                if p is not None and p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        try:
            time.sleep(1.0)
        except Exception:
            pass
        for p in (hud, backend):
            try:
                if p is not None and p.poll() is None:
                    p.kill()
            except Exception:
                pass
        print("[jarvis] stopped.", flush=True)


if __name__ == "__main__":
    main()
