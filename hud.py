"""JARVIS floating HUD — always-on-top status overlay (stdlib tkinter only).

Panel: state, last action, click coords, STOP button. Polls /api/hud.
Overlay: fullscreen click-through layer with the persistent blue aura around
the focused window + ghost cursor ring where JARVIS is pointing.

  python hud.py [--port 8000] [--no-overlay]
"""
import json
import os
import sys
import tempfile
import time
import urllib.request

PORT = 8000
NO_OVERLAY = False
if "--port" in sys.argv:
    try:
        PORT = int(sys.argv[sys.argv.index("--port") + 1])
    except Exception:
        pass
if "--no-overlay" in sys.argv:
    NO_OVERLAY = True

HUD_PATH = os.path.join(tempfile.gettempdir(), "jarvis_hud.json")


def fetch_hud():
    # Prefer HTTP (works even if tmp file is stale), fall back to file.
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/hud", timeout=2) as r:
            return json.loads(r.read().decode() or "{}")
    except Exception:
        pass
    try:
        with open(HUD_PATH, "r") as f:
            return json.loads(f.read() or "{}")
    except Exception:
        return {"state": "offline", "action": "backend offline?", "t": ""}


def post_kill():
    try:
        data = json.dumps({}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/os_kill", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        pass


def _make_click_through(win):
    """Make a tkinter top-level ignore mouse input. Returns True/False."""
    try:
        import ctypes
        hwnd = int(win.winfo_id())
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20
        WS_EX_NOACTIVATE = 0x08000000
        u32 = ctypes.windll.user32
        ex = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        u32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)
        return True
    except Exception:
        return False


def main():
    import tkinter as tk
    root = tk.Tk()
    root.title("JARVIS")
    root.attributes("-topmost", True)
    root.overrideredirect(False)
    root.geometry("300x190+1580+10")
    root.configure(bg="#020409")
    try:
        root.attributes("-alpha", 0.92)
    except Exception:
        pass

    title = tk.Label(root, text="J.A.R.V.I.S. // HUD", font=("Consolas", 9, "bold"),
                     fg="#00e5ff", bg="#020409")
    title.pack(pady=(8, 2))
    state_var = tk.StringVar(value="starting…")
    state = tk.Label(root, textvariable=state_var, font=("Consolas", 11, "bold"),
                     fg="#00ff9d", bg="#020409")
    state.pack()
    action_var = tk.StringVar(value="")
    action = tk.Label(root, textvariable=action_var, font=("Consolas", 8),
                      fg="#9fdcff", bg="#020409", wraplength=280, justify="left")
    action.pack(pady=(4, 2))
    mark_var = tk.StringVar(value="")
    mark = tk.Label(root, textvariable=mark_var, font=("Consolas", 8),
                    fg="#ffb300", bg="#020409")
    mark.pack()

    btn_row = tk.Frame(root, bg="#020409")
    btn_row.pack(pady=6)

    def on_stop():
        post_kill()
        action_var.set("STOP sent — task killed.")
        try:
            with open(HUD_PATH, "w") as f:
                f.write(json.dumps({"state": "idle", "action": "stopped by HUD",
                                    "t": time.strftime("%H:%M:%S")}))
        except Exception:
            pass

    stop = tk.Button(btn_row, text="⏹ STOP", command=on_stop,
                     fg="#ff3b5c", bg="#1a0206", font=("Consolas", 9, "bold"),
                     relief="groove", padx=10)
    stop.pack(side="left", padx=6)
    hide = tk.Button(btn_row, text="—", command=root.iconify,
                     fg="#6f93ad", bg="#06121e", font=("Consolas", 9),
                     relief="groove", padx=10)
    hide.pack(side="left")
    quit_b = tk.Button(btn_row, text="✕", command=root.destroy,
                       fg="#6f93ad", bg="#06121e", font=("Consolas", 9),
                       relief="groove", padx=10)
    quit_b.pack(side="left", padx=6)

    # Draggable via title bar area.
    _drag = {"x": 0, "y": 0}

    def _start(e):
        _drag["x"], _drag["y"] = e.x, e.y

    def _move(e):
        try:
            root.geometry(f"+{root.winfo_x() + e.x - _drag['x']}+{root.winfo_y() + e.y - _drag['y']}")
        except Exception:
            pass

    title.bind("<Button-1>", _start)
    title.bind("<B1-Motion>", _move)

    # ---- Fullscreen aura + ghost overlay (persistent until next focus) ----
    overlay, canvas, overlay_ok = None, None, False
    if not NO_OVERLAY:
        try:
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            overlay = tk.Toplevel(root)
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.geometry(f"{sw}x{sh}+0+0")
            try:
                overlay.attributes("-transparentcolor", "black")
            except Exception:
                pass
            overlay.configure(bg="black")
            canvas = tk.Canvas(overlay, width=sw, height=sh, bg="black",
                               highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)
            overlay_ok = _make_click_through(overlay)
            if not overlay_ok:
                # Never break clicks: no click-through == no overlay.
                try:
                    overlay.withdraw()
                except Exception:
                    pass
                overlay, canvas = None, None
            else:
                try:
                    overlay.withdraw()
                except Exception:
                    pass
        except Exception:
            overlay, canvas, overlay_ok = None, None, False

    def draw_overlay(h):
        if not overlay_ok or canvas is None or overlay is None:
            return
        try:
            canvas.delete("all")
            drew = False
            # Persistent blue aura around the focused window (no expiry).
            try:
                fx, fy, fw, fh = h.get("fx"), h.get("fy"), h.get("fw"), h.get("fh")
                if fx is not None and fy is not None and fw and fh:
                    fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
                    # Outer glow layers + bright core.
                    canvas.create_rectangle(fx - 8, fy - 8, fx + fw + 8, fy + fh + 8,
                                            outline="#003a4a", width=6)
                    canvas.create_rectangle(fx - 4, fy - 4, fx + fw + 4, fy + fh + 4,
                                            outline="#0090b3", width=3)
                    canvas.create_rectangle(fx, fy, fx + fw, fy + fh,
                                            outline="#00e5ff", width=3)
                    # Corner ticks.
                    L = 26
                    for cx, cy, dx, dy in ((fx, fy, 1, 1), (fx + fw, fy, -1, 1),
                                           (fx, fy + fh, 1, -1), (fx + fw, fy + fh, -1, -1)):
                        canvas.create_line(cx, cy, cx + dx * L, cy, fill="#8ff5ff", width=4)
                        canvas.create_line(cx, cy, cx, cy + dy * L, fill="#8ff5ff", width=4)
                    drew = True
            except Exception:
                pass
            # Ghost cursor: JARVIS pointer, bright when fresh, dim persistent.
            try:
                gx, gy, mt = h.get("x"), h.get("y"), h.get("mark_t") or 0
                if gx is not None and gy is not None:
                    gx, gy = int(gx), int(gy)
                    try:
                        fresh = (time.time() - float(mt)) < 4 if mt else False
                    except Exception:
                        fresh = False
                    col = "#00e5ff" if fresh else "#005a70"
                    r = 16
                    canvas.create_oval(gx - r, gy - r, gx + r, gy + r,
                                       outline=col, width=3)
                    canvas.create_oval(gx - 3, gy - 3, gx + 3, gy + 3,
                                       fill=col, outline=col)
                    canvas.create_text(gx + 22, gy - 14, text="J",
                                       fill=col, font=("Consolas", 12, "bold"))
                    drew = True
            except Exception:
                pass
            try:
                if drew:
                    overlay.deiconify()
                    try:
                        overlay.attributes("-topmost", True)
                    except Exception:
                        pass
                else:
                    overlay.withdraw()
            except Exception:
                pass
        except Exception:
            pass

    def tick():
        try:
            h = fetch_hud() or {}
            st = str(h.get("state") or "idle").upper()
            act = str(h.get("action") or "")[:90]
            t = str(h.get("t") or "")
            state_var.set(f"{st} · {t}" if t else st)
            action_var.set(act or "Say HEY JARVIS — I am watching.")
            colors = {"WORKING": "#ffb300", "IDLE": "#00ff9d",
                      "OFFLINE": "#ff3b5c", "SPEAKING": "#00e5ff"}
            state.configure(fg=colors.get(st, "#00e5ff"))
            x, y, mt = h.get("x"), h.get("y"), h.get("mark_t") or 0
            try:
                fresh = (time.time() - float(mt)) < 1.5 if mt else False
            except Exception:
                fresh = False
            if fresh and x is not None:
                mark_var.set(f"◉ click @ {x},{y}")
                try:
                    root.configure(highlightbackground="#00e5ff",
                                   highlightthickness=2)
                except Exception:
                    pass
            else:
                mark_var.set("")
                try:
                    root.configure(highlightthickness=0)
                except Exception:
                    pass
            draw_overlay(h)
        except Exception:
            state_var.set("HUD ERROR")
        root.after(500, tick)

    root.after(400, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
