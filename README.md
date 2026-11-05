# J.A.R.V.I.S. Memory Galaxy

An interactive 3D memory visualization sphere and autonomous desktop AI assistant.

## Features
- **3D Memory Galaxy:** Three.js sphere of glowing memory stars representing long-term vector memories.
- **Hands-Free Voice:** Local wake-word voice activation ("Hey JARVIS").
- **Desktop Autopilot:** Windows screen control and UI automation via UI Automation, OCR, and browser CDP.
- **Local Inference:** Connects to local Ollama and `llama.cpp` backends.

## Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome/Edge with WebGL support)
- Local [Ollama](https://ollama.com/) instance

## Launching
```bash
python jarvis.py
```
This starts the backend server, floating status HUD, and web visualizer.
