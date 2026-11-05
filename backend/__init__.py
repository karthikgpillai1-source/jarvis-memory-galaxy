"""JARVIS backend package (BUILD 15+).

Additive inference layer — `backend.inference` owns all model calls.
Nothing here is imported by legacy code paths unless available;
`server.py` falls back to its direct Ollama helpers if this package
fails to import, so migration stays reversible.
"""
