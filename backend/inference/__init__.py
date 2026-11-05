"""Unified inference abstraction layer (BUILD 18).

Architecture::

    server.py
        |
    ModelRouter
        +-- LlamaCppBackend (ModelManager cache, load-once)

llama.cpp is the only backend. All model calls from business logic go
through ``ModelRouter``; business logic never imports
``llamacpp_backend`` directly and never touches ``urllib`` itself.
"""
from .inference_backend import InferenceBackend, InferenceError
from .model_router import ModelRouter, ToolBrain, get_router, reset_router

__all__ = [
    "InferenceBackend",
    "InferenceError",
    "ModelRouter",
    "ToolBrain",
    "get_router",
    "reset_router",
]
