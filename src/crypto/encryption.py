"""Encryption helpers – re-exports DIBLECore for backward-compat."""
from .dible import DIBLECore, DIBLEKey  # noqa: F401

# Alias for any legacy code
DIBLEEncryption = DIBLECore
