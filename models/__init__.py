"""
models package initializer

Provides unified access to all VC‑MOJI components.

Usage:
    from models import VCMOJI, NJXAttention, VariationalLatent
"""

from .vcmoji import VCMOJI
from .njx_attention import NJXAttention
from .variational import VariationalLatent
from .quantizer import CreniqQuantizer
from .eck_lock import ECKLock

__all__ = [
    "VCMOJI",
    "NJXAttention",
    "VariationalLatent",
    "CreniqQuantizer",
    "ECKLock",
]
