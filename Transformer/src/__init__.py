# src/__init__.py

"""
폴더에 `__init__.py` 가 있으면 그 폴더를 패키지로 인식
"""

from .attention import ScaledDotProductAttention

__all__ = [
    "ScaledDotProductAttention",
    "MultiHeadAttention"
]
__version__ = "0.0.1"