"""Public op surface registered with `torch.library`."""

from __future__ import annotations

from flash_attn_lab.ops.attention import attention

__all__ = ["attention"]
