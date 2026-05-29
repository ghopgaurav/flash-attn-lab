"""Device introspection helpers.

These helpers exist so kernels and benchmarks can branch on hardware
capability (e.g. FP8 only on SM89+) without duplicating CUDA-version logic.

All functions are safe to call on CPU-only machines: they return sensible
defaults and never raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Summary of the active CUDA device (or a CPU fallback)."""

    name: str
    is_cuda: bool
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    multi_processor_count: int
    supports_bf16: bool
    supports_fp16: bool
    supports_fp8: bool
    supports_tf32: bool
    cuda_runtime: Optional[str] = None
    torch_version: str = field(default_factory=lambda: torch.__version__)

    @property
    def sm(self) -> int:
        """Compute capability as a single integer (e.g. SM80 -> 80)."""
        return self.sm_major * 10 + self.sm_minor

    @property
    def sm_str(self) -> str:
        """Compute capability as `SMxx` string."""
        return f"SM{self.sm_major}{self.sm_minor}"


def _cuda_runtime_version() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        v = torch.version.cuda  # type: ignore[attr-defined]
        return str(v) if v is not None else None
    except Exception:  # pragma: no cover - defensive
        return None


def get_device_info(device: Optional[torch.device | int | str] = None) -> DeviceInfo:
    """Return a `DeviceInfo` for the given (or current) device.

    On CPU-only machines, returns a CPU-shaped record with `is_cuda=False`
    and zero capabilities. Never raises.
    """
    if not torch.cuda.is_available():
        return DeviceInfo(
            name="cpu",
            is_cuda=False,
            sm_major=0,
            sm_minor=0,
            total_memory_bytes=0,
            multi_processor_count=0,
            supports_bf16=False,
            supports_fp16=False,
            supports_fp8=False,
            supports_tf32=False,
            cuda_runtime=None,
        )

    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        idx = torch.device(device).index or 0
    else:
        idx = int(device)

    props = torch.cuda.get_device_properties(idx)
    sm_major = int(props.major)
    sm_minor = int(props.minor)
    sm = sm_major * 10 + sm_minor

    # Capability tables. Sources: NVIDIA CUDA C Programming Guide,
    # PTX ISA tables, and torch.cuda.is_bf16_supported / get_device_capability.
    supports_fp16 = sm >= 53  # FP16 since Maxwell GP100/Pascal; ubiquitous
    # bf16 native compute requires SM80+ (Ampere); torch reports this best.
    try:
        supports_bf16 = bool(torch.cuda.is_bf16_supported()) and sm >= 80
    except Exception:  # pragma: no cover
        supports_bf16 = sm >= 80
    supports_fp8 = sm >= 89  # Ada SM89 (e4m3/e5m2) and Hopper SM90+
    supports_tf32 = sm >= 80  # TF32 tensor-core path on Ampere+

    return DeviceInfo(
        name=props.name,
        is_cuda=True,
        sm_major=sm_major,
        sm_minor=sm_minor,
        total_memory_bytes=int(props.total_memory),
        multi_processor_count=int(props.multi_processor_count),
        supports_bf16=supports_bf16,
        supports_fp16=supports_fp16,
        supports_fp8=supports_fp8,
        supports_tf32=supports_tf32,
        cuda_runtime=_cuda_runtime_version(),
    )


# Preference order is descending precision/safety. We never silently pick a
# dtype the device doesn't support; we walk the preference list and return
# the first supported one, falling back to fp32.
_DTYPE_FALLBACK_ORDER = {
    "fp8_e4m3": ["fp8_e4m3", "bf16", "fp16", "fp32"],
    "fp8_e5m2": ["fp8_e5m2", "bf16", "fp16", "fp32"],
    "bf16": ["bf16", "fp16", "fp32"],
    "fp16": ["fp16", "bf16", "fp32"],
    "fp32": ["fp32"],
}


def _torch_dtype(name: str) -> Optional[torch.dtype]:
    table = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if name in table:
        return table[name]
    # FP8 dtypes only exist on torch builds with FP8 support.
    if name == "fp8_e4m3":
        return getattr(torch, "float8_e4m3fn", None)
    if name == "fp8_e5m2":
        return getattr(torch, "float8_e5m2", None)
    return None


def _device_supports(name: str, info: DeviceInfo) -> bool:
    if name == "fp32":
        return True
    # On CPU we deliberately fall back to fp32 even though PyTorch can
    # represent fp16/bf16 — this repo's kernels only run on GPU, and a
    # CPU caller asking for "bf16" is best served fp32 inputs.
    if not info.is_cuda:
        return False
    if name == "fp16":
        return info.supports_fp16
    if name == "bf16":
        return info.supports_bf16
    if name in ("fp8_e4m3", "fp8_e5m2"):
        return info.supports_fp8 and _torch_dtype(name) is not None
    return False


def select_dtype(
    prefer: str = "bf16",
    info: Optional[DeviceInfo] = None,
) -> torch.dtype:
    """Return the best supported torch dtype for `prefer`, falling back as needed.

    Args:
        prefer: One of `"fp32"`, `"fp16"`, `"bf16"`, `"fp8_e4m3"`, `"fp8_e5m2"`.
        info:   Optional pre-computed `DeviceInfo`. Defaults to current device.

    Returns:
        A `torch.dtype`. Always returns something usable; falls back to fp32.
    """
    if info is None:
        info = get_device_info()
    order = _DTYPE_FALLBACK_ORDER.get(prefer, _DTYPE_FALLBACK_ORDER["bf16"])
    for name in order:
        dt = _torch_dtype(name)
        if dt is not None and _device_supports(name, info):
            if name != prefer:
                logger.info(
                    "select_dtype: requested %s, falling back to %s on %s",
                    prefer,
                    name,
                    info.name,
                )
            return dt
    return torch.float32
