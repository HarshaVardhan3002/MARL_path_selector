"""Device selection with verified fallback.

Per PHASE_2_SPEC §1.1: try ROCm/CUDA, fall back to CPU on any failure.
Failure modes covered: device init, simple tensor op, gradient computation.
The chosen device + a brief diagnostic record are returned for inclusion
in the per-run manifest.
"""
from __future__ import annotations

import dataclasses
import logging
import platform
import traceback

import torch

logger = logging.getLogger("scion_marl.device")


@dataclasses.dataclass
class DeviceInfo:
    """What the trainer actually got, plus why."""

    device: torch.device
    backend: str  # "cuda", "rocm", "cpu"
    reason: str
    torch_version: str
    cuda_version: str | None
    hip_version: str | None
    num_cpu_threads: int
    platform: str


def _smoke_test_device(device: torch.device) -> tuple[bool, str]:
    """Verify init + matmul + autograd. Returns (ok, error_str)."""
    try:
        x = torch.randn(8, 8, device=device, requires_grad=True)
        y = (x @ x.T).sum()
        y.backward()
        if x.grad is None:
            return False, "autograd produced no grad"
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=2)


def select_device(prefer: str = "auto") -> DeviceInfo:
    """Pick the best available device with a verified backend.

    `prefer` may be 'auto', 'cuda', 'cpu'. 'auto' tries cuda/rocm first,
    falls back to CPU on any failure. Always returns a working device.
    """
    cuda_version = getattr(torch.version, "cuda", None)
    hip_version = getattr(torch.version, "hip", None)
    plat = f"{platform.system()} {platform.release()} ({platform.machine()})"

    if prefer == "cpu":
        return DeviceInfo(
            device=torch.device("cpu"),
            backend="cpu",
            reason="explicitly requested",
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            hip_version=hip_version,
            num_cpu_threads=torch.get_num_threads(),
            platform=plat,
        )

    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        dev = torch.device("cuda")
        ok, err = _smoke_test_device(dev)
        if ok:
            backend = "rocm" if hip_version else "cuda"
            logger.info("device: %s (%s) smoke-test passed", backend, dev)
            return DeviceInfo(
                device=dev,
                backend=backend,
                reason="cuda/rocm available + smoke-test passed",
                torch_version=torch.__version__,
                cuda_version=cuda_version,
                hip_version=hip_version,
                num_cpu_threads=torch.get_num_threads(),
                platform=plat,
            )
        logger.warning("cuda/rocm available but smoke-test failed; falling back to CPU. err=%s", err.strip()[:200])
        return DeviceInfo(
            device=torch.device("cpu"),
            backend="cpu",
            reason=f"cuda/rocm smoke-test failed: {err.strip()[:200]}",
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            hip_version=hip_version,
            num_cpu_threads=torch.get_num_threads(),
            platform=plat,
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        reason="no cuda/rocm advertised by torch",
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        hip_version=hip_version,
        num_cpu_threads=torch.get_num_threads(),
        platform=plat,
    )
