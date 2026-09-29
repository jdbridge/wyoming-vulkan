"""Checks on what the process really gets from the GPU stack: the GPU path must never fail silently."""

import json
import logging
import os
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_ICD_VARS = ("VK_DRIVER_FILES", "VK_ICD_FILENAMES")


def vulkan_icd_problem() -> str | None:
    """Return why the Vulkan driver restriction is not safe, or None if it is.

    All engines reach the GPU through the Vulkan loader. Without a restriction to real GPU drivers the loader
    also offers llvmpipe (software Vulkan on the CPU), and Dawn (ONNX Runtime WebGPU) may pick it. With a wrong
    path there are no drivers at all and ONNX Runtime quietly runs everything on the CPU.
    """
    values = {v: os.environ.get(v) for v in _ICD_VARS}
    paths = {p for p in values.values() if p}
    if not paths:
        return f"neither {' nor '.join(_ICD_VARS)} is set; the Vulkan loader would also offer llvmpipe (CPU)"
    for path in paths:
        parts = [part for part in path.split(os.pathsep) if part]
        if not any(Path(part).is_file() for part in parts):
            return f"no Vulkan driver file of {parts} exists (variables: {values})"
        for part in parts:
            p = Path(part)
            if not p.is_file():
                continue  # e.g. NVIDIA's without the NVIDIA runtime: the loader skips it
            try:
                library = json.loads(p.read_text())["ICD"]["library_path"]
            except (OSError, ValueError, KeyError) as err:
                return f"Vulkan driver file {part} is unreadable: {err}"
            if "lvp" in library or "swrast" in library:
                return f"Vulkan driver file {part} is a software renderer ({library})"
    return None


def prune_vulkan_drivers() -> list[str]:
    """Drop driver files that do not exist in this container from VK_DRIVER_FILES / VK_ICD_FILENAMES, before anything
    initialises Vulkan. The image lists Intel's and NVIDIA's; NVIDIA's is only there when the container runs with the
    NVIDIA runtime. If none exists, the variables stay as they are (vulkan_icd_problem then reports it). Returns
    the dropped files (logged by the caller)."""
    dropped: list[str] = []
    for var in _ICD_VARS:
        value = os.environ.get(var)
        if not value:
            continue
        parts = [p for p in value.split(os.pathsep) if p]
        present = [p for p in parts if Path(p).is_file()]
        if present and len(present) < len(parts):
            os.environ[var] = os.pathsep.join(present)
            dropped += [p for p in parts if p not in present and p not in dropped]
    return dropped


_GPUS: list | None = None


def gpu_devices() -> list:
    """The GPUs ggml sees in this process (ggml.GgmlDevice, type "gpu" = discrete or "igpu"), read once. They come
    through the same Vulkan loader and driver list as Dawn (ONNX Runtime WebGPU) and cosyvoice-server. [] without ggml."""
    global _GPUS
    if _GPUS is None:
        try:
            from .engines import ggml

            _GPUS = [d for d in ggml.list_devices() if d.type in ("gpu", "igpu")]
        except OSError as err:
            _LOGGER.debug("No ggml for the Vulkan device list: %s", err)
            return []
    return _GPUS


def vulkan_gpu_name(name_contains: str) -> str | None:
    """Name of the Vulkan GPU matching `name_contains`, e.g. "Intel(R) Graphics (ADL-N)"; None if there is none."""
    return next((d.description for d in gpu_devices() if name_contains in d.description), None)


def log_environment() -> None:
    _LOGGER.info("Vulkan driver restriction: %s", {v: os.environ.get(v) for v in _ICD_VARS})
    _LOGGER.info("Vulkan GPUs: %s", ", ".join(f"{d.name} = {d.description} ({'integrated' if d.type == 'igpu' else 'discrete'})"
                                                for d in gpu_devices()) or "none")
    render = sorted(str(p) for p in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    _LOGGER.info("Render nodes in the container: %s", render or "none")


def rss_mb() -> float:
    """Resident memory of this process in MiB (an iGPU's buffers live in system RAM, so they partly show here)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def cgroup_memory_mb() -> float | None:
    """The container's memory use (cgroup v2), including page cache; None outside a container."""
    try:
        return int(open("/sys/fs/cgroup/memory.current").read()) / 1024 / 1024
    except (OSError, ValueError):
        return None


def memory_mb() -> float:
    """What a model load costs: the container's memory (cgroup v2: includes GPU buffers of an iGPU and the page cache
    of freshly read model files) if available, else this process's RSS."""
    cg = cgroup_memory_mb()
    return cg if cg is not None else rss_mb()
