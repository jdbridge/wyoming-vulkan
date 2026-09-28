"""Checks on what the process really gets from the GPU stack: the GPU path must never fail silently."""

import json
import logging
import os
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

_ICD_VARS = ("VK_DRIVER_FILES", "VK_ICD_FILENAMES")


def vulkan_icd_problem() -> str | None:
    """Return why the Vulkan driver restriction is not safe, or None if it is.

    All engines reach the GPU through the Vulkan loader. Without a restriction to one driver the loader
    also offers llvmpipe (software Vulkan on the CPU), and Dawn (ONNX Runtime WebGPU) may pick it. With a wrong
    path there are no drivers at all and ONNX Runtime quietly runs everything on the CPU.
    """
    values = {v: os.environ.get(v) for v in _ICD_VARS}
    paths = {p for p in values.values() if p}
    if not paths:
        return f"neither {' nor '.join(_ICD_VARS)} is set; the Vulkan loader would also offer llvmpipe (CPU)"
    for path in paths:
        for part in path.split(os.pathsep):
            p = Path(part)
            if not p.is_file():
                return f"Vulkan driver file {part} does not exist (variables: {values})"
            try:
                library = json.loads(p.read_text())["ICD"]["library_path"]
            except (OSError, ValueError, KeyError) as err:
                return f"Vulkan driver file {part} is unreadable: {err}"
            if "lvp" in library or "swrast" in library:
                return f"Vulkan driver file {part} is a software renderer ({library})"
    return None


def vulkan_gpu_name(name_contains: str) -> str | None:
    """Name of the Vulkan GPU matching `name_contains`, e.g. "Intel(R) Graphics (ADL-N)".

    Read through ggml's device list (libggml is in the image), which uses the same Vulkan loader and driver
    restriction as Dawn, so it names the device ONNX Runtime's WebGPU EP runs on. None if unavailable.
    """
    try:
        from .engines import ggml

        for d in ggml.list_devices():
            if d.type in ("gpu", "igpu") and name_contains in d.description:
                return d.description
    except OSError as err:
        _LOGGER.debug("No ggml for the Vulkan device name: %s", err)
    return None


def log_environment() -> None:
    _LOGGER.info("Vulkan driver restriction: %s", {v: os.environ.get(v) for v in _ICD_VARS})
    render = sorted(str(p) for p in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    _LOGGER.info("Render nodes in the container: %s", render or "none")
