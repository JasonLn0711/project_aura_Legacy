import glob
import os
import site
import sys
from ctypes import CDLL, RTLD_GLOBAL


CUDA_RUNTIME_GLOBS = (
    "nvidia/cuda_runtime/lib/libcudart.so*",
    "nvidia/cublas/lib/libcublas.so*",
    "nvidia/cublas/lib/libcublasLt.so*",
    "nvidia/cudnn/lib/libcudnn*.so*",
)
CUDA_REQUIRED_LIBS = ("libcublas.so.12", "libcublasLt.so.12")


def _candidate_site_packages():
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []

    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass

    candidates.extend(
        [
            os.path.join(sys.prefix, "lib", py_ver, "site-packages"),
            os.path.join(os.path.dirname(sys.executable), "..", "lib", py_ver, "site-packages"),
            os.path.join(os.path.dirname(__file__), "../../../.record", "lib", py_ver, "site-packages"),
        ]
    )

    normalized = []
    seen = set()
    for path in candidates:
        real_path = os.path.realpath(path)
        if real_path not in seen and os.path.isdir(real_path):
            seen.add(real_path)
            normalized.append(real_path)
    return normalized


def preload_cuda_runtime_libraries():
    cached = getattr(preload_cuda_runtime_libraries, "_cache", None)
    if cached is not None:
        return cached

    try:
        for lib_name in CUDA_REQUIRED_LIBS:
            CDLL(lib_name, mode=RTLD_GLOBAL)
        result = (True, "system")
        preload_cuda_runtime_libraries._cache = result
        return result
    except OSError:
        pass

    seen = set()
    for base in _candidate_site_packages():
        for pattern in CUDA_RUNTIME_GLOBS:
            for path in sorted(glob.glob(os.path.join(base, pattern))):
                real_path = os.path.realpath(path)
                if real_path in seen:
                    continue
                seen.add(real_path)
                try:
                    CDLL(real_path, mode=RTLD_GLOBAL)
                except OSError:
                    continue

    try:
        for lib_name in CUDA_REQUIRED_LIBS:
            CDLL(lib_name, mode=RTLD_GLOBAL)
        result = (True, "bundled")
    except OSError as exc:
        result = (False, str(exc))

    preload_cuda_runtime_libraries._cache = result
    return result


def is_cuda_runtime_error(error_msg):
    lowered = str(error_msg).lower()
    needles = (
        "libcublas.so",
        "libcublaslt.so",
        "libcudnn",
        "cannot be loaded",
        "cannot open shared object file",
    )
    return any(needle in lowered for needle in needles)
