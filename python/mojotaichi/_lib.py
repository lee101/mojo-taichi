"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOTAICHI_LIB") or os.path.join(ROOT, "dist", "libmojo-taichi.so")

I = ctypes.c_int64
F = ctypes.c_float

# name -> (argtypes, restype)
_SIGNATURES = {
    "mt_simd_width": ([], ctypes.c_int64),
    "mt_taichi_logo": ([I, I, I, F], None),
    "mt_sf_advect_v2": ([I, I, I, I, F, F, I], None),
    "mt_sf_advect_v3": ([I, I, I, I, F, F, I], None),
    "mt_sf_apply_impulse": ([I, I, I, I, F, F, F, I], None),
    "mt_sf_divergence": ([I, I, I, I], None),
    "mt_sf_vorticity": ([I, I, I, I], None),
    "mt_sf_pressure_jacobi": ([I, I, I, I, I], None),
    "mt_sf_subtract_gradient": ([I, I, I, I], None),
    "mt_sf_enhance_vorticity": ([I, I, I, F, F, I], None),
    "mt_sf_copy_divergence": ([I, I, I, I], None),
    "mt_sf_apply_pressure": ([I, I, I, I], None),
    "mt_ef_init_field": ([I, I, I, I, I], None),
    "mt_ef_advection_v2": ([I, I, I, I, I, F, I], None),
    "mt_ef_advection_v3": ([I, I, I, I, I, F, I], None),
    "mt_ef_curl": ([I, I, I, I, I], None),
    "mt_ef_vorticity_projection": ([I, I, I, I, I, F, F, I], None),
    "mt_ef_divergence": ([I, I, I, I, I], None),
    "mt_ef_pressure_iteration": ([I, I, I, I, I, I], None),
    "mt_ef_pressure_projection": ([I, I, I, I, I, I], None),
    "mt_ef_apply_vel_bc": ([I, I, I, I], None),
    "mt_ef_apply_p_bc": ([I, I, I, I], None),
    "mt_ef_apply_c_bc": ([I, I, I, I], None),
    "mt_ef_mouse_addspeed": ([I, I, I, I, F, I, I, I, I, F, F, I], None),
    "mt_mpm88_substep": ([I, I, I, I, I, I, I, I] + [F] * 6 + [I, I], None),
    "mt_mpm88_init": ([I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOTAICHI_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOTAICHI_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-taichi.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f32(a, name: str = "array", shape=None) -> np.ndarray:
    """Validate that `a` can cross the FFI as a float32 buffer, and return it.

    Every ported kernel is float32 and row-major, exactly like the upstream
    `ti.field(float, ...)` it mirrors, so no precision is added or lost at the
    boundary -- and a buffer that is *not* already float32 and C-contiguous
    cannot be handed over at all.  Silently converting it would be worse than an
    error: these kernels write in place, so a converted copy would receive the
    result and the caller's array would be left untouched, with no indication
    that anything happened.

    `shape`, if given, is checked against `a.shape` element for element; a
    trailing `1` in the kernel's expected shape matches any extent, which is
    how a `vec3` dye field is accepted where a scalar field is also legal.
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(a).__name__}")
    if a.dtype != np.float32:
        raise TypeError(
            f"{name} must be float32, got {a.dtype}; the kernels are float32 "
            "and would otherwise write into a converted copy"
        )
    if not a.flags.c_contiguous:
        raise TypeError(
            f"{name} must be C-contiguous, got strides {a.strides}; the kernels "
            "address it as a flat row-major buffer"
        )
    if shape is not None and tuple(a.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {a.shape}")
    return a


def addr(a: np.ndarray) -> int:
    return a.ctypes.data

def nworkers(requested: int | None = None) -> int:
    """Worker threads a kernel should use when none is passed to it.

    The number is passed down to every kernel, which uses it to choose between a
    plain loop and `max.algorithm.parallelize`.  On the pinned toolchain no
    covered kernel clears the launch cost (see `PAR_MIN` in
    `src/taichi/vector.mojo`), so this value currently selects the plain loop in
    every case; it is still read, defaulted and forwarded so that lowering
    `PAR_MIN` to a reachable size is a one-number change rather than a rework of
    the bindings.

    Defaults to `min(os.cpu_count(), 32)`, which is where the worker-scaling
    measurement turned over.  Set `$MOJOTAICHI_THREADS` to override, or pass
    `nworkers=` per call.
    """
    if requested is not None:
        return int(requested)
    env = os.environ.get("MOJOTAICHI_THREADS")
    if env:
        return int(env)
    return min(os.cpu_count() or 1, 32)


def guarded(shape, comps: int = 1) -> np.ndarray:
    """A (n0, n1[, comps]) float32 field with a two-cell guard on every side.

    Several upstream kernels index `i + 1` / `i - 1` on the last and first
    row/column without clamping.  Upstream that is only defined on a GPU (the
    CPU backend raises).  The port keeps the arithmetic, so the field the
    kernels see is allocated with slack around it; the returned view is the
    logical (n0, n1[, comps]) field and the guard is never part of it.
    """
    n0, n1 = shape
    stride = n1 + 4
    buf = np.zeros((n0 + 4) * stride * comps, dtype=np.float32)
    off = (2 * stride + 2) * comps
    view = buf[off:off + n0 * n1 * comps]
    return view.reshape((n0, n1) if comps == 1 else (n0, n1, comps))


def main() -> int:
    """`python -m mojotaichi._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
