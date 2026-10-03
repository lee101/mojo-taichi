"""`taichi/examples/patterns.py`: the taichi logo used by `eulerfluid2d.init_field`."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f32, lib


def taichi_logo(pos, scale: float = 1 / 1.11):
    """`taichi_logo(Vector([x, y]), scale=1/1.11) -> float`.

    `pos` is a length-2 sequence or an (n, 2) array; the return type matches
    the input.  Values are 0.0 or 1.0.
    """
    p = np.ascontiguousarray(pos, dtype=np.float32)
    scalar = p.ndim == 1
    if scalar:
        # the kernel reads two floats per point, so a lone scalar is not a point
        if p.size != 2:
            raise ValueError(
                f"a single taichi_logo position must have 2 components, got {p.size}"
            )
        p = p.reshape(1, 2)
    elif p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"pos must be (n, 2) or a length-2 sequence, got {p.shape}")
    dst = np.zeros(p.shape[0], dtype=np.float32)
    lib().mt_taichi_logo(addr(f32(p, "pos")), addr(dst), p.shape[0], float(scale))
    return float(dst[0]) if scalar else dst
