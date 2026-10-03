"""mojo-taichi — Taichi's spatial simulation kernels, implemented in Mojo.

Ported from the `taichi` package (v1.7.4): the stable-fluids solver
(`taichi/examples/simulation/stable_fluid.py`), the 2D Euler fluid solver
(`taichi/examples/simulation/eulerfluid2d.py`) with the taichi logo pattern it
paints with, and the MPM88 material point solver
(`taichi/examples/simulation/mpm88.py`).

Kernels keep upstream's names and argument order.  What changes is the calling
convention: a `ti.field` becomes the C-contiguous float32 numpy array
`field.to_numpy()` returns, and upstream's module-level constants become
keyword arguments carrying upstream's defaults.

    import numpy as np
    import mojotaichi

    sim = mojotaichi.StableFluid(res=256, p_jacobi_iters=50)
    sim.step(mojotaichi.impulse(mouse_dir=(1.0, 0.0), mouse_pos=(128.0, 128.0),
                                color=(1.0, 0.4, 0.1)))
"""

from . import eulerfluid2d, mpm88, patterns, stable_fluid
from ._lib import build, guarded as guarded_field
from .eulerfluid2d import EulerFluid
from .mpm88 import MPM88
from .patterns import taichi_logo
from .stable_fluid import StableFluid, impulse

__version__ = "0.1.0"
__all__ = [
    "StableFluid", "EulerFluid", "MPM88", "impulse", "taichi_logo",
    "guarded_field", "build",
    "eulerfluid2d", "mpm88", "patterns", "stable_fluid",
]
