"""The Python half of `taichi/examples/simulation/eulerfluid2d.py`.

Same names, same argument order.  Upstream's `eulerSimParam` dict becomes
keyword arguments with upstream's defaults, and `shape` becomes `n0`, `n1`.

Several upstream kernels index `i + 1` / `i - 1` on the first and last
row/column without clamping.  Upstream that is only defined on a GPU; the CPU
backend raises.  The port keeps the arithmetic, so fields allocated by
`EulerFluid` (and anything built with `guarded_field`) carry a two-cell guard
around the logical field, and those accesses land in the guard instead of in
the next allocation.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f32, lib
from ._lib import guarded as guarded_field
from ._lib import nworkers as _default_workers

eulerSimParam = {  # noqa: N816 — upstream spells it this way
    "shape": [512, 512],
    "dt": 1 / 60.0,
    "iteration_step": 20,
    "mouse_radius": 0.01,  # [0.0,1.0] float
    "mouse_speed": 125.0,
    "mouse_respondDistance": 0.5,
    "curl_param": 15,
}


def _nw(requested):
    return _default_workers(requested)


def _f(a, comps: int, name: str, shape=None):
    """Validate one field and return it: float32, C-contiguous, `comps` wide."""
    f32(a, name)
    if a.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D or 2D+vector field, got shape {a.shape}")
    n0, n1 = (shape if shape is not None else a.shape)[:2]
    want = (n0, n1) if comps == 1 else (n0, n1, comps)
    if a.shape != want:
        raise ValueError(f"{name} must have shape {want}, got {a.shape}")
    return a


def _comps(a, name: str) -> int:
    """How many components `a` carries, rejecting anything that is not a field."""
    f32(a, name)
    if a.ndim == 2:
        return 1
    if a.ndim == 3:
        return a.shape[2]
    raise ValueError(f"{name} must be a 2D or 2D+vector field, got shape {a.shape}")


def _pair(vf, qf, new_qf):
    """All three fields share one grid, and `new_qf` mirrors `qf`."""
    vf = _f(vf, 2, "vf")
    c = _comps(qf, "qf")
    qf = _f(qf, c, "qf", vf.shape)
    new_qf = _f(new_qf, c, "new_qf", vf.shape)
    return vf, qf, new_qf


def init_field(pressField, velocityField, colorField):  # noqa: N803
    """Zero the pressure and velocity fields and paint the taichi logo into colour.

    Upstream's `init_field()` takes no arguments and reads `eulerSimParam`;
    here the fields carry their own shape.
    """
    pressField = _f(pressField, 1, "pressField")
    shape = pressField.shape[:2]
    velocityField = _f(velocityField, 2, "velocityField", shape)
    colorField = _f(colorField, 3, "colorField", shape)
    lib().mt_ef_init_field(
        addr(pressField), addr(velocityField), addr(colorField),
        shape[0], shape[1],
    )


def advection(vf, qf, new_qf, *, dt: float = eulerSimParam["dt"], nworkers=None):
    comps = _comps(qf, "qf")
    if comps not in (2, 3):
        raise ValueError(f"advection is ported for vec2 and vec3 fields, got {qf.shape}")
    vf, qf, new_qf = _pair(vf, qf, new_qf)
    n0, n1 = vf.shape[:2]
    fn = lib().mt_ef_advection_v2 if comps == 2 else lib().mt_ef_advection_v3
    fn(addr(vf), addr(qf), addr(new_qf), n0, n1, dt, _nw(nworkers))


def curl(vf, cf, *, nworkers=None):
    vf = _f(vf, 2, "vf")
    cf = _f(cf, 1, "cf", vf.shape[:2])
    lib().mt_ef_curl(addr(vf), addr(cf), vf.shape[0], vf.shape[1], _nw(nworkers))


def vorticity_projection(
    cf,
    vf,
    vf_new,
    *,
    dt: float = eulerSimParam["dt"],
    curl_param: float = eulerSimParam["curl_param"],
    nworkers=None,
):
    vf = _f(vf, 2, "vf")
    cf = _f(cf, 1, "cf", vf.shape[:2])
    vf_new = _f(vf_new, 2, "vf_new", vf.shape)
    lib().mt_ef_vorticity_projection(
        addr(cf), addr(vf), addr(vf_new),
        cf.shape[0], cf.shape[1], dt, curl_param, _nw(nworkers),
    )


def divergence(vf, divf, *, nworkers=None):
    vf = _f(vf, 2, "vf")
    divf = _f(divf, 1, "divf", vf.shape[:2])
    lib().mt_ef_divergence(addr(vf), addr(divf), vf.shape[0], vf.shape[1], _nw(nworkers))


def pressure_iteration(divf, pf, new_pf, *, nworkers=None):
    divf = _f(divf, 1, "divf")
    pf = _f(pf, 1, "pf", divf.shape)
    new_pf = _f(new_pf, 1, "new_pf", divf.shape)
    lib().mt_ef_pressure_iteration(
        addr(divf), addr(pf), addr(new_pf), divf.shape[0], divf.shape[1], _nw(nworkers),
    )


def pressure_solve(presspair, divf, *, iteration_step: int = eulerSimParam["iteration_step"],
                   nworkers=None):
    for _ in range(iteration_step):
        pressure_iteration(divf, presspair.cur, presspair.nxt, nworkers=nworkers)
        presspair.swap()
        apply_p_bc(presspair.cur, nworkers=nworkers)


def pressure_projection(pf, vf, vf_new, *, nworkers=None):
    vf = _f(vf, 2, "vf")
    vf_new = _f(vf_new, 2, "vf_new", vf.shape)
    pf = _f(pf, 1, "pf", vf.shape[:2])
    lib().mt_ef_pressure_projection(
        addr(pf), addr(vf), addr(vf_new), vf.shape[0], vf.shape[1], _nw(nworkers),
    )



def vel_with_boundary(vf, i, j):  # noqa: N802
    n0, n1 = vf.shape[0], vf.shape[1]
    if (i <= 0) or (i >= n0 - 1) or (j >= n1 - 1) or (j <= 0):
        vf[i, j, 0] = 0.0
        vf[i, j, 1] = 0.0
    return vf[i, j, 0]


def p_with_boundary(pf, i, j):  # noqa: N802
    n0, n1 = pf.shape[0], pf.shape[1]
    if (
        (i == j == 0)
        or (i == n0 - 1 and j == n1 - 1)
        or (i == 0 and j == n1 - 1)
        or (i == n0 - 1 and j == 0)
    ):
        pf[i, j] = 0.0
    elif i == 0:
        pf[0, j] = pf[1, j]
    elif j == 0:
        pf[i, 0] = pf[i, 1]
    elif i == n0 - 1:
        pf[n0 - 1, j] = pf[n0 - 2, j]
    elif j == n1 - 1:
        pf[i, n1 - 1] = pf[i, n1 - 2]
    return pf[i, j]


def c_with_boundary(cf, i, j):  # noqa: N802
    n0, n1 = cf.shape[0], cf.shape[1]
    if (i <= 0) or (i >= n0 - 1) or (j >= n1 - 1) or (j <= 0):
        cf[i, j] = 0.0
    return cf[i, j]


def apply_vel_bc(vf, *, nworkers=None):
    vf = _f(vf, 2, "vf")
    lib().mt_ef_apply_vel_bc(addr(vf), vf.shape[0], vf.shape[1], _nw(nworkers))


def apply_p_bc(pf, *, nworkers=None):
    pf = _f(pf, 1, "pf")
    lib().mt_ef_apply_p_bc(addr(pf), pf.shape[0], pf.shape[1], _nw(nworkers))


def apply_c_bc(cf, *, nworkers=None):
    cf = _f(cf, 1, "cf")
    lib().mt_ef_apply_c_bc(addr(cf), cf.shape[0], cf.shape[1], _nw(nworkers))


def mouse_addspeed(
    cur_posx,  # noqa: N803
    cur_posy,  # noqa: N803
    prev_posx,  # noqa: N803
    prev_posy,  # noqa: N803
    mouseRadius,  # noqa: N803
    vf,
    new_vf,
    *,
    mouse_speed: float = eulerSimParam["mouse_speed"],
    mouse_respondDistance: float = eulerSimParam["mouse_respondDistance"],  # noqa: N803
    nworkers=None,
):
    vf = _f(vf, 2, "vf")
    new_vf = _f(new_vf, 2, "new_vf", vf.shape)
    lib().mt_ef_mouse_addspeed(
        int(cur_posx), int(cur_posy), int(prev_posx), int(prev_posy), float(mouseRadius),
        addr(vf), addr(new_vf), vf.shape[0], vf.shape[1],
        float(mouse_speed), float(mouse_respondDistance), _nw(nworkers),
    )


class TexPair:
    """Upstream's double buffer."""

    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


class EulerFluid:
    """The upstream script's simulation state as an object.

    Field names match upstream: `velocityField`, `colorField`, `curlField`,
    `divField`, `pressField`, `velocities_pair`, `color_pair`, `pressure_pair`.
    """

    def __init__(
        self,
        shape=eulerSimParam["shape"],
        dt: float = eulerSimParam["dt"],
        iteration_step: int = eulerSimParam["iteration_step"],
        mouse_radius: float = eulerSimParam["mouse_radius"],
        mouse_speed: float = eulerSimParam["mouse_speed"],
        mouse_respondDistance: float = eulerSimParam["mouse_respondDistance"],  # noqa: N803
        curl_param: float = eulerSimParam["curl_param"],
        nworkers: int | None = None,
    ):
        self.shape = (int(shape[0]), int(shape[1]))
        self.dt = float(dt)
        self.iteration_step = int(iteration_step)
        self.mouse_radius = float(mouse_radius)
        self.mouse_speed = float(mouse_speed)
        self.mouse_respondDistance = float(mouse_respondDistance)
        self.curl_param = float(curl_param)
        self.nworkers = _default_workers(nworkers)

        self.pressField = guarded_field(self.shape, 1)
        self._new_pressField = guarded_field(self.shape, 1)
        self.velocityField = guarded_field(self.shape, 2)
        self._new_velocityField = guarded_field(self.shape, 2)
        self.colorField = guarded_field(self.shape, 3)
        self._new_colorField = guarded_field(self.shape, 3)
        self.curlField = guarded_field(self.shape, 1)
        self.divField = guarded_field(self.shape, 1)
        self.velocities_pair = TexPair(self.velocityField, self._new_velocityField)
        self.pressure_pair = TexPair(self.pressField, self._new_pressField)
        self.color_pair = TexPair(self.colorField, self._new_colorField)

        init_field(self.pressField, self.velocityField, self.colorField)
        apply_vel_bc(self.velocities_pair.cur, nworkers=self.nworkers)

    def advaction_step(self):  # upstream spelling
        advection(
            self.velocities_pair.cur, self.color_pair.cur, self.color_pair.nxt,
            dt=self.dt, nworkers=self.nworkers,
        )
        advection(
            self.velocities_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt,
            dt=self.dt, nworkers=self.nworkers,
        )
        self.color_pair.swap()
        self.velocities_pair.swap()
        apply_vel_bc(self.velocities_pair.cur, nworkers=self.nworkers)

    def voricity_step(self):  # upstream spelling
        curl(self.velocities_pair.cur, self.curlField, nworkers=self.nworkers)
        apply_c_bc(self.curlField, nworkers=self.nworkers)
        vorticity_projection(
            self.curlField, self.velocities_pair.cur, self.velocities_pair.nxt,
            dt=self.dt, curl_param=self.curl_param, nworkers=self.nworkers,
        )
        self.velocities_pair.swap()
        apply_vel_bc(self.velocities_pair.cur, nworkers=self.nworkers)

    def pressure_step(self):
        divergence(self.velocities_pair.cur, self.divField, nworkers=self.nworkers)
        pressure_solve(
            self.pressure_pair, self.divField,
            iteration_step=self.iteration_step, nworkers=self.nworkers,
        )
        pressure_projection(
            self.pressure_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt,
            nworkers=self.nworkers,
        )
        self.velocities_pair.swap()
        apply_vel_bc(self.velocities_pair.cur, nworkers=self.nworkers)

    def mouse_interaction(self, prev_posx, prev_posy, mouse_x, mouse_y):  # noqa: N802, N803
        mousePos_x = int(mouse_x * self.shape[0])  # noqa: N806
        mousePos_y = int(mouse_y * self.shape[1])  # noqa: N806
        if prev_posx == 0 and prev_posy == 0:
            prev_posx, prev_posy = mousePos_x, mousePos_y
        mouseRadius = self.mouse_radius * min(self.shape[0], self.shape[1])  # noqa: N806
        mouse_addspeed(
            mousePos_x, mousePos_y, prev_posx, prev_posy, mouseRadius,
            self.velocities_pair.cur, self.velocities_pair.nxt,
            mouse_speed=self.mouse_speed,
            mouse_respondDistance=self.mouse_respondDistance,
            nworkers=self.nworkers,
        )
        self.velocities_pair.swap()
        return mousePos_x, mousePos_y

    def step(self, prev_posx=0, prev_posy=0, mouse_x=0.0, mouse_y=0.0):
        """Upstream's frame loop, minus the GUI."""
        self.advaction_step()
        self.voricity_step()
        prev_posx, prev_posy = self.mouse_interaction(
            prev_posx, prev_posy, mouse_x, mouse_y
        )
        self.pressure_step()
        return prev_posx, prev_posy
