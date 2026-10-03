"""Stam's stable-fluids solver: the Python half of
`taichi/examples/simulation/stable_fluid.py`.

Every kernel keeps its upstream name and argument order.  Upstream passes
`ti.template()` fields and reads its parameters from module globals; here the
fields are C-contiguous float32 numpy arrays (the same memory
`field.to_numpy()` returns) and the globals became keyword arguments with
upstream's defaults.  So a call that reads

    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)

reads

    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)

Here too: `vf`, `qf` and `new_qf` keep their positions and names.

Not ported: the `-S` sparse-matrix pressure path (`ti.linalg.SparseSolver`).
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f32, lib
from ._lib import nworkers as _default_workers

res = 512
dt = 0.03
p_jacobi_iters = 500
f_strength = 10000.0
curl_strength = 0
maxfps = 60
time_c = 2
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0


def _res(vf) -> int:
    return int(vf.shape[0])


def _comps(a, name: str) -> int:
    """Components per cell of a field: 1 for a scalar field, else the trailing axis.

    Every kernel here addresses a field as a flat row-major buffer with a
    component stride it fixes at compile time (2 or 3), so a field whose
    trailing extent does not match is a buffer overrun, not a conversion.
    """
    if a.ndim == 2:
        return 1
    if a.ndim == 3:
        return int(a.shape[2])
    raise ValueError(
        f"{name} must be a 2D scalar field or a 2D+vector field, got shape {a.shape}"
    )


def _grid(*fields) -> tuple:
    """Upstream's stable_fluid is one square `res`; every field shares it.

    `vf` is a template upstream, so it is a vec2 velocity field in the
    velocity kernels and a scalar pressure field in the pressure ones; only
    the shared square extent is required here.  The trailing component axis is
    *not* required to match, because upstream genuinely passes a scalar
    `velocity_divs` beside a vec2 `vf` -- the callers that index a field at a
    fixed component stride check that themselves.
    """
    for a in fields:
        f32(a, "field")
    shapes = {a.shape[:2] for a in fields}
    square = len(shapes) == 1 and (n0 := next(iter(shapes)))[0] == n0[1]
    if not square:
        raise ValueError(
            "all stable_fluid fields share one square shape; got "
            + ", ".join(str(a.shape) for a in fields)
        )
    return fields


def _same_extent(*fields) -> tuple:
    """All of `fields` share one square extent *and* one component count.

    A kernel that reads `qf` at component stride 3 and writes `new_qf` at the
    same stride overruns `new_qf` if the two disagree, so that pair is checked
    together rather than field by field.
    """
    shapes = {a.shape for a in fields}
    if len(shapes) != 1:
        raise ValueError(
            "these fields must all have the same shape; got "
            + ", ".join(str(a.shape) for a in fields)
        )
    return fields


def _vec(a, comps: int, name: str) -> None:
    """Reject a field the kernel would index at a different component stride."""
    if _comps(a, name) != comps:
        raise ValueError(
            f"{name} must have {comps} components per cell, got shape {a.shape}"
        )


def _scalar(a, name: str) -> None:
    """Reject a vector field where the kernel indexes one float per cell."""
    _vec(a, 1, name)


def _check_vec2(vf) -> None:
    """The velocity field is vec2 wherever the kernel indexes `vf[i, j]` as one."""
    f32(vf, "vf")
    if vf.ndim != 3 or vf.shape[2] != 2:
        raise ValueError(f"vf must be a vec2 field, got shape {vf.shape}")


def advect(vf, qf, new_qf, *, dt: float = dt, dye_decay: float = dye_decay, nworkers=None):
    """Semi-Lagrangian advection of `qf` along `vf` with 3rd-order RK3 backtrace."""
    vf, qf, new_qf = _grid(vf, qf, new_qf)
    _check_vec2(vf)
    r = _res(vf)
    comps = _comps(qf, "qf")
    if comps not in (2, 3):
        raise ValueError(f"advect is ported for vec2 and vec3 fields, got {qf.shape}")
    # one kernel reads `qf` and writes `new_qf` at the same stride, so the two
    # must be the same width or the write runs off the end of `new_qf`
    qf, new_qf = _same_extent(qf, new_qf)
    fn = lib().mt_sf_advect_v2 if comps == 2 else lib().mt_sf_advect_v3
    fn(addr(vf), addr(qf), addr(new_qf), r, dt, dye_decay, _nw(nworkers))


def apply_impulse(
    vf,
    dyef,
    imp_data,
    *,
    dt: float = dt,
    f_strength: float = f_strength,
    force_radius: float = force_radius,
    nworkers=None,
):
    vf, dyef = _grid(vf, dyef)
    _check_vec2(vf)
    # the dye field is indexed three components wide at a fixed stride
    _vec(dyef, 3, "dyef")
    f32(imp_data, "imp_data", (8,))
    lib().mt_sf_apply_impulse(
        addr(vf), addr(dyef), addr(imp_data), _res(vf), dt, f_strength, force_radius,
        _nw(nworkers),
    )


def divergence(vf, velocity_divs, *, nworkers=None):
    vf, velocity_divs = _grid(vf, velocity_divs)
    _check_vec2(vf)
    _scalar(velocity_divs, "velocity_divs")
    lib().mt_sf_divergence(addr(vf), addr(velocity_divs), _res(vf), _nw(nworkers))


def vorticity(vf, velocity_curls, *, nworkers=None):
    vf, velocity_curls = _grid(vf, velocity_curls)
    _check_vec2(vf)
    _scalar(velocity_curls, "velocity_curls")
    lib().mt_sf_vorticity(addr(vf), addr(velocity_curls), _res(vf), _nw(nworkers))


def pressure_jacobi(pf, new_pf, velocity_divs, *, nworkers=None):
    pf, new_pf, velocity_divs = _grid(pf, new_pf, velocity_divs)
    # all three are scalar fields and all three are indexed at stride one, so a
    # vector field here would be read as scalars and the caller would never learn
    # which half of it the kernel touched
    _scalar(pf, "pf")
    _scalar(new_pf, "new_pf")
    _scalar(velocity_divs, "velocity_divs")
    lib().mt_sf_pressure_jacobi(
        addr(pf), addr(new_pf), addr(velocity_divs), _res(pf), _nw(nworkers)
    )


def subtract_gradient(vf, pf, *, nworkers=None):
    vf, pf = _grid(vf, pf)
    _check_vec2(vf)
    _scalar(pf, "pf")
    lib().mt_sf_subtract_gradient(addr(vf), addr(pf), _res(vf), _nw(nworkers))


def enhance_vorticity(
    vf, cf, *, curl_strength: float = curl_strength, dt: float = dt, nworkers=None
):
    vf, cf = _grid(vf, cf)
    _check_vec2(vf)
    _scalar(cf, "cf")
    lib().mt_sf_enhance_vorticity(
        addr(vf), addr(cf), _res(vf), curl_strength, dt, _nw(nworkers)
    )


def copy_divergence(div_in, div_out, *, nworkers=None):
    div_in, div_out = _grid(div_in, div_out)
    _scalar(div_in, "div_in")
    _scalar(div_out, "div_out")
    lib().mt_sf_copy_divergence(addr(div_in), addr(div_out), _res(div_in), _nw(nworkers))


def apply_pressure(p_in, p_out, *, nworkers=None):
    p_in, p_out = _grid(p_in, p_out)
    _scalar(p_in, "p_in")
    _scalar(p_out, "p_out")
    lib().mt_sf_apply_pressure(addr(p_in), addr(p_out), _res(p_in), _nw(nworkers))


def _nw(requested):
    return _default_workers(requested)


class TexPair:
    """Upstream's double buffer."""

    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


class StableFluid:
    """The whole upstream script's simulation state as an object.

    Field names match upstream: `velocities_pair`, `pressures_pair`,
    `dyes_pair`, `velocity_divs`, `velocity_curls`.
    """

    def __init__(
        self,
        res: int = res,
        dt: float = dt,
        p_jacobi_iters: int = p_jacobi_iters,
        f_strength: float = f_strength,
        curl_strength: float = curl_strength,
        dye_decay: float = dye_decay,
        force_radius: float | None = None,
        nworkers: int | None = None,
    ):
        self.res = int(res)
        self.dt = float(dt)
        self.p_jacobi_iters = int(p_jacobi_iters)
        self.f_strength = float(f_strength)
        self.curl_strength = float(curl_strength)
        self.dye_decay = float(dye_decay)
        self.force_radius = float(res / 2.0 if force_radius is None else force_radius)
        self.nworkers = _default_workers(nworkers)

        r = self.res
        velocities = np.zeros((r, r, 2), dtype=np.float32)
        new_velocities = np.zeros((r, r, 2), dtype=np.float32)
        self.velocity_divs = np.zeros((r, r), dtype=np.float32)
        self.velocity_curls = np.zeros((r, r), dtype=np.float32)
        pressures = np.zeros((r, r), dtype=np.float32)
        new_pressures = np.zeros((r, r), dtype=np.float32)
        dyes = np.zeros((r, r, 3), dtype=np.float32)
        new_dyes = np.zeros((r, r, 3), dtype=np.float32)
        self.velocities_pair = TexPair(velocities, new_velocities)
        self.pressures_pair = TexPair(pressures, new_pressures)
        self.dyes_pair = TexPair(dyes, new_dyes)

    def solve_pressure_jacobi(self):
        for _ in range(self.p_jacobi_iters):
            pressure_jacobi(
                self.pressures_pair.cur,
                self.pressures_pair.nxt,
                self.velocity_divs,
                nworkers=self.nworkers,
            )
            self.pressures_pair.swap()

    def step(self, mouse_data):
        advect(
            self.velocities_pair.cur, self.velocities_pair.cur,
            self.velocities_pair.nxt, dt=self.dt, dye_decay=self.dye_decay,
            nworkers=self.nworkers,
        )
        advect(
            self.velocities_pair.cur, self.dyes_pair.cur, self.dyes_pair.nxt,
            dt=self.dt, dye_decay=self.dye_decay, nworkers=self.nworkers,
        )
        self.velocities_pair.swap()
        self.dyes_pair.swap()

        apply_impulse(
            self.velocities_pair.cur, self.dyes_pair.cur, mouse_data,
            dt=self.dt, f_strength=self.f_strength, force_radius=self.force_radius,
            nworkers=self.nworkers,
        )

        divergence(self.velocities_pair.cur, self.velocity_divs, nworkers=self.nworkers)

        if self.curl_strength:
            vorticity(self.velocities_pair.cur, self.velocity_curls, nworkers=self.nworkers)
            enhance_vorticity(
                self.velocities_pair.cur, self.velocity_curls,
                curl_strength=self.curl_strength, dt=self.dt, nworkers=self.nworkers,
            )

        self.solve_pressure_jacobi()

        subtract_gradient(
            self.velocities_pair.cur, self.pressures_pair.cur, nworkers=self.nworkers
        )

    def reset(self):
        self.velocities_pair.cur.fill(0)
        self.pressures_pair.cur.fill(0)
        self.dyes_pair.cur.fill(0)


def impulse(mouse_dir=(0.0, 0.0), mouse_pos=(0.0, 0.0), color=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Upstream's 8-float `mouse_data` layout, as a float32 array."""
    m = np.zeros(8, dtype=np.float32)
    m[0], m[1] = mouse_dir
    m[2], m[3] = mouse_pos
    m[4], m[5], m[6] = color
    return m
