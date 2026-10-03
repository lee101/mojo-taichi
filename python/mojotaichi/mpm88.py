"""The Python half of `taichi/examples/simulation/mpm88.py`.

MPM88: 2D material point method, quadratic B-spline transfer, APIC affine
velocity field.  Same names, same argument order; upstream's module-level
scalars became keyword arguments with upstream's defaults.

`init` takes the uniform [0, 1) draws as an argument instead of calling
`ti.random()`, so the caller owns the seed and the port is reproducible.
Upstream's `ti.random()` stream cannot be reproduced bit-for-bit anyway.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f32, lib
from ._lib import guarded as guarded_field
from ._lib import nworkers as _default_workers

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400


def substep(
    x,
    v,
    C,
    J,
    grid_v,
    grid_m,
    *,
    n_grid: int = n_grid,
    dx: float = dx,
    dt: float = dt,
    p_vol: float = p_vol,
    p_mass: float = p_mass,
    E: float = E,
    gravity: float = gravity,
    bound: int = bound,
    nworkers=None,
):
    """One MPM88 timestep: clear grid, P2G, normalize + gravity, G2P."""
    n_particles = len(x)
    # the kernel indexes v, C and J to n_particles, so they must all be that long
    f32(x, "x", (n_particles, 2))
    f32(v, "v", (n_particles, 2))
    f32(C, "C", (n_particles, 2, 2))
    f32(J, "J", (n_particles,))
    f32(grid_v, "grid_v", (int(n_grid), int(n_grid), 2))
    f32(grid_m, "grid_m", (int(n_grid), int(n_grid)))
    lib().mt_mpm88_substep(
        addr(x), addr(v), addr(C), addr(J), addr(grid_v), addr(grid_m),
        int(n_particles), int(n_grid), float(dx), float(dt), float(p_vol),
        float(p_mass), float(E), float(gravity), int(bound),
        _default_workers(nworkers),
    )


def init(x, v, J, uniforms):  # noqa: N803
    """Place particles at `uniforms * 0.4 + 0.2`, `v = [0, -1]`, `J = 1`."""
    n_particles = len(x)
    f32(x, "x", (n_particles, 2))
    f32(v, "v", (n_particles, 2))
    f32(J, "J", (n_particles,))
    f32(uniforms, "uniforms", (n_particles, 2))
    lib().mt_mpm88_init(addr(x), addr(v), addr(J), addr(uniforms), int(n_particles))


class MPM88:
    """The upstream script's state as an object; field names match upstream."""

    def __init__(
        self,
        n_particles: int = n_particles,
        n_grid: int = n_grid,
        dx: float | None = None,
        dt: float = dt,
        p_rho: float = p_rho,
        p_vol: float | None = None,
        gravity: float = gravity,
        bound: int = bound,
        E: float = E,  # noqa: N803
        nworkers: int | None = None,
    ):
        self.n_particles = int(n_particles)
        self.n_grid = int(n_grid)
        self.dx = float(1 / n_grid if dx is None else dx)
        self.dt = float(dt)
        self.p_rho = float(p_rho)
        self.p_vol = float((self.dx * 0.5) ** 2 if p_vol is None else p_vol)
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = float(gravity)
        self.bound = int(bound)
        self.E = float(E)
        self.nworkers = _default_workers(nworkers)

        n, g = self.n_particles, self.n_grid
        self.x = np.zeros((n, 2), dtype=np.float32)
        self.v = np.zeros((n, 2), dtype=np.float32)
        self.C = np.zeros((n, 2, 2), dtype=np.float32)
        self.J = np.zeros(n, dtype=np.float32)
        # the grid is scattered into, so it carries the same two-cell guard
        self.grid_v = guarded_field((g, g), 2)
        self.grid_m = guarded_field((g, g), 1)

    def init(self, seed: int = 0):
        uniforms = np.random.default_rng(seed).random(
            (self.n_particles, 2), dtype=np.float32
        )
        init(self.x, self.v, self.J, uniforms)
        return self

    def substep(self):
        substep(
            self.x, self.v, self.C, self.J, self.grid_v, self.grid_m,
            n_grid=self.n_grid, dx=self.dx, dt=self.dt, p_vol=self.p_vol,
            p_mass=self.p_mass, E=self.E, gravity=self.gravity, bound=self.bound,
            nworkers=self.nworkers,
        )

    def run(self, steps: int):
        for _ in range(steps):
            self.substep()
        return self
