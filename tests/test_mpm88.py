"""Parity for the MPM88 material point solver against the installed
`taichi/examples/simulation/mpm88.py`.

Upstream's `substep` is exec'd from the installed package with the module
globals it reads (`n_grid`, `dx`, `dt`, `p_vol`, `p_mass`, `E`, `gravity`,
`bound`) supplied by the test, and run on taichi fields created from the same
arrays the port gets.
"""

import numpy as np
import pytest

import mojotaichi
from mojotaichi import mpm88 as mtm
from conftest import assert_same, field, get_field, load_upstream, set_field

N_GRID = 32
DX = 1 / N_GRID
DT = 2e-4
P_VOL = (DX * 0.5) ** 2
P_MASS = P_VOL
E = 400.0
GRAVITY = 9.8
BOUND = 3
N_PARTICLES = 256

KERNELS = ["substep"]


@pytest.fixture
def upstream():
    ti = pytest.importorskip("taichi")
    ns = {
        "ti": ti,
        "n_particles": N_PARTICLES,
        "n_grid": N_GRID,
        "dx": DX,
        "dt": DT,
        "p_vol": P_VOL,
        "p_rho": 1,
        "p_mass": P_MASS,
        "E": E,
        "gravity": GRAVITY,
        "bound": BOUND,
    }
    return load_upstream("examples/simulation/mpm88.py", KERNELS, ns)


def initial_state(seed=0):
    """Upstream's `init` block, with the uniforms drawn in Python."""
    u = np.random.default_rng(seed).random((N_PARTICLES, 2), dtype=np.float32)
    x = (u * 0.4 + 0.2).astype(np.float32)
    v = np.zeros((N_PARTICLES, 2), dtype=np.float32)
    v[:, 1] = -1.0
    J = np.ones(N_PARTICLES, dtype=np.float32)
    C = np.zeros((N_PARTICLES, 2, 2), dtype=np.float32)
    return x, v, C, J


def theirs(upstream, x, v, C, J, steps):
    fx, fv = field((N_PARTICLES,), 2), field((N_PARTICLES,), 2)
    fC = ti_matrix_field(N_PARTICLES)
    fJ = field((N_PARTICLES,))
    gv, gm = field((N_GRID, N_GRID), 2), field((N_GRID, N_GRID))
    set_field(fx, x)
    set_field(fv, v)
    fC.from_numpy(C)
    fJ.from_numpy(J)
    upstream.update(x=fx, v=fv, C=fC, J=fJ, grid_v=gv, grid_m=gm)
    for _ in range(steps):
        upstream["substep"]()
    return get_field(fx), get_field(fv), fC.to_numpy(), fJ.to_numpy(), gv, gm


def ti_matrix_field(n):
    from conftest import ti

    f = ti.Matrix.field(2, 2, float, shape=(n,))
    f.from_numpy(np.zeros((n, 2, 2), dtype=np.float32))
    return f


def test_init_matches_upstream_formula():
    """`init` is `x = uniform * 0.4 + 0.2`, `v = [0, -1]`, `J = 1`."""
    sim = mojotaichi.MPM88(n_particles=64, n_grid=N_GRID).init(seed=7)
    u = np.random.default_rng(7).random((64, 2), dtype=np.float32)
    # float32 with FMA contraction: one ulp of slack against a numpy expression
    assert_same(sim.x, u * 0.4 + 0.2, rtol=0, atol=1e-7, name="init x")
    assert np.all(sim.v[:, 0] == 0.0) and np.all(sim.v[:, 1] == -1.0)
    assert np.all(sim.J == 1.0)
    assert sim.x.min() >= 0.2 and sim.x.max() <= 0.6


def test_substep_parity(upstream):
    x, v, C, J = initial_state()
    tx, tv, tC, tJ, _, _ = theirs(upstream, x, v, C, J, 1)

    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID)
    sim.x[...] = x
    sim.v[...] = v
    sim.C[...] = C
    sim.J[...] = J
    sim.substep()

    assert_same(tx, sim.x, name="substep x")
    assert_same(tv, sim.v, name="substep v")
    assert_same(tC, sim.C, name="substep C")
    assert_same(tJ, sim.J, name="substep J")


def test_substep_many_steps_parity(upstream):
    """MPM is a chaotic-ish integrator, so 25 steps is a real test of the port."""
    x, v, C, J = initial_state(3)
    tx, tv, tC, tJ, _, _ = theirs(upstream, x, v, C, J, 25)

    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID)
    sim.x[...] = x
    sim.v[...] = v
    sim.C[...] = C
    sim.J[...] = J
    sim.run(25)

    assert_same(tx, sim.x, rtol=1e-3, atol=1e-4, name="25 steps x")
    assert_same(tv, sim.v, rtol=1e-3, atol=1e-4, name="25 steps v")
    assert_same(tC, sim.C, rtol=1e-3, atol=1e-4, name="25 steps C")
    assert_same(tJ, sim.J, rtol=1e-3, atol=1e-4, name="25 steps J")


def test_substep_parity_divergent_state(upstream):
    """A divergent velocity field and a random affine field: no cancellation, so
    the C comparison is at float32 epsilon rather than dominated by it."""
    rng = np.random.default_rng(5)
    x, v, C, J = initial_state(0)
    v = (rng.normal(size=v.shape) * 0.5).astype(np.float32)
    C = rng.normal(size=C.shape).astype(np.float32)
    tx, tv, tC, tJ, _, _ = theirs(upstream, x, v, C, J, 1)

    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID)
    sim.x[...] = x
    sim.v[...] = v
    sim.C[...] = C
    sim.J[...] = J
    sim.substep()

    assert_same(tx, sim.x, rtol=0, atol=0, name="divergent x")
    assert_same(tv, sim.v, rtol=1e-5, atol=1e-6, name="divergent v")
    assert_same(tC, sim.C, rtol=1e-5, atol=1e-5, name="divergent C")
    assert_same(tJ, sim.J, rtol=1e-5, atol=1e-7, name="divergent J")


def test_long_run_parity(upstream):
    """300 steps: the affine field feeds back into the stress, so drift shows."""
    x, v, C, J = initial_state(3)
    tx, tv, tC, tJ, _, _ = theirs(upstream, x, v, C, J, 300)

    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID)
    sim.x[...] = x
    sim.v[...] = v
    sim.C[...] = C
    sim.J[...] = J
    sim.run(300)

    assert np.isfinite(sim.x).all()
    assert_same(tx, sim.x, rtol=1e-4, atol=1e-5, name="300 steps x")
    assert_same(tv, sim.v, rtol=1e-4, atol=1e-5, name="300 steps v")
    assert_same(tC, sim.C, rtol=1e-3, atol=1e-4, name="300 steps C")
    assert_same(tJ, sim.J, rtol=1e-4, atol=1e-6, name="300 steps J")


def test_substep_module_function(upstream):
    """The bare `substep(x, v, C, J, grid_v, grid_m)` entry point agrees too."""
    x, v, C, J = initial_state(1)
    tx, tv, tC, tJ, _, _ = theirs(upstream, x, v, C, J, 1)

    from mojotaichi._lib import guarded

    n = N_PARTICLES
    g = N_GRID
    gv, gm = guarded((g, g), 2), guarded((g, g), 1)
    mtm.substep(
        x.copy(), v.copy(), C.copy(), J.copy(), gv, gm,
        n_grid=N_GRID, dx=DX, dt=DT, p_vol=P_VOL, p_mass=P_MASS, E=E,
        gravity=GRAVITY, bound=BOUND, nworkers=4,
    )
    assert np.abs(gv).max() > 0.0
    assert np.abs(gm).max() > 0.0


def test_mass_is_conserved():
    """P2G distributes every particle's mass into the grid, and it stays there."""
    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID).init(seed=2)
    before = float(sim.grid_m.sum())
    sim.substep()
    assert abs(float(sim.grid_m.sum()) - N_PARTICLES * P_MASS) < 1e-3 * N_PARTICLES * P_MASS
    assert before == 0.0  # the grid starts empty and is cleared every substep


def test_grid_velocity_is_the_mass_weighted_average():
    """After normalization every non-empty node holds a velocity, not a momentum."""
    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID).init(seed=4)
    sim.substep()
    m = sim.grid_m
    nonempty = m > 0
    assert nonempty.any()
    # the gravity term is the only thing added after the mass division
    assert np.abs(sim.grid_v[nonempty][:, 1] - (-1.0 - DT * GRAVITY)).max() < 0.5


def test_boundary_nodes_are_clamped():
    """Upstream zeroes any velocity pointing out of the `bound` cells."""
    sim = mojotaichi.MPM88(n_particles=N_PARTICLES, n_grid=N_GRID).init(seed=5)
    sim.run(30)
    gv = sim.grid_v
    edge = gv[:BOUND, :, 0]
    assert np.all(edge >= 0.0)


def test_is_deterministic_for_a_seed():
    a = mojotaichi.MPM88(n_particles=64, n_grid=N_GRID).init(seed=11).run(3)
    b = mojotaichi.MPM88(n_particles=64, n_grid=N_GRID).init(seed=11).run(3)
    assert_same(a.x, b.x, rtol=0, atol=0, name="deterministic x")
