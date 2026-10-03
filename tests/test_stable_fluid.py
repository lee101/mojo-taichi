"""Parity between mojo-taichi's stable_fluid kernels and the installed taichi
example `taichi/examples/simulation/stable_fluid.py`.

The upstream kernels are exec'd straight out of the installed taichi package
(see `conftest.load_upstream`), with the module globals they read supplied by
the test, so the arithmetic being compared is upstream's own source.
"""

import numpy as np
import pytest

import mojotaichi
from mojotaichi import stable_fluid as mtsf
from conftest import assert_same, field, get_field, load_upstream, set_field

RES = 16
DT = 0.03
DYE_DECAY = 1 - 1 / (60 * 2)
F_STRENGTH = 10000.0
FORCE_RADIUS = RES / 2.0
P_JACOBI_ITERS = 20

SF = "examples/simulation/stable_fluid.py"

KERNELS = [
    "sample", "lerp", "bilerp", "backtrace", "TexPair", "advect", "apply_impulse",
    "divergence", "vorticity", "pressure_jacobi", "subtract_gradient",
    "enhance_vorticity", "copy_divergence", "apply_pressure", "solve_pressure_jacobi",
    "step", "reset",
]


@pytest.fixture
def upstream():
    ti = pytest.importorskip("taichi")
    ns = {
        "ti": ti,
        "np": np,
        "res": RES,
        "dt": DT,
        "dye_decay": DYE_DECAY,
        "f_strength": F_STRENGTH,
        "force_radius": FORCE_RADIUS,
        "curl_strength": 0,
        "p_jacobi_iters": P_JACOBI_ITERS,
    }
    return load_upstream(SF, KERNELS, ns)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_field(rng, shape, comps=0, scale=0.1):
    if comps:
        return (rng.normal(size=shape + (comps,)) * scale).astype(np.float32)
    return (rng.normal(size=shape) * scale).astype(np.float32)


# ------------------------------------------------------------------- advect
def test_advect_v2(upstream, rng):
    q = random_field(rng, (RES, RES), 2)
    v = random_field(rng, (RES, RES), 2, 0.5)
    qf, vf, nq = field((RES, RES), 2), field((RES, RES), 2), field((RES, RES), 2)
    set_field(qf, q)
    set_field(vf, v)

    upstream["advect"](vf, qf, nq)
    out = np.zeros_like(q)
    mtsf.advect(v, q, out, dt=DT, dye_decay=DYE_DECAY, nworkers=4)

    assert_same(get_field(nq), out, name="advect vec2")


def test_advect_v3(upstream, rng):
    q = random_field(rng, (RES, RES), 3)
    v = random_field(rng, (RES, RES), 2, 0.5)
    qf, vf, nq = field((RES, RES), 3), field((RES, RES), 2), field((RES, RES), 3)
    set_field(qf, q)
    set_field(vf, v)

    upstream["advect"](vf, qf, nq)
    out = np.zeros_like(q)
    mtsf.advect(v, q, out, dt=DT, dye_decay=DYE_DECAY, nworkers=4)

    assert_same(get_field(nq), out, name="advect vec3")


def test_advect_decays_dye(upstream, rng):
    """With a zero velocity field, `advect` is pure bilinear sampling * decay."""
    q = random_field(rng, (RES, RES), 3)
    v = np.zeros((RES, RES, 2), dtype=np.float32)
    qf, vf, nq = field((RES, RES), 3), field((RES, RES), 2), field((RES, RES), 3)
    set_field(qf, q)
    set_field(vf, v)
    upstream["advect"](vf, qf, nq)
    out = np.zeros_like(q)
    mtsf.advect(v, q, out, dt=DT, dye_decay=DYE_DECAY, nworkers=4)
    assert_same(get_field(nq), out, name="advect decay")


# ------------------------------------------------------------- apply_impulse
def test_apply_impulse(upstream, rng):
    v = random_field(rng, (RES, RES), 2)
    dye = random_field(rng, (RES, RES), 3)
    imp = mojotaichi.impulse(
        mouse_dir=(0.8, -0.3), mouse_pos=(7.0, 9.0), color=(0.2, 0.5, 1.0)
    )
    vf, dyef = field((RES, RES), 2), field((RES, RES), 3)
    set_field(vf, v)
    set_field(dyef, dye)
    upstream["apply_impulse"](vf, dyef, imp)

    got_v, got_d = np.array(v), np.array(dye)
    mtsf.apply_impulse(
        got_v, got_d, imp, dt=DT, f_strength=F_STRENGTH, force_radius=FORCE_RADIUS,
        nworkers=4,
    )
    assert_same(get_field(vf), got_v, name="apply_impulse velocity")
    assert_same(get_field(dyef), got_d, name="apply_impulse dye")


# ------------------------------------------------------------- local stencils
def test_divergence(upstream, rng):
    v = random_field(rng, (RES, RES), 2)
    vf, div = field((RES, RES), 2), field((RES, RES))
    set_field(vf, v)
    upstream["velocity_divs"] = div
    upstream["divergence"](vf)
    out = np.zeros((RES, RES), dtype=np.float32)
    mtsf.divergence(v, out, nworkers=4)
    assert_same(get_field(div), out, name="divergence")


def test_vorticity(upstream, rng):
    v = random_field(rng, (RES, RES), 2)
    vf, curl = field((RES, RES), 2), field((RES, RES))
    set_field(vf, v)
    upstream["velocity_curls"] = curl
    upstream["vorticity"](vf)
    out = np.zeros((RES, RES), dtype=np.float32)
    mtsf.vorticity(v, out, nworkers=4)
    assert_same(get_field(curl), out, name="vorticity")


def test_pressure_jacobi(upstream, rng):
    p = random_field(rng, (RES, RES))
    d = random_field(rng, (RES, RES))
    pf, npf, df = field((RES, RES)), field((RES, RES)), field((RES, RES))
    set_field(pf, p)
    set_field(df, d)
    upstream["velocity_divs"] = df
    upstream["pressure_jacobi"](pf, npf)
    out = np.zeros((RES, RES), dtype=np.float32)
    mtsf.pressure_jacobi(p, out, d, nworkers=4)
    assert_same(get_field(npf), out, name="pressure_jacobi")


def test_subtract_gradient(upstream, rng):
    v = random_field(rng, (RES, RES), 2)
    p = random_field(rng, (RES, RES))
    vf, pf = field((RES, RES), 2), field((RES, RES))
    set_field(vf, v)
    set_field(pf, p)
    upstream["subtract_gradient"](vf, pf)
    out = np.array(v)
    mtsf.subtract_gradient(out, p, nworkers=4)
    assert_same(get_field(vf), out, name="subtract_gradient")


def test_enhance_vorticity(upstream, rng):
    v = random_field(rng, (RES, RES), 2)
    c = random_field(rng, (RES, RES))
    upstream["curl_strength"] = 7.0
    vf, cf = field((RES, RES), 2), field((RES, RES))
    set_field(vf, v)
    set_field(cf, c)
    upstream["enhance_vorticity"](vf, cf)
    out = np.array(v)
    mtsf.enhance_vorticity(out, c, curl_strength=7.0, dt=DT, nworkers=4)
    assert_same(get_field(vf), out, name="enhance_vorticity")


def test_copy_divergence_and_apply_pressure(upstream, rng):
    # both take a 1D `ti.types.ndarray` for the flat side, as upstream declares
    ti = upstream["ti"]
    d = random_field(rng, (RES, RES))
    p = random_field(rng, (RES, RES))
    df = field((RES, RES))
    set_field(df, d)
    flat = np.zeros(RES * RES, dtype=np.float32)
    upstream["copy_divergence"](df, flat)
    out = np.zeros_like(d)
    mtsf.copy_divergence(d, out, nworkers=4)
    assert_same(flat.reshape(RES, RES), out, name="copy_divergence")
    assert np.allclose(out, -d, atol=0)

    pf = field((RES, RES))
    flat2 = np.ascontiguousarray(p.reshape(-1), dtype=np.float32)
    upstream["apply_pressure"](flat2, pf)
    out = np.zeros_like(p)
    mtsf.apply_pressure(p, out, nworkers=4)
    assert_same(get_field(pf), out, name="apply_pressure")


# -------------------------------------------------------------- pressure solve
def test_solve_pressure_jacobi(upstream, rng):
    """Upstream's `solve_pressure_jacobi` loop, kernel for kernel."""
    p = random_field(rng, (RES, RES))
    d = random_field(rng, (RES, RES))
    pair = upstream["TexPair"](field((RES, RES)), field((RES, RES)))
    divs = field((RES, RES))
    set_field(pair.cur, p)
    set_field(divs, d)
    upstream["pressures_pair"] = pair
    upstream["velocity_divs"] = divs
    upstream["solve_pressure_jacobi"]()

    ours = mojotaichi.StableFluid(res=RES, p_jacobi_iters=P_JACOBI_ITERS)
    ours.pressures_pair.cur[...] = p
    ours.velocity_divs[...] = d
    ours.solve_pressure_jacobi()

    assert_same(get_field(pair.cur), ours.pressures_pair.cur, name="pressure solve")


def test_step(upstream, rng):
    """Upstream `step` with the optional sparse-matrix path off."""
    sim = mojotaichi.StableFluid(res=RES, p_jacobi_iters=P_JACOBI_ITERS)
    v0 = random_field(rng, (RES, RES), 2, 3.0)
    d0 = random_field(rng, (RES, RES), 3)
    sim.velocities_pair.cur[...] = v0
    sim.dyes_pair.cur[...] = d0
    imp = mojotaichi.impulse(
        mouse_dir=(0.5, 0.5), mouse_pos=(8.0, 4.0), color=(0.9, 0.1, 0.1)
    )

    pair_v = upstream["TexPair"](field((RES, RES), 2), field((RES, RES), 2))
    pair_p = upstream["TexPair"](field((RES, RES)), field((RES, RES)))
    pair_d = upstream["TexPair"](field((RES, RES), 3), field((RES, RES), 3))
    set_field(pair_v.cur, v0)
    set_field(pair_d.cur, d0)
    divs = field((RES, RES))
    upstream.update(
        velocities_pair=pair_v, pressures_pair=pair_p, dyes_pair=pair_d,
        velocity_divs=divs, velocity_curls=field((RES, RES)),
        use_sparse_matrix=False, debug=False, curl_strength=0,
    )
    upstream["step"](imp)

    sim.step(imp)

    assert_same(get_field(pair_v.cur), sim.velocities_pair.cur, name="step velocity")
    assert_same(get_field(pair_d.cur), sim.dyes_pair.cur, name="step dye")
    assert_same(get_field(pair_p.cur), sim.pressures_pair.cur, name="step pressure")
    assert_same(get_field(divs), sim.velocity_divs, name="step divergence")


def test_reset(upstream, rng):
    sim = mojotaichi.StableFluid(res=RES)
    sim.velocities_pair.cur[...] = 1.0
    sim.dyes_pair.cur[...] = 1.0
    sim.reset()
    assert not sim.velocities_pair.cur.any()
    assert not sim.dyes_pair.cur.any()


# ------------------------------------------------------------------ behaviour
def test_subtract_gradient_removes_a_pure_gradient(rng):
    """Feed it exactly the gradient of p and it must return a zero field."""
    p = random_field(rng, (RES, RES))
    i, j = np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")
    pl = p[np.clip(i - 1, 0, RES - 1), np.clip(j, 0, RES - 1)]
    pr = p[np.clip(i + 1, 0, RES - 1), np.clip(j, 0, RES - 1)]
    pb = p[np.clip(i, 0, RES - 1), np.clip(j - 1, 0, RES - 1)]
    pt = p[np.clip(i, 0, RES - 1), np.clip(j + 1, 0, RES - 1)]
    v = np.stack([0.5 * (pr - pl), 0.5 * (pt - pb)], axis=-1).astype(np.float32)
    out = np.array(v)
    mtsf.subtract_gradient(out, p, nworkers=4)
    assert np.abs(out).max() < 1e-4 * max(1.0, np.abs(v).max())


def test_pressure_jacobi_converges_to_poisson_solution(rng):
    """`pressure_jacobi` has the Poisson fixed point: sum(neighbours) - 4p = div."""
    r = 32
    d = random_field(rng, (r, r))
    p = np.zeros((r, r), dtype=np.float32)
    nxt = np.zeros((r, r), dtype=np.float32)
    for _ in range(3000):
        mtsf.pressure_jacobi(p, nxt, d, nworkers=4)
        p, nxt = nxt, p
    i, j = np.meshgrid(np.arange(r), np.arange(r), indexing="ij")
    pl = p[np.clip(i - 1, 0, r - 1), np.clip(j, 0, r - 1)]
    pr = p[np.clip(i + 1, 0, r - 1), np.clip(j, 0, r - 1)]
    pb = p[np.clip(i, 0, r - 1), np.clip(j - 1, 0, r - 1)]
    pt = p[np.clip(i, 0, r - 1), np.clip(j + 1, 0, r - 1)]
    residual = (pl + pr + pb + pt - 4.0 * p) - d
    assert np.abs(residual).max() < 1e-2 * max(1.0, np.abs(d).max())


def test_advect_parallel_path_matches_serial_path(rng):
    """`advect` must give the same answer whichever driver it is given.

    `PAR_MIN` currently sits past every size the library accepts, so both calls
    take the plain loop; this still holds if it is ever lowered to a reachable
    size, which is when it would start to matter.
    """
    big = 512
    v = random_field(rng, (big, big), 2, 0.2)
    q = random_field(rng, (big, big), 3, 0.2)
    ser = np.zeros_like(q)
    par = np.zeros_like(q)
    mtsf.advect(v, q, ser, dt=DT, dye_decay=DYE_DECAY, nworkers=1)
    mtsf.advect(v, q, par, dt=DT, dye_decay=DYE_DECAY, nworkers=8)
    assert_same(par, ser, name="advect parallel vs serial")
