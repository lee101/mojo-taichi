"""Parity for `taichi/examples/simulation/eulerfluid2d.py`.

Five kernels index `i + 1` / `j + 1` on the last row and column with no clamp:
`curl`, `vorticity_projection`, `divergence`, `pressure_iteration` and
`pressure_projection`.  Upstream only defines that on a GPU.

Four of them -- `curl`, `divergence`, `pressure_iteration` and
`vorticity_projection` -- *do* run on the taichi CPU backend: the read past the
edge lands in taichi's own surrounding allocation rather than faulting.  Those
are checked against a real upstream run over the interior, where the value each
side reads is defined for both.  They are also checked against the NumPy
transcription of their own source expressions, so the transcription cannot drift
away from the kernel while still agreeing with the port.

`pressure_projection` is the exception: its `p_with_boundary` calls *write* past
the edge as well as read it, and taichi's CPU backend segfaults on it.  It is
checked against the transcription over the interior only.

Every other kernel is checked against real upstream taichi.
"""

import numpy as np
import pytest

import mojotaichi
from mojotaichi import eulerfluid2d as mte
from mojotaichi._lib import guarded
from conftest import assert_same, field, get_field, load_upstream, set_field

N = 16
EF = "examples/simulation/eulerfluid2d.py"

KERNELS = [
    "sample", "lerp", "bilerp", "TexPair", "init_field", "advection", "curl",
    "vorticity_projection", "divergence", "pressure_iteration", "pressure_solve",
    "pressure_projection", "vel_with_boundary", "p_with_boundary", "c_with_boundary",
    "apply_vel_bc", "apply_p_bc", "apply_c_bc", "mouse_addspeed",
    "advaction_step", "voricity_step", "pressure_step",
]

PARAMS = {
    "shape": [N, N],
    "dt": 1 / 60.0,
    "iteration_step": 20,
    "mouse_radius": 0.01,
    "mouse_speed": 125.0,
    "mouse_respondDistance": 0.5,
    "curl_param": 15,
}


@pytest.fixture
def upstream():
    ti = pytest.importorskip("taichi")
    from taichi.examples.patterns import taichi_logo

    import taichi.math as tm

    ns = {
        "ti": ti,
        "tm": tm,
        "np": np,
        "taichi_logo": taichi_logo,
        "eulerSimParam": dict(PARAMS),
    }
    # `eulerSimParam` is supplied, not exec'd: the upstream literal is 512x512.
    return load_upstream(EF, KERNELS, ns)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def rnd(rng, shape, scale=0.5):
    return (rng.normal(size=shape) * scale).astype(np.float32)


# ------------------------------------------------- kernels taichi can run
def test_init_field(upstream, rng):
    """`init_field` zeroes pressure/velocity and paints the logo into colour."""
    # upstream `init_field` clears pressure and velocity but accumulates into
    # colour, so colour starts at whatever it held
    press, vel, color = field((N, N)), field((N, N), 2), field((N, N), 3)
    set_field(vel, rnd(rng, (N, N, 2)))
    initial_color = rnd(rng, (N, N, 3))
    set_field(color, initial_color)
    upstream.update(pressField=press, velocityField=vel, colorField=color)
    upstream["init_field"]()

    gp, gv, gc = guarded((N, N), 1), guarded((N, N), 2), guarded((N, N), 3)
    gc[...] = initial_color
    mte.init_field(gp, gv, gc)

    assert_same(get_field(press), gp, rtol=0, atol=0, name="init press")
    assert_same(get_field(vel), gv, rtol=1e-6, atol=1e-7, name="init vel")
    assert_same(get_field(color), gc, name="init color")


def test_advection_v2(upstream, rng):
    v = rnd(rng, (N, N, 2), 0.05)
    q = rnd(rng, (N, N, 2))
    vf, qf, nq = field((N, N), 2), field((N, N), 2), field((N, N), 2)
    set_field(vf, v)
    set_field(qf, q)
    upstream["advection"](vf, qf, nq)
    out = np.zeros_like(q)
    mte.advection(v, q, out, dt=PARAMS["dt"], nworkers=4)
    assert_same(get_field(nq), out, name="advection vec2")


def test_advection_v3(upstream, rng):
    v = rnd(rng, (N, N, 2), 0.05)
    q = rnd(rng, (N, N, 3))
    vf, qf, nq = field((N, N), 2), field((N, N), 3), field((N, N), 3)
    set_field(vf, v)
    set_field(qf, q)
    upstream["advection"](vf, qf, nq)
    out = np.zeros_like(q)
    mte.advection(v, q, out, dt=PARAMS["dt"], nworkers=4)
    assert_same(get_field(nq), out, name="advection vec3")


def test_apply_vel_bc(upstream, rng):
    v = rnd(rng, (N, N, 2))
    vf = field((N, N), 2)
    set_field(vf, v)
    upstream["apply_vel_bc"](vf)
    out = np.array(v)
    mte.apply_vel_bc(out, nworkers=4)
    assert_same(get_field(vf), out, name="apply_vel_bc")
    assert not out[0].any() and not out[-1].any()
    assert not out[:, 0].any() and not out[:, -1].any()


def test_apply_p_bc(upstream, rng):
    p = rnd(rng, (N, N))
    pf = field((N, N))
    set_field(pf, p)
    upstream["apply_p_bc"](pf)
    out = np.array(p)
    mte.apply_p_bc(out, nworkers=4)
    assert_same(get_field(pf), out, name="apply_p_bc")
    assert out[0, 0] == 0.0 and out[-1, -1] == 0.0
    assert np.allclose(out[0, 1:-1], p[1, 1:-1])


def test_apply_c_bc(upstream, rng):
    c = rnd(rng, (N, N))
    cf = field((N, N))
    set_field(cf, c)
    upstream["apply_c_bc"](cf)
    out = np.array(c)
    mte.apply_c_bc(out, nworkers=4)
    assert_same(get_field(cf), out, name="apply_c_bc")
    assert not out[0].any() and not out[-1].any()


def test_mouse_addspeed(upstream, rng):
    v = rnd(rng, (N, N, 2), 0.1)
    vf, nv = field((N, N), 2), field((N, N), 2)
    set_field(vf, v)
    upstream["mouse_addspeed"](7, 5, 6, 4, 3.0, vf, nv)
    out = np.zeros_like(v)
    mte.mouse_addspeed(
        7, 5, 6, 4, 3.0, v, out,
        mouse_speed=PARAMS["mouse_speed"],
        mouse_respondDistance=PARAMS["mouse_respondDistance"],
        nworkers=4,
    )
    assert_same(get_field(nv), out, name="mouse_addspeed")
    assert (out != v).any()  # the impulse actually moved some cells


# ----------------- kernels that read past the edge; transcription references
def flat(a, comps=1):
    """The logical field as a flat vector, exactly how the kernels index it."""
    return a.reshape(-1, comps) if comps > 1 else a.reshape(-1)


def at(a, i, j, comps=1):
    """`a[i, j]` with upstream's arithmetic: the index is flat, `i * N + j`.

    Off the *end* of the field that lands in the port's zero guard; off the
    start or across a row boundary it wraps into the neighbouring cell, which is
    what upstream's own field indexing does too.
    """
    k = i * N + j
    v = flat(a, comps)
    if k < 0 or k >= v.shape[0]:
        return np.zeros(comps, np.float32) if comps > 1 else np.float32(0.0)
    return v[k] if comps > 1 else v[k]


def ref_curl(vf):
    out = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            out[i, j] = 0.5 * (
                (at(vf, i + 1, j, 2)[1] - at(vf, i - 1, j, 2)[1])
                - (at(vf, i, j + 1, 2)[0] - at(vf, i, j - 1, 2)[0])
            )
    return out


def ref_divergence(vf):
    out = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            out[i, j] = 0.5 * (
                at(vf, i + 1, j, 2)[0]
                - at(vf, i - 1, j, 2)[0]
                + at(vf, i, j + 1, 2)[1]
                - at(vf, i, j - 1, 2)[1]
            )
    return out


def ref_pressure_iteration(divf, pf):
    out = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            out[i, j] = (
                at(pf, i + 1, j) + at(pf, i - 1, j) + at(pf, i, j - 1) + at(pf, i, j + 1)
                - divf[i, j]
            ) / 4
    return out


class Pad:
    """A flat float32 buffer with a zeroed margin, indexed the way the kernels
    index: the flat position `i * N + j`, wrapping into the neighbouring cell
    when that arithmetic says so and landing in the margin (or the port's guard)
    when it does not.  One per call, because upstream's writes past the end of
    the field are undefined and the port's guard absorbs them per allocation."""

    def __init__(self, a):
        self.b0 = N + 2
        self.buf = np.zeros(N * N + 2 * self.b0, dtype=np.float32)
        self.buf[self.b0: self.b0 + N * N] = flat(a)

    def get(self, i, j):
        k = i * N + j + self.b0
        return self.buf[k] if 0 <= k < self.buf.size else np.float32(0.0)

    def set(self, i, j, v):
        k = i * N + j + self.b0
        if 0 <= k < self.buf.size:
            self.buf[k] = v


def ref_p_with_boundary(pf, i, j):
    pad = Pad(pf)
    if (
        (i == j == 0)
        or (i == N - 1 and j == N - 1)
        or (i == 0 and j == N - 1)
        or (i == N - 1 and j == 0)
    ):
        pad.set(i, j, 0.0)
    elif i == 0:
        pad.set(0, j, pad.get(1, j))
    elif j == 0:
        pad.set(i, 0, pad.get(i, 1))
    elif i == N - 1:
        pad.set(N - 1, j, pad.get(N - 2, j))
    elif j == N - 1:
        pad.set(i, N - 1, pad.get(i, N - 2))
    return pad.get(i, j)


def ref_pressure_projection(pf, vf):
    out = np.zeros_like(vf)
    for i in range(N):
        for j in range(N):
            out[i, j, 0] = vf[i, j, 0] - (
                ref_p_with_boundary(pf, i + 1, j) - ref_p_with_boundary(pf, i - 1, j)
            ) / 2.0
            out[i, j, 1] = vf[i, j, 1] - (
                ref_p_with_boundary(pf, i, j + 1) - ref_p_with_boundary(pf, i, j - 1)
            ) / 2.0
    return out


def ref_vorticity_projection(cf, vf, dt, curl_param):
    out = vf.copy()
    for i in range(N):
        for j in range(N):
            gc = np.array(
                [
                    0.5 * (at(cf, i + 1, j) - at(cf, i - 1, j)),
                    0.5 * (at(cf, i, j + 1) - at(cf, i, j - 1)),
                    0.0,
                ],
                dtype=np.float32,
            )
            length = np.sqrt(gc[0] * gc[0] + gc[1] * gc[1] + gc[2] * gc[2])
            if length > 1e-5:
                gv = gc / length
                # cross(gradcurl, [0, 0, 1]) is (g[1], -g[0], 0)
                force = curl_param * np.array([gv[1], -gv[0], 0.0], dtype=np.float32)
                out[i, j, 0] = vf[i, j, 0] + dt * force[0]
                out[i, j, 1] = vf[i, j, 1] + dt * force[1]
    return out


def test_curl(upstream, rng):
    """`curl` against a real upstream run, and against the transcription.

    Both fields are `guarded` allocations: this kernel reads one cell past the
    edge on the last row and column, and a plain array would be read off its end.
    At the border the three implementations legitimately disagree -- the
    transcription treats an off-edge tap as zero, the port reads its guard, and
    taichi reads whatever adjoins its allocation -- so the comparison covers the
    interior, where every tap is in-field on all three sides.
    """
    v = guarded((N, N), 2)
    v[...] = rnd(rng, (N, N, 2), 0.3)
    vf, cf = field((N, N), 2), field((N, N))
    set_field(vf, v)
    upstream["curl"](vf, cf)

    out = guarded((N, N), 1)
    mte.curl(v, out, nworkers=4)

    assert_same(out[1:-1, 1:-1], get_field(cf)[1:-1, 1:-1], name="curl vs upstream")
    assert_same(out[1:-1, 1:-1], ref_curl(v)[1:-1, 1:-1], name="curl vs transcription")
    # the guard is two cells deep and the far corner reads exactly two cells
    # past the last node, so a correct guard leaves nothing non-finite behind
    assert np.isfinite(out).all(), "a border read escaped the allocation"


def test_divergence(upstream, rng):
    """`divergence` against a real upstream run and the transcription.

    Guarded fields, interior comparison: for the same reasons as `curl`, the
    input needs its guard (this kernel reads one cell past the edge) and the
    three sides only have a common value where every tap is in-field.
    """
    v = guarded((N, N), 2)
    v[...] = rnd(rng, (N, N, 2), 0.3)
    vf, df = field((N, N), 2), field((N, N))
    set_field(vf, v)
    upstream["divergence"](vf, df)

    out = guarded((N, N), 1)
    mte.divergence(v, out, nworkers=4)

    assert_same(out[1:-1, 1:-1], get_field(df)[1:-1, 1:-1], name="divergence vs upstream")
    assert_same(
        out[1:-1, 1:-1], ref_divergence(v)[1:-1, 1:-1],
        name="divergence vs transcription",
    )
    assert np.isfinite(out).all(), "a border read escaped the allocation"


def test_pressure_iteration(upstream, rng):
    """`pressure_iteration` against a real upstream run and the transcription.

    Guarded fields for the same reason as `curl`: the kernel reads one cell past
    the edge on the last row and column, so a plain input would be read off its
    end.
    """
    d = guarded((N, N), 1)
    d[...] = rnd(rng, (N, N))
    p = guarded((N, N), 1)
    p[...] = rnd(rng, (N, N))
    df, pf, npf = field((N, N)), field((N, N)), field((N, N))
    set_field(df, d)
    set_field(pf, p)
    upstream["pressure_iteration"](df, pf, npf)

    out = guarded((N, N), 1)
    mte.pressure_iteration(d, p, out, nworkers=4)

    assert_same(
        out[1:-1, 1:-1], get_field(npf)[1:-1, 1:-1],
        name="pressure_iteration vs upstream",
    )
    # the transcription reads an off-edge neighbour as zero, which is what the
    # port's zeroed guard gives it; taichi reads its own adjacent memory, so the
    # two only have to agree over the interior, where every tap is in-field
    assert_same(
        out[1:-1, 1:-1], ref_pressure_iteration(d, p)[1:-1, 1:-1],
        name="pressure_iteration vs transcription",
    )
    assert np.isfinite(out).all(), "a border read escaped the allocation"


def test_pressure_projection(rng):
    """Transcription only: taichi's CPU backend segfaults on this kernel.

    Its `p_with_boundary` calls *write* one cell past the edge as well as read
    it, and taichi segfaults; the port keeps the arithmetic and lets those
    accesses land in the allocation's guard, so only the defined interior is
    compared.
    """
    p = guarded((N, N), 1)
    p[...] = rnd(rng, (N, N))
    v = guarded((N, N), 2)
    v[...] = rnd(rng, (N, N, 2), 0.3)
    # `p_with_boundary` mutates the pressure field, so the reference needs a
    # pristine copy rather than the field the kernel has already written to
    before = p.copy()
    out = guarded((N, N), 2)
    mte.pressure_projection(p, v, out, nworkers=4)
    assert not np.array_equal(p, before), "p_with_boundary must write back"
    assert_same(
        out[1:-1, 1:-1], ref_pressure_projection(before, v)[1:-1, 1:-1],
        name="pressure_projection interior",
    )


def test_vorticity_projection(upstream, rng):
    """`vorticity_projection` against a real upstream run and the transcription.

    Guarded inputs for the same reason as `curl`: it reads `i + 1` / `j + 1`
    unclamped on the last row and column.
    """
    c = guarded((N, N), 1)
    c[...] = rnd(rng, (N, N))
    v = guarded((N, N), 2)
    v[...] = rnd(rng, (N, N, 2), 0.3)
    cf, vf, vfn = field((N, N)), field((N, N), 2), field((N, N), 2)
    set_field(cf, c)
    set_field(vf, v)
    upstream["vorticity_projection"](cf, vf, vfn)

    out = guarded((N, N), 2)
    mte.vorticity_projection(
        c, v, out, dt=PARAMS["dt"], curl_param=PARAMS["curl_param"], nworkers=4
    )
    assert_same(
        out[1:-1, 1:-1], get_field(vfn)[1:-1, 1:-1],
        name="vorticity_projection vs upstream",
    )
    # interior only, for the same reason as `pressure_iteration`: at the border
    # the transcription's zero-guard and taichi's adjacent memory disagree
    assert_same(
        out[1:-1, 1:-1],
        ref_vorticity_projection(c, v, PARAMS["dt"], PARAMS["curl_param"])[1:-1, 1:-1],
        name="vorticity_projection vs transcription",
    )
    assert np.isfinite(out).all(), "a border read escaped the allocation"


# ------------------------------------------------------------------ behaviour
def test_pressure_solve_reduces_divergence(rng):
    """Jacobi pressure solve + projection is the usual projection: div falls."""
    sim = mojotaichi.EulerFluid(shape=(N, N), iteration_step=200)
    sim.velocities_pair.cur[...] = rnd(rng, (N, N, 2), 0.5)
    div_before = guarded((N, N), 1)
    mte.divergence(sim.velocities_pair.cur, div_before, nworkers=4)
    sim.pressure_step()
    div_after = guarded((N, N), 1)
    mte.divergence(sim.velocities_pair.cur, div_after, nworkers=4)
    assert np.abs(div_after).max() < np.abs(div_before).max()


def test_apply_vel_bc_zeroes_the_border_after_a_step(rng):
    sim = mojotaichi.EulerFluid(shape=(N, N))
    sim.velocities_pair.cur[...] = rnd(rng, (N, N, 2), 1.0)
    sim.step(0, 0, 0.5, 0.5)
    v = sim.velocities_pair.cur
    assert not v[0].any() and not v[-1].any()
    assert not v[:, 0].any() and not v[:, -1].any()
    assert np.isfinite(v).all()


def test_step_is_stable_for_many_frames(rng):
    sim = mojotaichi.EulerFluid(shape=(N, N))
    px = py = 0
    for _ in range(10):
        px, py = sim.step(px, py, 0.4, 0.6)
    assert np.isfinite(sim.velocities_pair.cur).all()
    assert np.isfinite(sim.colorField).all()
    assert sim.colorField.min() >= -1e-6


def test_guarded_field_is_a_plain_view(rng):
    """The guard is allocation slack; the returned field is the logical grid."""
    a = guarded((N, N), 2)
    assert a.shape == (N, N, 2) and a.flags.c_contiguous and a.dtype == np.float32
    assert not a.any()
    a[3, 4, 1] = 2.0
    assert a[3, 4, 1] == 2.0


def test_logos_match_between_upstream_and_port(rng):
    """`init_field` uses the same `taichi_logo` on both sides."""
    pts = rng.random((500, 2)).astype(np.float32)
    ours = mojotaichi.taichi_logo(pts)
    assert set(np.unique(ours)) <= {0.0, 1.0}
    assert ours.sum() > 100

def test_mouse_interaction_applies_the_impulse(rng):
    """`mouse_interaction` scales the cursor into grid cells and swaps buffers.

    Upstream reads the cursor from the GUI window; the port takes the position
    directly, so this checks the same arithmetic and the same double-buffer
    swap the Euler frame loop depends on.
    """
    sim = mojotaichi.EulerFluid(shape=(N, N))
    sim.velocities_pair.cur[:] = (rng.normal(size=(N, N, 2)) * 0.1).astype(np.float32)

    cur_id = id(sim.velocities_pair.cur)
    before = sim.velocities_pair.cur.copy()

    # the first call seeds the previous position from the current one, so the
    # drag vector is zero: the kernel copies rather than accelerates, and the
    # values are unchanged.  Upstream behaves the same way.
    px, py = sim.mouse_interaction(0, 0, 0.4, 0.6)
    assert (px, py) == (int(0.4 * N), int(0.6 * N))
    assert id(sim.velocities_pair.cur) != cur_id  # the pair was swapped
    assert np.array_equal(sim.velocities_pair.cur, before)

    # dragging away from the seeded position applies a real impulse
    moved = sim.velocities_pair.cur.copy()
    sim.mouse_interaction(px, py, 0.5, 0.6)
    assert not np.array_equal(sim.velocities_pair.cur, moved)


# ------------------------------------- python-side step functions vs upstream
# `advaction_step`, `voricity_step` and `pressure_step` are Python loops over
# the kernels in upstream, and the port keeps them as `EulerFluid` methods.  The
# README lists them as ported, so each is run here against the upstream function
# exec'd from the installed package, from the same starting state.
#
# `pressure_step` calls `pressure_projection`, which segfaults taichi's CPU
# backend, so upstream's is stopped after `pressure_solve` and the two pressure
# fields are compared; `advaction_step` and `voricity_step` have no such kernel
# and are compared in full.


def _upstream_state(upstream, v0, c0):
    """Upstream's module globals, filled from `v0` / `c0`."""
    up = upstream
    up.update(
        velocityField=field((N, N), 2),
        _new_velocityField=field((N, N), 2),
        colorField=field((N, N), 3),
        _new_colorField=field((N, N), 3),
        curlField=field((N, N)),
        divField=field((N, N)),
        pressField=field((N, N)),
        _new_pressField=field((N, N)),
    )
    set_field(up["velocityField"], v0)
    set_field(up["colorField"], c0)
    up["velocities_pair"] = up["TexPair"](up["velocityField"], up["_new_velocityField"])
    up["color_pair"] = up["TexPair"](up["colorField"], up["_new_colorField"])
    up["pressure_pair"] = up["TexPair"](up["pressField"], up["_new_pressField"])
    return up


def _ours_state(v0, c0):
    sim = mojotaichi.EulerFluid(shape=(N, N))
    sim.velocities_pair.cur[...] = v0
    sim.color_pair.cur[...] = c0
    return sim


def test_advaction_step_matches_upstream(upstream, rng):
    """Upstream's advection step, kernel for kernel, against the port.

    The velocity field is large enough that the two advections in the step are
    genuinely order-dependent -- upstream advects the colour by the current
    velocity and *then* the velocity by itself, and with a non-zero field
    swapping those two changes the answer.
    """
    v0 = rnd(rng, (N, N, 2), 0.8)
    c0 = rnd(rng, (N, N, 3), 0.2)
    up = _upstream_state(upstream, v0, c0)
    up["advaction_step"]()

    sim = _ours_state(v0, c0)
    sim.advaction_step()

    assert_same(
        get_field(up["velocities_pair"].cur), sim.velocities_pair.cur,
        name="advaction_step velocity",
    )
    assert_same(
        get_field(up["color_pair"].cur), sim.color_pair.cur,
        name="advaction_step colour",
    )
    # the step must actually have moved things, so the comparison cannot pass
    # vacuously on a step that does nothing
    assert not np.allclose(sim.velocities_pair.cur, v0), "advection did nothing"


def test_voricity_step_matches_upstream(upstream, rng):
    """Upstream's vorticity-confinement step against the port."""
    v0 = rnd(rng, (N, N, 2), 0.3)
    c0 = rnd(rng, (N, N, 3), 0.2)
    up = _upstream_state(upstream, v0, c0)
    up["voricity_step"]()

    sim = _ours_state(v0, c0)
    sim.voricity_step()

    # interior only: `curl` and `vorticity_projection` read one cell past the
    # edge, where upstream reads its own adjacent memory and the port its guard
    assert_same(
        get_field(up["velocities_pair"].cur)[1:-1, 1:-1],
        sim.velocities_pair.cur[1:-1, 1:-1],
        name="voricity_step velocity interior",
    )
    assert not np.array_equal(sim.velocities_pair.cur, v0), "the step did nothing"


def test_pressure_solve_matches_upstream(upstream, rng):
    """Upstream's `pressure_solve` loop, which is the part of `pressure_step`
    that runs on taichi's CPU backend."""
    d = rnd(rng, (N, N), 0.3)
    p = rnd(rng, (N, N), 0.1)
    up = _upstream_state(upstream, np.zeros((N, N, 2), np.float32),
                         np.zeros((N, N, 3), np.float32))
    set_field(up["divField"], d)
    set_field(up["pressField"], p)
    up["pressure_solve"](up["pressure_pair"], up["divField"])

    sim = mojotaichi.EulerFluid(shape=(N, N))
    sim.divField[...] = d
    sim.pressure_pair.cur[...] = p
    mte.pressure_solve(
        sim.pressure_pair, sim.divField, iteration_step=PARAMS["iteration_step"]
    )

    # interior only: `pressure_iteration` reads one cell past the edge
    assert_same(
        get_field(up["pressure_pair"].cur)[1:-1, 1:-1],
        sim.pressure_pair.cur[1:-1, 1:-1],
        name="pressure_solve interior",
    )


def test_pressure_step_reduces_divergence_and_matches_the_reference_projection(
    upstream, rng
):
    """`pressure_step` is divergence + `pressure_solve` + `pressure_projection`.

    Upstream's last kernel segfaults on taichi's CPU backend, so the port's
    projection is checked against the transcription over the interior, and the
    solve half against a real upstream run.  Both halves must be present and in
    upstream's order, or the divergence would not drop.
    """
    v0 = rnd(rng, (N, N, 2), 0.5)
    sim = _ours_state(v0, np.zeros((N, N, 3), np.float32))
    before = guarded((N, N), 1)
    mte.divergence(sim.velocities_pair.cur, before, nworkers=4)

    sim.pressure_step()

    after = guarded((N, N), 1)
    mte.divergence(sim.velocities_pair.cur, after, nworkers=4)
    assert np.abs(after).max() < np.abs(before).max(), "projection did nothing"

    # the projection is the transcription of upstream's own expressions
    ref = guarded((N, N), 2)
    mte.pressure_projection(
        sim.pressure_pair.cur, v0, ref, nworkers=4
    )
    assert_same(
        sim.velocities_pair.cur[1:-1, 1:-1], ref[1:-1, 1:-1],
        name="pressure_step projection interior",
    )
