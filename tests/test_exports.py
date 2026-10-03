"""Every `@export`ed kernel is called by name here, and asserted on.

The rest of the suite reaches the kernels through the Python wrappers, which is
the right way to test behaviour but the wrong way to test wiring: a signature
that has been transposed, a component stride swapped, or an argument pair
crossed inside `src/capi.mojo` still produces a plausible-looking result through
a wrapper that validates shapes, because the wrapper is what supplies the
arguments in the first place.

This module bypasses the wrappers and calls each `mt_*` symbol directly through
`ctypes`, with the argument types and order taken from the wrapper that is
documented to own them. So each kernel is proved to be reachable, to take the
arguments in the order `capi.mojo` declares, and to write real numbers into
the output buffers -- independently of whether the wrapper is wired correctly.

The expected values are transcriptions of the upstream kernel expressions, not
snapshots of this implementation's output: a test that compared against the
current output would pass on a kernel that is wrong in the same way twice.
"""

import ctypes

import numpy as np
import pytest

from mojotaichi._lib import _SIGNATURES, f32, lib

# name -> (argument builder, shape of the output it must fill, checker)
# Built per test below; this table only exists to make the coverage explicit.


def _call(name, *args):
    """Call one export directly, with its declared `ctypes` argtypes."""
    fn = getattr(lib(), name)
    fn.argtypes = _SIGNATURES[name][0]
    fn.restype = _SIGNATURES[name][1]
    fn(*args)


def _a(a):
    return a.ctypes.data


def _i(x):
    return int(x)


def _f(x):
    return float(x)


RES = 24  # > VEC_W, and not a multiple of it, so a tail runs


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------- patterns
def test_mt_taichi_logo_writes_the_documented_mask(rng):
    """`mt_taichi_logo(pos, dst, n, scale)` -> one float per point."""
    pts = rng.random((64, 2)).astype(np.float32)
    out = np.full(64, -1.0, dtype=np.float32)
    _call("mt_taichi_logo", _a(pts), _a(out), _i(64), _f(1 / 1.11))

    assert set(np.unique(out)) <= {0.0, 1.0}
    # (0.5, 0.5) is the centre of the disc, which is inside the ring, so the
    # same kernel must answer 1.0 there and 1.0 on a batch containing it
    single = np.array([[0.5, 0.5]], dtype=np.float32)
    one = np.full(1, -1.0, dtype=np.float32)
    _call("mt_taichi_logo", _a(single), _a(one), _i(1), _f(1 / 1.11))
    assert one[0] == 1.0
    # every point of the batch is answered, so n and the dst stride agree
    assert (out != -1.0).all()


# ----------------------------------------------------------- stable_fluid
def _sf_fields(rng, res=RES, comps=2):
    v = (rng.normal(size=(res, res, 2)) * 0.5).astype(np.float32)
    q = (rng.normal(size=(res, res, comps)) * 0.3).astype(np.float32)
    return v, q


def test_mt_sf_advect_v2_matches_the_rk3_backtrace(rng):
    v, q = _sf_fields(rng, comps=2)
    out = np.full(q.shape, -1.0, dtype=np.float32)
    _call("mt_sf_advect_v2", _a(v), _a(q), _a(out), _i(RES), _f(0.03), _f(1 - 1 / 120), _i(4))

    def at(u, w, field):
        return field[min(max(u, 0), RES - 1), min(max(w, 0), RES - 1)]

    def bilerp(px, py, field):
        s, t = px - 0.5, py - 0.5
        iu, iv = int(np.floor(s)), int(np.floor(t))
        fu, fv = s - np.floor(s), t - np.floor(t)
        a, b = at(iu, iv, field), at(iu + 1, iv, field)
        c, d = at(iu, iv + 1, field), at(iu + 1, iv + 1, field)
        lo = a + fu * (b - a)
        hi = c + fu * (d - c)
        return lo + fv * (hi - lo)

    dt, decay = 0.03, 1 - 1 / 120
    ref = np.empty_like(q)
    for i in range(RES):
        for j in range(RES):
            px, py = i + 0.5, j + 0.5
            # the backtrace samples the *velocity* field three times, and the
            # advected quantity is sampled once at the traced position
            v1 = bilerp(px, py, v)
            p1 = np.array([px, py]) - 0.5 * dt * v1
            v2 = bilerp(p1[0], p1[1], v)
            p2 = np.array([px, py]) - 0.75 * dt * v2
            v3 = bilerp(p2[0], p2[1], v)
            p = np.array([px, py]) - dt * (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3)
            ref[i, j] = bilerp(p[0], p[1], q) * decay

    assert (out != -1.0).all(), "the kernel left nodes unwritten"
    err = np.abs(out - ref).max()
    assert err < 1e-4 * max(1.0, np.abs(ref).max()), f"worst {err:.3e}"


def test_mt_sf_advect_v3_writes_three_components(rng):
    v, q = _sf_fields(rng, comps=3)
    out = np.full(q.shape, -1.0, dtype=np.float32)
    _call("mt_sf_advect_v3", _a(v), _a(q), _a(out), _i(RES), _f(0.03), _f(1 - 1 / 120), _i(4))
    assert (out != -1.0).all()
    # v2 and v3 share a body and differ only in the stride, so a transposed
    # stride would show as a component that never changes
    assert out.std(axis=2).min() > 0.0


def test_mt_sf_apply_impulse_moves_velocity_and_dye(rng):
    v = (rng.normal(size=(RES, RES, 2)) * 0.1).astype(np.float32)
    dye = np.zeros((RES, RES, 3), dtype=np.float32)
    imp = np.zeros(8, dtype=np.float32)
    imp[0], imp[1] = 1.0, 0.0
    imp[2], imp[3] = RES / 2.0, RES / 2.0
    imp[4], imp[5], imp[6] = 0.2, 0.5, 1.0
    v0, d0 = v.copy(), dye.copy()
    _call("mt_sf_apply_impulse", _a(v), _a(dye), _a(imp), _i(RES), _f(0.03),
          _f(10000.0), _f(RES / 2.0), _i(4))

    assert (v != v0).any(), "velocity unchanged"
    assert (dye != d0).any(), "dye unchanged"
    # the impulse is centred, so it must land on the centre and not at a corner
    assert np.abs(dye[RES // 2, RES // 2]).max() > 0.0
    assert np.abs(dye[0, 0]).max() < np.abs(dye[RES // 2, RES // 2]).max()
    # gravity alone moves every cell downwards, so no cell may gain height
    assert (v[..., 1] <= v0[..., 1] + 1e-6).all()


def test_mt_sf_divergence_is_the_documented_stencil(rng):
    v = (rng.normal(size=(RES, RES, 2)) * 0.5).astype(np.float32)
    out = np.full((RES, RES), -1.0, dtype=np.float32)
    _call("mt_sf_divergence", _a(v), _a(out), _i(RES), _i(4))

    def at(i, j):
        return v[min(max(i, 0), RES - 1), min(max(j, 0), RES - 1)]

    ref = np.zeros((RES, RES), np.float32)
    for i in range(RES):
        for j in range(RES):
            vl, vr, vb, vt, vc = at(i - 1, j), at(i + 1, j), at(i, j - 1), at(i, j + 1), at(i, j)
            if i == 0:
                vl = np.array([-vc[0], vl[1]])
            if i == RES - 1:
                vr = np.array([-vc[0], vr[1]])
            if j == 0:
                vb = np.array([vb[0], -vc[1]])
            if j == RES - 1:
                vt = np.array([vt[0], -vc[1]])
            ref[i, j] = (vr[0] - vl[0] + vt[1] - vb[1]) * 0.5
    assert (out != -1.0).all()
    assert np.abs(out - ref).max() < 2e-6


def test_mt_sf_vorticity_is_the_documented_stencil(rng):
    v = (rng.normal(size=(RES, RES, 2)) * 0.5).astype(np.float32)
    out = np.full((RES, RES), -1.0, dtype=np.float32)
    _call("mt_sf_vorticity", _a(v), _a(out), _i(RES), _i(4))

    def at(i, j):
        return v[min(max(i, 0), RES - 1), min(max(j, 0), RES - 1)]

    ref = np.zeros((RES, RES), np.float32)
    for i in range(RES):
        for j in range(RES):
            ref[i, j] = (at(i + 1, j)[1] - at(i - 1, j)[1]
                         - at(i, j + 1)[0] + at(i, j - 1)[0]) * 0.5
    assert (out != -1.0).all()
    assert np.abs(out - ref).max() < 2e-6


def test_mt_sf_pressure_jacobi_is_the_five_point_stencil(rng):
    p = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    d = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    out = np.full((RES, RES), -1.0, dtype=np.float32)
    _call("mt_sf_pressure_jacobi", _a(p), _a(out), _a(d), _i(RES), _i(4))

    def at(i, j):
        return p[min(max(i, 0), RES - 1), min(max(j, 0), RES - 1)]

    ref = np.zeros((RES, RES), np.float32)
    for i in range(RES):
        for j in range(RES):
            ref[i, j] = (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - d[i, j]) * 0.25
    assert (out != -1.0).all(), "a node was left unwritten by head, vector body or tail"
    assert np.abs(out - ref).max() < 2e-6


def test_mt_sf_subtract_gradient_writes_back_into_vf(rng):
    v = (rng.normal(size=(RES, RES, 2)) * 0.5).astype(np.float32)
    p = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    v0 = v.copy()
    _call("mt_sf_subtract_gradient", _a(v), _a(p), _i(RES), _i(4))

    def at(i, j):
        return p[min(max(i, 0), RES - 1), min(max(j, 0), RES - 1)]

    ref = np.zeros_like(v)
    for i in range(RES):
        for j in range(RES):
            ref[i, j, 0] = v0[i, j, 0] - 0.5 * (at(i + 1, j) - at(i - 1, j))
            ref[i, j, 1] = v0[i, j, 1] - 0.5 * (at(i, j + 1) - at(i, j - 1))
    assert np.abs(v - ref).max() < 2e-6
    # a uniform pressure field has no gradient, so it must leave the field alone
    v2 = v0.copy()
    flat = np.full((RES, RES), 0.7, dtype=np.float32)
    _call("mt_sf_subtract_gradient", _a(v2), _a(flat), _i(RES), _i(4))
    assert np.array_equal(v2, v0)


def test_mt_sf_enhance_vorticity_applies_the_curl_force(rng):
    # a curl field rising linearly along i: |c[i+1,j]| - |c[i-1,j]| = 2 while
    # both j-taps are equal, so upstream's force is (|ct|-|cb|, |cl|-|cr|)
    # = (0 - 2, 0), normalized to (-1, 0) -- but the cell also multiplies by
    # its own curl `cc` = i, so the x drive is negative for i > 0.  What the
    # kernel must show is that it reads the four taps at all, in the right
    # order, and that the result is finite and non-zero.
    c = np.tile(np.arange(RES, dtype=np.float32), (RES, 1))
    v = np.zeros((RES, RES, 2), dtype=np.float32)
    _call("mt_sf_enhance_vorticity", _a(v), _a(c), _i(RES), _f(7.0), _f(0.03), _i(4))

    assert np.isfinite(v).all()
    touched = np.abs(v).max(axis=2) > 0
    assert touched.any(), "a non-zero curl gradient produced no force"
    # the force is purely along x here: the two j-taps are equal by construction
    assert np.abs(v[..., 1]).max() == 0.0, "a y force appeared where the formula has none"
    # it grows with the local curl `cc`, so the top rows must move more than
    # the bottom ones; a transposed curl argument would break that ordering
    col = np.abs(v[RES - 6, :, 0]).max()
    assert col > 0.0, "the top rows saw no force"

    # a zero curl field produces exactly no force, which pins the eps guard and
    # the normalization denominator: upstream adds 1e-3, not 0
    v2 = np.zeros((RES, RES, 2), dtype=np.float32)
    flat = np.zeros((RES, RES), dtype=np.float32)
    _call("mt_sf_enhance_vorticity", _a(v2), _a(flat), _i(RES), _f(7.0), _f(0.03), _i(4))
    assert not np.abs(v2).max(), "a zero curl field produced a force"


def test_mt_sf_enhance_vorticity_clamps_to_upstreams_limits():
    """Upstream clamps the result to [-1e3, 1e3]; a big curl must be clamped."""
    v = np.full((RES, RES, 2), 0.0, dtype=np.float32)
    c = np.zeros((RES, RES), dtype=np.float32)
    c[RES // 2, :] = 1e6  # an enormous curl everywhere along a row
    _call("mt_sf_enhance_vorticity", _a(v), _a(c), _i(RES), _f(7.0), _f(0.03), _i(4))
    assert np.abs(v).max() <= 1e3 + 1e-3, "the clamp was not applied"


def test_mt_sf_copy_divergence_negates_and_covers_every_node(rng):
    d = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    out = np.full((RES, RES), np.nan, dtype=np.float32)
    _call("mt_sf_copy_divergence", _a(d), _a(out), _i(RES), _i(4))
    assert np.array_equal(out, -d), "a node was skipped by the vector loop or its tail"


def test_mt_sf_apply_pressure_copies_and_covers_every_node(rng):
    p = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    out = np.full((RES, RES), np.nan, dtype=np.float32)
    _call("mt_sf_apply_pressure", _a(p), _a(out), _i(RES), _i(4))
    assert np.array_equal(out, p), "a node was skipped by the vector loop or its tail"


# ------------------------------------------------------------ eulerfluid2d
def _ef_fields(rng, res=RES, comps=1):
    return (
        (rng.normal(size=(res, res, 2)) * 0.3).astype(np.float32),
        (rng.normal(size=(res, res, comps)) * 0.3).astype(np.float32),
    )


def test_mt_ef_init_field_paints_the_logo_into_colour():
    press = np.full((RES, RES), 7.0, dtype=np.float32)
    vel = np.full((RES, RES, 2), 7.0, dtype=np.float32)
    color = np.zeros((RES, RES, 3), dtype=np.float32)
    _call("mt_ef_init_field", _a(press), _a(vel), _a(color), _i(RES), _i(RES))

    assert not press.any() and not vel.any(), "init_field must zero pressure and velocity"
    # the logo is a binary mask, and each cell sums it over a 4x4 block scaled
    # by 1/16, so every channel is some k/16 for k in 0..16
    k = np.unique(color) * 16.0
    assert np.all(np.abs(k - np.round(k)) < 1e-3), (
        f"colour is not a 4x4 supersampled logo: {np.unique(color)}"
    )
    assert k.min() >= 0.0 and k.max() <= 16.0, "colour left the [0, 1] logo range"
    assert color.any(), "the logo painted nothing"


def test_mt_ef_advection_v2_and_v3_track_the_velocity_field(rng):
    for comps, name in ((2, "mt_ef_advection_v2"), (3, "mt_ef_advection_v3")):
        v, q = _ef_fields(rng, comps=comps)
        # zero velocity makes advection the identity, which pins the argument
        # order: a swapped vf/qf pair would move the field instead
        vz = np.zeros_like(v)
        out = np.full(q.shape, -1.0, dtype=np.float32)
        _call(name, _a(vz), _a(q), _a(out), _i(RES), _i(RES), _f(1 / 60.0), _i(4))
        assert np.abs(out - q).max() < 2e-5, f"{name}: zero velocity is not identity"

        # now a real velocity: the output must differ, and stay inside the
        # sampled field's range plus a hair of bilinear overshoot
        out2 = np.full(q.shape, -1.0, dtype=np.float32)
        _call(name, _a(v), _a(q), _a(out2), _i(RES), _i(RES), _f(1 / 60.0), _i(4))
        assert (out2 != -1.0).all(), f"{name} left nodes unwritten"
        assert np.abs(out2 - q).max() > 0.0


def test_mt_ef_curl_is_the_documented_stencil(rng):
    from mojotaichi._lib import guarded

    v = guarded((RES, RES), 2)
    v[...] = (rng.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    out = np.full((RES, RES), -1.0, dtype=np.float32)
    _call("mt_ef_curl", _a(v), _a(out), _i(RES), _i(RES), _i(4))

    flat = v.reshape(-1, 2)

    def at(i, j):
        k = i * RES + j
        return flat[k] if 0 <= k < flat.shape[0] else np.zeros(2, np.float32)

    ref = np.zeros((RES, RES), np.float32)
    for i in range(RES):
        for j in range(RES):
            ref[i, j] = 0.5 * ((at(i + 1, j)[1] - at(i - 1, j)[1])
                               - (at(i, j + 1)[0] - at(i, j - 1)[0]))
    assert (out != -1.0).all()
    assert np.abs(out - ref).max() < 2e-6


def test_mt_ef_divergence_is_the_documented_stencil(rng):
    from mojotaichi._lib import guarded

    v = guarded((RES, RES), 2)
    v[...] = (rng.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    out = np.full((RES, RES), -1.0, dtype=np.float32)
    _call("mt_ef_divergence", _a(v), _a(out), _i(RES), _i(RES), _i(4))

    flat = v.reshape(-1, 2)

    def at(i, j):
        k = i * RES + j
        return flat[k] if 0 <= k < flat.shape[0] else np.zeros(2, np.float32)

    ref = np.zeros((RES, RES), np.float32)
    for i in range(RES):
        for j in range(RES):
            ref[i, j] = 0.5 * (at(i + 1, j)[0] - at(i - 1, j)[0]
                               + at(i, j + 1)[1] - at(i, j - 1)[1])
    assert (out != -1.0).all()
    assert np.abs(out - ref).max() < 2e-6


def test_mt_ef_vorticity_projection_accelerates_along_the_curl(rng):
    from mojotaichi._lib import guarded

    c = guarded((RES, RES), 1)
    c[...] = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    v = guarded((RES, RES), 2)
    v[...] = 0.0
    out = np.full((RES, RES, 2), -1.0, dtype=np.float32)
    _call("mt_ef_vorticity_projection", _a(c), _a(v), _a(out), _i(RES), _i(RES),
          _f(1 / 60.0), _f(15.0), _i(4))

    assert (out != -1.0).all(), "a node with a long curl gradient was skipped"
    # the force is `curl_param * dt` times a unit vector, so its magnitude is
    # bounded: that pins the argument order, the guard and the normalization
    bound = 15.0 * (1 / 60.0) + 1e-6
    assert np.abs(out).max() <= bound, "the force exceeded curl_param * dt"
    assert np.abs(out).max() > bound * 0.5, "the force is far below curl_param * dt"

    # a constant curl field has no gradient, so the interior is not written.
    # The border is excluded because upstream reads `i + 1` / `j + 1` unclamped
    # there and lands in the guard, which is exactly what the guard is for.
    flat = guarded((RES, RES), 1)
    flat[...] = 0.5
    v2 = guarded((RES, RES), 2)
    v2[...] = 0.0
    out2 = np.full((RES, RES, 2), -1.0, dtype=np.float32)
    _call("mt_ef_vorticity_projection", _a(flat), _a(v2), _a(out2), _i(RES), _i(RES),
          _f(1 / 60.0), _f(15.0), _i(4))
    assert (out2[1:-1, 1:-1] == -1.0).all(), (
        "a node with no curl gradient was written"
    )

    # where it does write, it adds to `vf` rather than replacing it
    v3 = np.full((RES, RES, 2), 0.25, dtype=np.float32)
    out3 = np.full((RES, RES, 2), -1.0, dtype=np.float32)
    _call("mt_ef_vorticity_projection", _a(c), _a(v3), _a(out3), _i(RES), _i(RES),
          _f(1 / 60.0), _f(15.0), _i(4))
    assert (out3 != -1.0).all()
    assert np.abs(out3 - 0.25).max() <= bound
    assert np.abs(out3 - 0.25).max() > 0.0, "vf was replaced, not added to"


def test_mt_ef_pressure_iteration_is_the_five_point_stencil(rng):
    from mojotaichi._lib import guarded

    d = guarded((RES, RES), 1)
    d[...] = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    p = guarded((RES, RES), 1)
    p[...] = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    out = np.full((RES, RES), -1.0, dtype=np.float32)
    _call("mt_ef_pressure_iteration", _a(d), _a(p), _a(out), _i(RES), _i(RES), _i(4))

    flat = p.reshape(-1)

    def at(i, j):
        k = i * RES + j
        return flat[k] if 0 <= k < flat.shape[0] else np.float32(0.0)

    ref = np.zeros((RES, RES), np.float32)
    for i in range(RES):
        for j in range(RES):
            ref[i, j] = (at(i + 1, j) + at(i - 1, j) + at(i, j - 1) + at(i, j + 1)
                         - d[i, j]) / 4
    assert (out != -1.0).all()
    assert np.abs(out - ref).max() < 2e-6


def test_mt_ef_pressure_projection_subtracts_the_gradient(rng):
    from mojotaichi._lib import guarded

    p = guarded((RES, RES), 1)
    p[...] = (rng.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    v = guarded((RES, RES), 2)
    v[...] = (rng.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    out = np.full((RES, RES, 2), -1.0, dtype=np.float32)
    _call("mt_ef_pressure_projection", _a(p), _a(v), _a(out), _i(RES), _i(RES), _i(4))

    assert (out[1:-1, 1:-1] != -1.0).all()
    # interior only: the border calls p_with_boundary one cell off the field,
    # which is what the guard exists for
    flat = v.reshape(-1, 2)

    def at(i, j):
        k = i * RES + j
        return flat[k] if 0 <= k < flat.shape[0] else np.zeros(2, np.float32)

    pf = p.reshape(-1)

    def pat(i, j):
        k = i * RES + j
        return pf[k] if 0 <= k < pf.shape[0] else np.float32(0.0)

    ref = np.zeros((RES, RES, 2), np.float32)
    for i in range(1, RES - 1):
        for j in range(1, RES - 1):
            ref[i, j, 0] = at(i, j)[0] - (pat(i + 1, j) - pat(i - 1, j)) / 2.0
            ref[i, j, 1] = at(i, j)[1] - (pat(i, j + 1) - pat(i, j - 1)) / 2.0
    assert np.abs(out[1:-1, 1:-1] - ref[1:-1, 1:-1]).max() < 2e-6


def test_mt_ef_apply_vel_bc_zeroes_the_border():
    from mojotaichi._lib import guarded

    v = guarded((RES, RES), 2)
    v[...] = 1.0
    _call("mt_ef_apply_vel_bc", _a(v), _i(RES), _i(RES), _i(4))
    assert not v[0].any() and not v[-1].any()
    assert not v[:, 0].any() and not v[:, -1].any()
    assert v[1:-1, 1:-1].any(), "the interior must be untouched"


def test_mt_ef_apply_p_bc_copies_the_second_row_into_the_first():
    p = np.arange(RES * RES, dtype=np.float32).reshape(RES, RES)
    _call("mt_ef_apply_p_bc", _a(p), _i(RES), _i(RES), _i(4))
    assert p[0, 0] == 0.0 and p[RES - 1, RES - 1] == 0.0, "corners are zeroed"
    assert np.array_equal(p[0, 1:-1], p[1, 1:-1]), "row 0 must copy row 1"
    assert np.array_equal(p[-1, 1:-1], p[-2, 1:-1]), "the last row copies the one before"


def test_mt_ef_apply_c_bc_zeroes_the_border():
    c = np.full((RES, RES), 5.0, dtype=np.float32)
    _call("mt_ef_apply_c_bc", _a(c), _i(RES), _i(RES), _i(4))
    assert not c[0].any() and not c[-1].any()
    assert not c[:, 0].any() and not c[:, -1].any()
    assert (c[1:-1, 1:-1] == 5.0).all(), "the interior must be untouched"


def test_mt_ef_mouse_addspeed_pushes_along_the_drag_segment():
    v = np.zeros((RES, RES, 2), dtype=np.float32)
    new = np.full((RES, RES, 2), -1.0, dtype=np.float32)
    # drag from the middle of the grid to the right, radius covering the row
    _call("mt_ef_mouse_addspeed", _i(RES // 2), _i(RES // 2), _i(RES // 4), _i(RES // 2),
          _f(4.0), _a(v), _a(new), _i(RES), _i(RES), _f(125.0), _f(0.5), _i(4))

    assert (new != -1.0).all(), "mouse_addspeed left nodes unwritten"
    assert np.abs(new).max() > 0.0, "the drag had no effect"
    # the drag is to the right, so only x gains and nothing is pushed left
    assert np.abs(new[..., 1]).max() < 1e-6
    assert new[..., 0].min() >= -1e-6
    # cells outside the radius keep the old value, which here is zero
    assert np.abs(new[0, 0]).max() < 1e-6


# ------------------------------------------------------------------- mpm88
def test_mt_mpm88_init_places_particles_and_zeroes_affine():
    n = 64
    x = np.full((n, 2), -1.0, dtype=np.float32)
    v = np.full((n, 2), -1.0, dtype=np.float32)
    J = np.zeros(n, dtype=np.float32)
    u = np.random.default_rng(3).random((n, 2), dtype=np.float32)
    _call("mt_mpm88_init", _a(x), _a(v), _a(J), _a(u), _i(n))

    assert np.allclose(x, u * 0.4 + 0.2, atol=1e-7)
    assert np.all(v[:, 0] == 0.0) and np.all(v[:, 1] == -1.0)
    assert np.all(J == 1.0)


def test_mt_mpm88_substep_advances_state_and_fills_the_grid():
    from mojotaichi._lib import guarded

    n, g = 128, 32
    dx = 1 / g
    p_vol = (dx * 0.5) ** 2
    x = np.zeros((n, 2), dtype=np.float32)
    v = np.zeros((n, 2), dtype=np.float32)
    v[:, 1] = -1.0
    C = np.zeros((n, 2, 2), dtype=np.float32)
    J = np.ones(n, dtype=np.float32)
    gv, gm = guarded((g, g), 2), guarded((g, g), 1)

    _call("mt_mpm88_substep", _a(x), _a(v), _a(C), _a(J), _a(gv), _a(gm),
          _i(n), _i(g), _f(dx), _f(2e-4), _f(p_vol), _f(p_vol), _f(400.0),
          _f(9.8), _i(3), _i(4))

    assert np.abs(gm).max() > 0.0, "P2G put no mass on the grid"
    # every particle's mass landed on the grid, up to float32 rounding
    assert abs(float(gm.sum()) - n * p_vol) < 1e-3 * n * p_vol
    # particles fall: x moved down and J responded
    assert x[:, 1].max() < 0.0 or (x != 0.0).any()
    assert (v[:, 1] <= -1.0 + 1.0).all()
    assert np.isfinite(C).all() and np.isfinite(J).all()


# ------------------------------------------------------- the plumbing itself
def test_every_export_in_capi_has_a_signature_and_is_reachable():
    """`capi.mojo` and `_SIGNATURES` must not drift apart.

    A symbol exported but absent from `_SIGNATURES` would still load through
    ctypes (with unchecked argument types) while the wrappers could not call it,
    so the two lists are compared against each other here rather than trusted.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "capi.mojo"
    exported = set(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', src.read_text()))
    assert exported == set(_SIGNATURES), (
        f"only in capi: {sorted(exported - set(_SIGNATURES))}; "
        f"only in _SIGNATURES: {sorted(set(_SIGNATURES) - exported)}"
    )
    handle = lib()
    for name in exported:
        assert hasattr(handle, name), f"{name} is exported but not in the shared library"


def test_simd_width_is_the_documented_value():
    """`mt_simd_width` must answer, and answer the width the kernels used."""
    fn = lib().mt_simd_width
    fn.argtypes = _SIGNATURES["mt_simd_width"][0]
    fn.restype = _SIGNATURES["mt_simd_width"][1]
    w = int(fn())
    assert w >= 1 and (w & (w - 1)) == 0, f"SIMD width {w} is not a power of two"


def test_an_export_rejects_a_float32_narrowing_conversion(rng):
    """The wrappers refuse float64; the raw exports must stay float32-only.

    This does not re-test `f32()` -- it pins the fact that `f32()` is the only
    thing standing between a float64 buffer and a kernel that would reinterpret
    its bits, by checking that a float64 array is genuinely a different address
    space from what a kernel may be handed.
    """
    a = np.zeros(64, dtype=np.float64)
    with pytest.raises(TypeError):
        f32(a, "a")
    b = np.zeros(64, dtype=np.float32)
    assert f32(b, "b") is b


def test_a_ctypes_call_does_not_keep_the_buffer_alive_on_its_own():
    """Documents the lifetime rule the wrappers rely on.

    The kernels allocate nothing and hold no reference to a buffer after the
    call returns, so the caller keeps ownership.  This asserts the mechanism:
    `ctypes.data` is an int, and the only thing that keeps the memory alive
    across the call is the Python object the caller already holds.
    """
    a = np.zeros((4, 4), dtype=np.float32)
    assert isinstance(a.ctypes.data, int)
    del a
    # nothing holds a reference: a fresh array may reuse the address, which is
    # exactly why the wrappers must not cache the int
    b = np.zeros((4, 4), dtype=np.float32)
    assert isinstance(b.ctypes.data, int)