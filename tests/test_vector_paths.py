"""Coverage for the vectorised inner loops and their scalar tails.

Every ported stencil sweeps a row in `VEC_W`-wide chunks and finishes whatever is
left over in a scalar loop, and takes a different path for the first and last
`VEC_W - 1` nodes of a row because upstream's `sample` clamps there.  Those two
pieces of code are the ones a width change or a refactor would break, and none
of them is visible from a single even grid size, so this module pins them down:

  * `VEC_W` is read from the compiled library rather than assumed, so a toolchain
    that reports a different width fails loudly here instead of silently
    skipping every vector body.
  * grid widths on both sides of a `VEC_W` boundary and a `VEC_W` boundary itself
    are checked for byte-exact agreement, which is what a tail that drops or
    double-writes a node would show up as.
  * the `sample`-clamp head and tail of a row is checked against the upstream
    `sample` transcription rather than against nothing.
  * a grid too narrow to hold one vector is checked, since that is the case where
    the vector loop never runs and only the scalar body can be right.
"""

import numpy as np
import pytest

from mojotaichi import stable_fluid as mtsf
from mojotaichi._lib import lib

DT = 0.03
DYE_DECAY = 1 - 1 / (60 * 2)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def vec_w() -> int:
    """The SIMD width the kernels were compiled with.

    `VEC_W` is `simd_width_of[DType.float32]()` in `taichi/vector.mojo`; this
    library exposes the same value so the tests can build widths around it
    instead of hard-coding 8 and quietly passing if that ever changes.
    """
    fn = getattr(lib(), "mt_simd_width", None)
    if fn is None:  # pragma: no cover - only on a build without the symbol
        pytest.skip("this build does not expose mt_simd_width")
    return int(fn())


def _fields(rng, n):
    v = (rng.normal(size=(n, n, 2)) * 0.3).astype(np.float32)
    p = (rng.normal(size=(n, n)) * 0.3).astype(np.float32)
    d = (rng.normal(size=(n, n)) * 0.3).astype(np.float32)
    q = (rng.normal(size=(n, n, 3)) * 0.3).astype(np.float32)
    return v, p, d, q


@pytest.mark.parametrize("n", [8, 9, 15, 16, 17, 31, 32, 33, 64, 65])
def test_stencils_cover_every_node_at_every_width(rng, vec_w, n):
    """Every node of a grid of width `n` is written, and matches a NumPy reference.

    The reference is a transcription of upstream's kernel with its own clamp, so a
    tail that skips a node, repeats one, or mis-clamps the boundary cannot pass.
    """
    rng = np.random.default_rng(n)
    _, p, d, _ = _fields(rng, n)

    def at(a, i, j):
        return a[min(max(i, 0), n - 1), min(max(j, 0), n - 1)]

    ref = np.empty((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            ref[i, j] = (at(p, i - 1, j) + at(p, i + 1, j)
                         + at(p, i, j - 1) + at(p, i, j + 1) - d[i, j]) * 0.25
    got = np.zeros((n, n), np.float32)
    mtsf.pressure_jacobi(p, got, d, nworkers=4)
    assert np.array_equal(got, ref), f"pressure_jacobi n={n}"

    # the two whole-field copies have no clamp at all: every node must be
    # written, and a sentinel in the output would show an unwritten one
    got = np.full((n, n), np.nan, np.float32)
    mtsf.copy_divergence(d, got, nworkers=4)
    assert np.array_equal(got, -d), f"copy_divergence n={n}"

    got = np.full((n, n), np.nan, np.float32)
    mtsf.apply_pressure(p, got, nworkers=4)
    assert np.array_equal(got, p), f"apply_pressure n={n}"


def test_vec2_stencils_agree_with_upstream_at_every_width(rng, vec_w):
    """The vec2 stencils are scalar by design, and stay correct at every width.

    `divergence`, `vorticity` and `subtract_gradient` read one component of an
    interleaved velocity field across consecutive columns, which is a stride-2
    load; there is no `VEC_W`-wide contiguous load that produces it and
    `SIMD.gather` is broken on this toolchain, so those kernels have no vector
    body at all.  This checks them over grid widths on both sides of a `VEC_W`
    boundary so that stays true.
    """
    for n in (vec_w * 2 - 1, vec_w * 2, vec_w * 2 + 1):
        rng2 = np.random.default_rng(n)
        v = (rng2.normal(size=(n, n, 2)) * 0.5).astype(np.float32)
        p = (rng2.normal(size=(n, n)) * 0.3).astype(np.float32)

        def at(i, j):
            return v[min(max(i, 0), n - 1), min(max(j, 0), n - 1)]

        ref = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(n):
                vl, vr = at(i - 1, j), at(i + 1, j)
                vb, vt = at(i, j - 1), at(i, j + 1)
                vc = at(i, j)
                if i == 0:
                    vl = (-vc[0], vl[1])
                if i == n - 1:
                    vr = (-vc[0], vr[1])
                if j == 0:
                    vb = (vb[0], -vc[1])
                if j == n - 1:
                    vt = (vt[0], -vc[1])
                ref[i, j] = (vr[0] - vl[0] + vt[1] - vb[1]) * 0.5
        got = np.zeros((n, n), np.float32)
        mtsf.divergence(v, got, nworkers=4)
        assert np.abs(got - ref).max() < 2e-6, f"divergence n={n}"

        refv = np.zeros((n, n), np.float32)
        for i in range(n):
            for j in range(n):
                refv[i, j] = (at(i + 1, j)[1] - at(i - 1, j)[1]
                              - at(i, j + 1)[0] + at(i, j - 1)[0]) * 0.5
        gotv = np.zeros((n, n), np.float32)
        mtsf.vorticity(v, gotv, nworkers=4)
        assert np.abs(gotv - refv).max() < 2e-6, f"vorticity n={n}"

        gotg = np.array(v)
        refg = np.array(v)
        for i in range(n):
            for j in range(n):
                def atp(i2, j2):
                    return p[min(max(i2, 0), n - 1), min(max(j2, 0), n - 1)]

                refg[i, j, 0] = v[i, j, 0] - 0.5 * (atp(i + 1, j) - atp(i - 1, j))
                refg[i, j, 1] = v[i, j, 1] - 0.5 * (atp(i, j + 1) - atp(i, j - 1))
        mtsf.subtract_gradient(gotg, p, nworkers=4)
        assert np.abs(gotg - refg).max() < 2e-6, f"subtract_gradient n={n}"


def test_clamped_stencil_head_and_tail_match_upstream_sample(rng, vec_w):
    """`pressure_jacobi`'s vector body and its two scalar heads agree.

    The vector body is only legal for columns `HEAD .. res - HEAD - VEC_W`,
    because upstream's `sample` clamps `j - 1` and `j + 1` to the grid.  This
    runs a grid where the vector body covers whole columns between the two
    scalar heads and checks the whole field against a transcription of upstream's
    kernel, so neither the head, the tail nor the interior can drift.
    """
    # wide enough that the vector body covers whole columns between the two
    # scalar heads: HEAD < n - HEAD - VEC_W
    n = vec_w * 4 + 1
    rng = np.random.default_rng(11)
    p = (rng.normal(size=(n, n)) * 0.3).astype(np.float32)
    d = (rng.normal(size=(n, n)) * 0.3).astype(np.float32)
    out = np.zeros((n, n), np.float32)
    mtsf.pressure_jacobi(p, out, d, nworkers=4)

    def at(i, j):
        return p[min(max(i, 0), n - 1), min(max(j, 0), n - 1)]

    ref = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            ref[i, j] = (
                at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - d[i, j]
            ) * 0.25

    scale = max(1.0, float(np.abs(ref).max()))
    err = np.abs(out - ref)
    assert err.max() < 2e-6 * scale, f"worst {err.max():.3e}"
    head = vec_w - 1
    for j in list(range(head)) + list(range(n - head, n)):
        assert err[:, j].max() < 2e-6 * scale, f"column {j} is scalar-path only"


def test_grid_narrower_than_one_vector_is_still_written(vec_w):
    """A grid narrower than `VEC_W` never enters a vector loop.

    That is the case where every node is tail, so it is checked on its own: a
    `while j + VEC_W <= res` guard that skipped the remainder would leave the
    output zero.
    """
    for n in range(1, vec_w + 1):
        d = np.arange(n * n, dtype=np.float32).reshape(n, n)
        out = np.zeros((n, n), np.float32)
        mtsf.copy_divergence(d, out, nworkers=4)
        assert np.array_equal(out, -d), f"copy_divergence n={n}"


def test_worker_count_does_not_change_the_result(rng, vec_w):
    """The fork-join driver and the plain loop run the same body.

    `PAR_MIN` sits past every grid the library accepts, so `nworkers` is
    currently a no-op; this is the test that says so out loud, and it keeps
    saying so if `PAR_MIN` is ever lowered to a reachable size.
    """
    for n in (vec_w * 2, vec_w * 2 + 1, 512):
        rng2 = np.random.default_rng(n)
        p = (rng2.normal(size=(n, n)) * 0.3).astype(np.float32)
        d = (rng2.normal(size=(n, n)) * 0.3).astype(np.float32)
        ser = np.zeros((n, n), np.float32)
        par = np.zeros((n, n), np.float32)
        mtsf.pressure_jacobi(p, ser, d, nworkers=1)
        mtsf.pressure_jacobi(p, par, d, nworkers=8)
        assert np.array_equal(ser, par), f"pressure_jacobi n={n}"

        v = (rng2.normal(size=(n, n, 2)) * 0.3).astype(np.float32)
        q = (rng2.normal(size=(n, n, 3)) * 0.3).astype(np.float32)
        a = np.zeros((n, n, 3), np.float32)
        b = np.zeros((n, n, 3), np.float32)
        mtsf.advect(v, q, a, dt=DT, dye_decay=DYE_DECAY, nworkers=1)
        mtsf.advect(v, q, b, dt=DT, dye_decay=DYE_DECAY, nworkers=8)
        assert np.array_equal(a, b), f"advect n={n}"


def test_pressure_jacobi_reaches_its_fixed_point_through_the_vector_path(vec_w):
    """A whole solve, so the tail's output feeds the next iteration's input."""
    n = vec_w * 2
    rng = np.random.default_rng(3)
    d = (rng.normal(size=(n, n)) * 0.3).astype(np.float32)
    p = np.zeros((n, n), np.float32)
    nxt = np.zeros((n, n), np.float32)
    for _ in range(3000):
        mtsf.pressure_jacobi(p, nxt, d, nworkers=4)
        p, nxt = nxt, p
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pl = p[np.clip(i - 1, 0, n - 1), np.clip(j, 0, n - 1)]
    pr = p[np.clip(i + 1, 0, n - 1), np.clip(j, 0, n - 1)]
    pb = p[np.clip(i, 0, n - 1), np.clip(j - 1, 0, n - 1)]
    pt = p[np.clip(i, 0, n - 1), np.clip(j + 1, 0, n - 1)]
    residual = (pl + pr + pb + pt - 4.0 * p) - d
    assert np.abs(residual).max() < 1e-2 * max(1.0, np.abs(d).max())