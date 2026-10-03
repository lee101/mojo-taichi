"""Memory safety and argument validation at the FFI boundary.

Every kernel here writes through a raw pointer built from a numpy address, so
the contracts these tests pin down are the difference between a clear Python
error and silent memory corruption:

  * a buffer that is not float32 / C-contiguous is rejected rather than
    silently converted, because a converted copy would receive the result and
    the caller's array would be left untouched;
  * a buffer of the wrong shape is rejected before the kernel indexes it;
  * MPM88's grid scatter and gather stay inside the allocation even when a
    particle has left the domain;
  * `taichi_logo` refuses a position that is not a 2-vector, which it would
    otherwise read past the end of.
"""

import numpy as np
import pytest

import mojotaichi
from mojotaichi import eulerfluid2d as mte
from mojotaichi import mpm88 as mtm
from mojotaichi import patterns
from mojotaichi import stable_fluid as mtsf
from mojotaichi._lib import guarded

N = 16


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _vec2(rng=None, n=N):
    a = np.zeros((n, n, 2), dtype=np.float32)
    if rng is not None:
        a[:] = rng.normal(size=a.shape).astype(np.float32)
    return a


# ------------------------------------------------- no silent narrowing / copies
def test_float64_output_is_rejected_not_silently_copied(rng):
    """A float64 output must raise, not quietly receive nothing.

    `f32()` used to convert, and the kernel wrote the result into the
    converted copy, so the caller's array came back untouched and unchanged.
    """
    v = _vec2(rng)
    out64 = np.zeros((N, N), dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        mtsf.divergence(v, out64)
    with pytest.raises(TypeError, match="float32"):
        mte.curl(v, out64)
    assert not out64.any()  # nothing was written to it


def test_non_contiguous_output_is_rejected(rng):
    """A strided view cannot be addressed as a flat row-major buffer."""
    v = _vec2(rng)
    strided = np.zeros((N * 2, N), dtype=np.float32)[::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(TypeError, match="contiguous"):
        mtsf.divergence(v, strided)


def test_non_array_input_is_rejected():
    with pytest.raises(TypeError, match="numpy array"):
        mtsf.divergence([[0.0] * N] * N, np.zeros((N, N), dtype=np.float32))


# ----------------------------------------------- component-width contracts
# A kernel fixes its component stride at compile time (2 or 3 for a vector
# field, 1 for a scalar one), so a field narrower than the stride it is indexed
# at is a buffer overrun.  `_grid` only compared the square extent every
# stable_fluid field shares, which left the trailing component axis unchecked:
# `advect` picked its kernel variant from `qf` alone, so a vec3 `qf` with a
# vec2 `new_qf` wrote a third component past the end of `new_qf`, and
# `apply_impulse` read and wrote three components of a field it accepted as a
# scalar.  Both corrupted the heap with no Python-level error.


class _Canary:
    """Fields carved out of one large allocation, with slack after the last.

    A regression that overruns writes into that slack, which is readable here --
    so the test fails with a number rather than only with a segfault.
    """

    def __init__(self, words: int = 500_000):
        self.buf = np.full(words, -777.0, dtype=np.float32)
        self._at = 1024

    def field(self, shape):
        count = int(np.prod(shape))
        off = self._at
        self._at += count + 4096  # slack after every field
        return np.frombuffer(
            self.buf, dtype=np.float32, count=count, offset=off * 4
        ).reshape(shape)

    def tail_touched(self, last_field, shape) -> int:
        """How many floats after `last_field` no longer hold the canary value."""
        count = int(np.prod(shape))
        end = self._at - 4096
        after = self.buf[end : end + 4096]
        return int((after != -777.0).sum())


def test_advect_rejects_a_new_qf_narrower_than_qf(rng):
    """`advect` writes `qf`'s width into `new_qf`; a narrower one overruns it."""
    v = _vec2(rng)
    q3 = np.zeros((N, N, 3), dtype=np.float32)
    nq2 = np.zeros((N, N, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        mtsf.advect(v, q3, nq2)
    assert not nq2.any()


def test_advect_does_not_write_past_a_correctly_sized_new_qf(rng):
    """The positive case, with the real output watched for overruns.

    `test_advect_rejects_...` proves the mismatch raises; this proves the
    accepted path still writes exactly `new_qf` and no further, which is the
    half that would be silently wrong if the check were written as a cast.
    """
    c = _Canary()
    v = _vec2(rng)
    q3 = c.field((N, N, 3))
    q3[:] = rng.normal(size=q3.shape)
    nq3 = c.field((N, N, 3))
    mtsf.advect(v, q3, nq3)
    assert nq3.any(), "the kernel wrote nothing"
    assert c.tail_touched(nq3, nq3.shape) == 0, "wrote past the end of new_qf"


def test_apply_impulse_rejects_a_dye_field_that_is_not_vec3(rng):
    """`apply_impulse` reads and writes three dye components per cell."""
    v = _vec2(rng)
    imp = mojotaichi.impulse(
        mouse_dir=(1.0, 0.0), mouse_pos=(8.0, 8.0), color=(1.0, 0.5, 0.2)
    )
    for bad in (
        np.zeros((N, N), dtype=np.float32),        # scalar: reads 3, has 1
        np.zeros((N, N, 2), dtype=np.float32),     # vec2: reads 3, has 2
        np.zeros((N, N, 4), dtype=np.float32),     # vec4: would read the wrong ones
    ):
        with pytest.raises(ValueError, match="3 components"):
            mtsf.apply_impulse(v, bad, imp)
        assert not bad.any()


def test_apply_impulse_does_not_write_past_a_vec3_dye_field(rng):
    """The accepted path still writes exactly the dye field it was given."""
    c = _Canary()
    v = c.field((N, N, 2))
    v[:] = rng.normal(size=v.shape)
    dye = c.field((N, N, 3))
    dye[:] = 0.25
    mtsf.apply_impulse(v, dye, mojotaichi.impulse(
        mouse_dir=(1.0, 0.0), mouse_pos=(8.0, 8.0), color=(1.0, 0.5, 0.2)
    ))
    assert dye.any()
    assert c.tail_touched(dye, dye.shape) == 0, "wrote past the end of dyef"


@pytest.mark.parametrize(
    "call",
    [
        lambda v, f: mtsf.divergence(v, f),
        lambda v, f: mtsf.vorticity(v, f),
        lambda v, f: mtsf.pressure_jacobi(f, f, f),
        lambda v, f: mtsf.subtract_gradient(v, f),
        lambda v, f: mtsf.enhance_vorticity(v, f),
        lambda v, f: mtsf.copy_divergence(f, f),
        lambda v, f: mtsf.apply_pressure(f, f),
    ],
)
def test_scalar_stencils_reject_a_vector_field(rng, call):
    """The scalar kernels index one float per cell.

    Handed a vec3 field they would stride over it and touch a third of what
    the caller thinks they touched, which is a wrong answer rather than a
    crash -- so it is rejected instead.
    """
    v = _vec2(rng)
    f = np.zeros((N, N, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="1 components"):
        call(v, f)


def test_vector_stencils_still_accept_their_documented_widths(rng):
    """The tightened checks must not reject the combinations upstream allows."""
    v = _vec2(rng)
    mtsf.divergence(v, np.zeros((N, N), np.float32))          # scalar output
    mtsf.vorticity(v, np.zeros((N, N), np.float32))           # scalar output
    mtsf.pressure_jacobi(*(np.zeros((N, N), np.float32) for _ in range(3)))
    mtsf.subtract_gradient(v, np.zeros((N, N), np.float32))   # scalar pressure
    mtsf.advect(v, np.zeros((N, N, 2), np.float32), np.zeros((N, N, 2), np.float32))
    mtsf.advect(v, np.zeros((N, N, 3), np.float32), np.zeros((N, N, 3), np.float32))


# ------------------------------------------------------------- shape contracts
def test_mismatched_field_shapes_are_rejected(rng):
    v = _vec2(rng)
    with pytest.raises(ValueError, match="shape"):
        mtsf.divergence(v, np.zeros((N // 2, N // 2), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        mte.curl(v, np.zeros((N // 2, N // 2), dtype=np.float32))
    with pytest.raises(ValueError, match="square"):
        mtsf.divergence(
            np.zeros((N, N + 1, 2), np.float32),
            np.zeros((N, N + 1), np.float32),
        )


def test_mpm88_rejects_particle_arrays_of_differing_length():
    """`n_particles` comes from `x`; a shorter `J` would be read past its end."""
    n = 4
    g = 8
    for bad in (np.zeros((n - 1,), np.float32), np.zeros((n - 1, 2), np.float32)):
        with pytest.raises(ValueError, match="shape"):
            mtm.substep(
                np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32),
                np.zeros((n, 2, 2), np.float32), bad,
                guarded((g, g), 2), guarded((g, g), 1), n_grid=g,
            )


def test_mpm88_rejects_a_grid_of_the_wrong_size():
    n, g = 4, 8
    with pytest.raises(ValueError, match="shape"):
        mtm.substep(
            np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32),
            np.zeros((n, 2, 2), np.float32), np.zeros((n,), np.float32),
            guarded((g + 1, g), 2), guarded((g, g), 1), n_grid=g,
        )


# ------------------------------------------------- MPM88 grid scatter / gather
@pytest.mark.parametrize("pos", [-0.9, -0.4, 1.4, 2.5])
def test_mpm88_escaped_particles_stay_inside_the_allocation(pos):
    """A particle outside [0, 1] must not scatter or gather off the grid.

    Upstream indexes `grid_v[base + offset]` with no clamp, so an escaped
    particle walks off the field.  The port clamps the node; a canary array
    allocated alongside catches any write that still escapes.
    """
    canary = np.full(100_000, -1234.0, dtype=np.float32)
    before = canary.copy()

    sim = mojotaichi.MPM88(n_particles=64, n_grid=32).init(seed=0)
    sim.x[:] = [pos, pos]
    sim.run(20)

    assert np.array_equal(canary, before), "wrote outside the grid allocation"
    assert np.isfinite(sim.x).all()
    assert np.isfinite(sim.v).all()
    assert np.isfinite(sim.grid_v).all()
    # the border node is where a clamped particle lands, not a wild address
    assert np.isfinite(sim.grid_m).all()


# ------------------------------------------------------------- taichi_logo input
@pytest.mark.parametrize("bad", [0.5, [0.5], np.zeros((1,), np.float32), np.zeros((3,), np.float32)])
def test_taichi_logo_rejects_a_non_2vector_position(bad):
    """The kernel reads two floats per point, so one float is an OOB read."""
    with pytest.raises(ValueError):
        patterns.taichi_logo(bad)


def test_taichi_logo_rejects_a_wrong_width_array():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        patterns.taichi_logo(np.zeros((10, 3), dtype=np.float32))


def test_taichi_logo_still_accepts_the_documented_forms():
    assert patterns.taichi_logo([0.5, 0.5]) == 1.0
    assert patterns.taichi_logo(np.array([0.5, 0.75], np.float32)) in (0.0, 1.0)
    pts = np.random.default_rng(0).random((64, 2)).astype(np.float32)
    out = patterns.taichi_logo(pts)
    assert out.shape == (64,) and set(np.unique(out)) <= {0.0, 1.0}
