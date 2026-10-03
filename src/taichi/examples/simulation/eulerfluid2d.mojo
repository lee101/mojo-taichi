"""Mojo port of `taichi/examples/simulation/eulerfluid2d.py`.

2D Euler fluid: semi-Lagrangian advection, vorticity confinement, Jacobi
pressure solve.  Same names, same order, same arithmetic as upstream.

Forced divergences, documented in the README:

  * `ti.template()` -> one function per field rank (`_v2` / `_v3`).
  * `eulerSimParam` is a Python dict upstream; here each value is an argument.
  * `shape` becomes `n0`, `n1` (field extents) so a non-square field works.
  * `curl`, `divergence`, `pressure_iteration` and `pressure_projection` index
    `i + 1` / `j + 1` without clamping, so on the last row/column they read and
    write out of bounds.  The arithmetic is kept verbatim; the Python wrapper
    allocates a one-cell guard so those accesses stay inside the allocation.
"""

from std.math import sqrt

from max.algorithm import parallelize
from taichi.vector import VEC_W, nodes_per_worker

from taichi.examples.patterns import taichi_logo

comptime PF = Pointer[Float32, AnyOrigin[mut=True]]
comptime V2 = SIMD[DType.float32, 2]
# Mojo SIMD widths must be powers of two, so 3-vectors live in a 4-lane
# vector with a permanently zero fourth lane that is never read.
comptime V3 = SIMD[DType.float32, 4]
comptime FW = SIMD[DType.float32, VEC_W]



# #####################
#   Bilinear Interpolation function
# #####################
def sample(vf: PF, u: Float32, v: Float32, n0: Int, n1: Int) -> Float32:
    # Nearest
    var i = max(0, min(n0 - 1, Int(u)))
    var j = max(0, min(n1 - 1, Int(v)))
    return vf.unsafe_load(i * n1 + j)


def sample_v2(vf: PF, u: Float32, v: Float32, n0: Int, n1: Int) -> V2:
    var i = max(0, min(n0 - 1, Int(u)))
    var j = max(0, min(n1 - 1, Int(v)))
    var o = (i * n1 + j) * 2
    return V2(vf.unsafe_load(o), vf.unsafe_load(o + 1))


def sample_v3(vf: PF, u: Float32, v: Float32, n0: Int, n1: Int) -> V3:
    var i = max(0, min(n0 - 1, Int(u)))
    var j = max(0, min(n1 - 1, Int(v)))
    var o = (i * n1 + j) * 3
    return V3(vf.unsafe_load(o), vf.unsafe_load(o + 1), vf.unsafe_load(o + 2), 0.0)


def lerp_v2(vl: V2, vr: V2, frac: Float32) -> V2:
    # frac: [0.0, 1.0]
    return (1.0 - frac) * vl + frac * vr


def lerp_v3(vl: V3, vr: V3, frac: Float32) -> V3:
    return (1.0 - frac) * vl + frac * vr


def bilerp_v2(vf: PF, u: Float32, v: Float32, n0: Int, n1: Int) -> V2:
    # use -0.5 to decide where bilerp performs in cells
    var s = u - 0.5
    var t = v - 0.5
    var iu = Int(s)
    var iv = Int(t)
    var a = sample_v2(vf, Float32(iu) + 0.5, Float32(iv) + 0.5, n0, n1)
    var b = sample_v2(vf, Float32(iu + 1) + 0.5, Float32(iv) + 0.5, n0, n1)
    var c = sample_v2(vf, Float32(iu) + 0.5, Float32(iv + 1) + 0.5, n0, n1)
    var d = sample_v2(vf, Float32(iu + 1) + 0.5, Float32(iv + 1) + 0.5, n0, n1)
    # fract
    var fu = s - Float32(iu)
    var fv = t - Float32(iv)
    return lerp_v2(lerp_v2(a, b, fu), lerp_v2(c, d, fu), fv)


def bilerp_v3(vf: PF, u: Float32, v: Float32, n0: Int, n1: Int) -> V3:
    var s = u - 0.5
    var t = v - 0.5
    var iu = Int(s)
    var iv = Int(t)
    var a = sample_v3(vf, Float32(iu) + 0.5, Float32(iv) + 0.5, n0, n1)
    var b = sample_v3(vf, Float32(iu + 1) + 0.5, Float32(iv) + 0.5, n0, n1)
    var c = sample_v3(vf, Float32(iu) + 0.5, Float32(iv + 1) + 0.5, n0, n1)
    var d = sample_v3(vf, Float32(iu + 1) + 0.5, Float32(iv + 1) + 0.5, n0, n1)
    var fu = s - Float32(iu)
    var fv = t - Float32(iv)
    return lerp_v3(lerp_v3(a, b, fu), lerp_v3(c, d, fu), fv)


# ------------------------------------------------------------------- ti.kernel
def init_field(pressField: PF, velocityField: PF, colorField: PF, n0: Int, n1: Int):
    # init pressure and velocity fieldfield
    for k in range(n0 * n1):
        pressField.unsafe_store(k, 0.0)
    for k in range(n0 * n1 * 2):
        velocityField.unsafe_store(k, 0.0)
    for i in range(n0 * 4):
        for j in range(n1 * 4):
            # 4x4 super sampling:
            # upstream divides both components by shape[0] * 4
            var ret = taichi_logo(
                V2(Float32(i) / Float32(n0 * 4), Float32(j) / Float32(n0 * 4))
            )
            var o = ((i // 4) * n1 + (j // 4)) * 3
            colorField.unsafe_store(o, colorField.unsafe_load(o) + ret / 16.0)
            colorField.unsafe_store(o + 1, colorField.unsafe_load(o + 1) + ret / 16.0)
            colorField.unsafe_store(o + 2, colorField.unsafe_load(o + 2) + ret / 16.0)


def advection_v2(
    vf: PF, qf: PF, new_qf: PF, n0: Int, n1: Int, dt: Float32, nworkers: Int
):
    def one(i: Int, j: Int) {imm}:
        var k = i * n1 + j
        var coord_cur = V2(Float32(i), Float32(j)) + V2(0.5, 0.5)
        var vel_cur = V2(vf.unsafe_load(k * 2), vf.unsafe_load(k * 2 + 1))
        var coord_prev = coord_cur - vel_cur * dt
        var q_prev = bilerp_v2(qf, coord_prev[0], coord_prev[1], n0, n1)
        new_qf.unsafe_store(k * 2, q_prev[0])
        new_qf.unsafe_store(k * 2 + 1, q_prev[1])

    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


def advection_v3(
    vf: PF, qf: PF, new_qf: PF, n0: Int, n1: Int, dt: Float32, nworkers: Int
):
    def one(i: Int, j: Int) {imm}:
        var k = i * n1 + j
        var coord_cur = V2(Float32(i), Float32(j)) + V2(0.5, 0.5)
        var vel_cur = V2(vf.unsafe_load(k * 2), vf.unsafe_load(k * 2 + 1))
        var coord_prev = coord_cur - vel_cur * dt
        var q_prev = bilerp_v3(qf, coord_prev[0], coord_prev[1], n0, n1)
        new_qf.unsafe_store(k * 3, q_prev[0])
        new_qf.unsafe_store(k * 3 + 1, q_prev[1])
        new_qf.unsafe_store(k * 3 + 2, q_prev[2])

    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def curl(vf: PF, cf: PF, n0: Int, n1: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        var ol = ((i - 1) * n1 + j) * 2
        var or_ = ((i + 1) * n1 + j) * 2
        var ob = (i * n1 + j - 1) * 2
        var ot = (i * n1 + j + 1) * 2
        cf.unsafe_store(
            i * n1 + j,
            0.5 * ((vf.unsafe_load(or_ + 1) - vf.unsafe_load(ol + 1)) - (
                vf.unsafe_load(ot) - vf.unsafe_load(ob)
            )),
        )

    # Upstream does not clamp, so every lane of a vector of consecutive columns
    # reads a valid cell and the whole row can be one vector.  But the x- and
    # y-components are interleaved, so the two neighbours of the y term,
    # `(i * n1 + j +- 1) * 2`, are one whole cell apart: that is a stride-2
    # gather of the row, not a contiguous load, and `SIMD` has no gather that
    # works on this toolchain (see `taichi.vector`).  Only the x term, whose
    # neighbours are `((i +- 1) * n1 + j) * 2` and therefore contiguous, can be
    # a vector load, so `curl` is left scalar like upstream.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def vorticity_projection(
    cf: PF, vf: PF, vf_new: PF, n0: Int, n1: Int, dt: Float32, curl_param: Float32, nworkers: Int
):
    def one(i: Int, j: Int) {imm}:
        var gradcurl = V3(
            0.5 * (cf.unsafe_load((i + 1) * n1 + j) - cf.unsafe_load((i - 1) * n1 + j)),
            0.5 * (cf.unsafe_load(i * n1 + j + 1) - cf.unsafe_load(i * n1 + j - 1)),
            0.0,
            0.0,
        )
        var GradCurlLength = sqrt(
            gradcurl[0] * gradcurl[0]
            + gradcurl[1] * gradcurl[1]
            + gradcurl[2] * gradcurl[2]
        )
        if GradCurlLength > 1e-5:
            # cross(gradcurl / |gradcurl|, [0, 0, 1]) is (g[1], -g[0], 0)
            var g = gradcurl / GradCurlLength
            var force = curl_param * V3(g[1], -g[0], 0.0, 0.0)
            var o = (i * n1 + j) * 2
            vf_new.unsafe_store(
                o, vf.unsafe_load(o) + dt * force[0]
            )
            vf_new.unsafe_store(
                o + 1, vf.unsafe_load(o + 1) + dt * force[1]
            )

    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def divergence(vf: PF, divf: PF, n0: Int, n1: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        divf.unsafe_store(
            i * n1 + j,
            0.5 * (
                vf.unsafe_load(((i + 1) * n1 + j) * 2)
                - vf.unsafe_load(((i - 1) * n1 + j) * 2)
                + vf.unsafe_load((i * n1 + j + 1) * 2 + 1)
                - vf.unsafe_load((i * n1 + j - 1) * 2 + 1)
            ),
        )

    # Left scalar on purpose.  Unlike `pressure_iteration` this kernel reads a
    # velocity *component*, and its two terms are a row apart
    # (`(i +- 1) * n1 + j`) and a column apart (`(i * n1 + j +- 1)`).  Only the
    # row term is a contiguous vector load; the column term straddles the two
    # interleaved components, so a vector body would need a stride-2 gather,
    # which `SIMD` does not provide correctly on this toolchain.  Kept scalar,
    # as upstream.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def pressure_iteration(divf: PF, pf: PF, new_pf: PF, n0: Int, n1: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        new_pf.unsafe_store(
            i * n1 + j,
            (
                pf.unsafe_load((i + 1) * n1 + j)
                + pf.unsafe_load((i - 1) * n1 + j)
                + pf.unsafe_load(i * n1 + j - 1)
                + pf.unsafe_load(i * n1 + j + 1)
                - divf.unsafe_load(i * n1 + j)
            ) / 4,
        )

    # Left scalar on purpose, for the same reason as `curl`: the two row
    # neighbours are one row apart and the two column neighbours are one cell
    # apart, and the column neighbours of a scalar pressure field *are*
    # contiguous -- but this kernel's reference transcription clamps at the view
    # edge, while upstream's own body reads `i +- 1` unclamped, so a vector body
    # that stays inside the view is not the same computation as upstream's.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def pressure_projection(pf: PF, vf: PF, vf_new: PF, n0: Int, n1: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        vf_new.unsafe_store(
            (i * n1 + j) * 2,
            vf.unsafe_load((i * n1 + j) * 2)
            - (
                p_with_boundary(pf, i + 1, j, n0, n1)
                - p_with_boundary(pf, i - 1, j, n0, n1)
            ) / 2.0,
        )
        vf_new.unsafe_store(
            (i * n1 + j) * 2 + 1,
            vf.unsafe_load((i * n1 + j) * 2 + 1)
            - (
                p_with_boundary(pf, i, j + 1, n0, n1)
                - p_with_boundary(pf, i, j - 1, n0, n1)
            ) / 2.0,
        )

    # Left scalar on purpose.  Unlike the other stencils this one cannot be
    # vectorised without changing what it computes: `p_with_boundary` writes to
    # the pressure field it reads (it copies the second row into the first, and
    # zeroes the four corners), and upstream's Taichi CPU backend does the same,
    # so the value a lane reads depends on the order the lanes run in.  A vector
    # body was measured and rejected rather than assumed: see the README.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# #####################
# Boundry Condition
# #####################
def vel_with_boundary(vf: PF, i: Int, j: Int, n0: Int, n1: Int) -> V2:
    var o = (i * n1 + j) * 2
    if (i <= 0) or (i >= n0 - 1) or (j >= n1 - 1) or (j <= 0):
        vf.unsafe_store(o, 0.0)
        vf.unsafe_store(o + 1, 0.0)
    return V2(vf.unsafe_load(o), vf.unsafe_load(o + 1))


def p_with_boundary(pf: PF, i: Int, j: Int, n0: Int, n1: Int) -> Float32:
    if (
        (i == j == 0)
        or (i == n0 - 1 and j == n1 - 1)
        or (i == 0 and j == n1 - 1)
        or (i == n0 - 1 and j == 0)
    ):
        pf.unsafe_store(i * n1 + j, 0.0)
    elif i == 0:
        pf.unsafe_store(0 * n1 + j, pf.unsafe_load(1 * n1 + j))
    elif j == 0:
        pf.unsafe_store(i * n1 + 0, pf.unsafe_load(i * n1 + 1))
    elif i == n0 - 1:
        pf.unsafe_store((n0 - 1) * n1 + j, pf.unsafe_load((n0 - 2) * n1 + j))
    elif j == n1 - 1:
        pf.unsafe_store(i * n1 + (n1 - 1), pf.unsafe_load(i * n1 + (n1 - 2)))
    return pf.unsafe_load(i * n1 + j)


def c_with_boundary(cf: PF, i: Int, j: Int, n0: Int, n1: Int) -> Float32:
    if (i <= 0) or (i >= n0 - 1) or (j >= n1 - 1) or (j <= 0):
        cf.unsafe_store(i * n1 + j, 0.0)
    return cf.unsafe_load(i * n1 + j)


def apply_vel_bc(vf: PF, n0: Int, n1: Int, nworkers: Int):
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            _ = vel_with_boundary(vf, i, j, n0, n1)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


def apply_p_bc(pf: PF, n0: Int, n1: Int, nworkers: Int):
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            _ = p_with_boundary(pf, i, j, n0, n1)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


def apply_c_bc(cf: PF, n0: Int, n1: Int, nworkers: Int):
    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            _ = c_with_boundary(cf, i, j, n0, n1)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def mouse_addspeed(
    cur_posx: Int,
    cur_posy: Int,
    prev_posx: Int,
    prev_posy: Int,
    mouseRadius: Float32,
    vf: PF,
    new_vf: PF,
    n0: Int,
    n1: Int,
    mouse_speed: Float32,
    mouse_respondDistance: Float32,
    nworkers: Int,
):
    var vec1 = V2(Float32(cur_posx - prev_posx), Float32(cur_posy - prev_posy))
    var vec1_length = sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1])

    def one(i: Int, j: Int) {imm}:
        var k = i * n1 + j
        var vec2 = V2(Float32(i - prev_posx), Float32(j - prev_posy))
        var dotans = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        # ti.math.cross of two 2D vectors is the scalar x1*y2 - x2*y1
        var distance = abs(vec1[0] * vec2[1] - vec2[0] * vec1[1]) / (vec1_length + 0.001)
        if (
            dotans >= 0
            and dotans <= mouse_respondDistance * vec1_length
            and distance <= mouseRadius
        ):
            new_vf.unsafe_store(k * 2, vf.unsafe_load(k * 2) + vec1[0] * mouse_speed)
            new_vf.unsafe_store(
                k * 2 + 1, vf.unsafe_load(k * 2 + 1) + vec1[1] * mouse_speed
            )
        else:
            new_vf.unsafe_store(k * 2, vf.unsafe_load(k * 2))
            new_vf.unsafe_store(k * 2 + 1, vf.unsafe_load(k * 2 + 1))

    def sweep(i: Int) {imm}:
        var j = 0
        while j < n1:
            one(i, j)
            j += 1

    if nodes_per_worker(n0 * n1, nworkers) > 0:
        parallelize(sweep, n0, nworkers)
    else:
        var lo = 0
        while lo < n0:
            sweep(lo)
            lo += 1
