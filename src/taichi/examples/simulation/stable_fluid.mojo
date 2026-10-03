"""Mojo port of `taichi/examples/simulation/stable_fluid.py`.

Stam's stable-fluids solver (GPU Gems ch. 38).  Every kernel below is the
upstream kernel with the same name, the same argument order and the same
arithmetic; upstream's module-level constants (`res`, `dt`, `dye_decay`,
`f_strength`, `force_radius`, `curl_strength`) become arguments because Mojo
has no module globals.

Forced divergences, all documented in the README:

  * `ti.template()` fields do not exist in Mojo, so a function upstream
    instantiates per field rank becomes one function per rank:
    `sample`/`lerp`/`bilerp` gain a `_v2`/`_v3` sibling.
  * `for i, j in vf` (a 2D field iteration) becomes the two-level loop described
    in `taichi.vector`: an outer sweep over rows and an inner sweep of `VEC_W`
    columns, so the row index stays in a register instead of being recovered
    from `k` by division.  Body order is unchanged.
  * `sample` clamps to the grid.  A whole vector of `VEC_W` consecutive columns
    has every lane's `j - 1` and `j + 1` inside the row only when the vector
    starts at or after column 1 and ends at or before column `res - 2`, so a
    row's first and last `VEC_W - 1` nodes take the scalar path along with the
    first and last row.  Same arithmetic as the clamp, just not re-derived per
    lane.
  * `advect` and `apply_impulse` stay scalar.  Both bodies are per-node scalar
    arithmetic over node-dependent addresses -- four bilinear samplings for
    `advect`, two exponentials for `apply_impulse` -- and `SIMD` has no working
    `gather` on this toolchain, so there is no load to widen and widening the
    arithmetic only adds work.  Both measurements are in the README.
"""

from std.math import exp, floor, sqrt

from max.algorithm import parallelize
from taichi.vector import VEC_W, nodes_per_worker

comptime PF = Pointer[Float32, AnyOrigin[mut=True]]
comptime V2 = SIMD[DType.float32, 2]
# Mojo SIMD widths must be powers of two, so 3-vectors live in a 4-lane
# vector with a permanently zero fourth lane that is never read.
comptime V3 = SIMD[DType.float32, 4]
comptime FW = SIMD[DType.float32, VEC_W]

# A row's first and last `HEAD` columns cannot be loaded as one vector without
# the `sample` clamp reappearing, so those nodes take the scalar path.
comptime HEAD = VEC_W - 1



# --------------------------------------------------------------- ti.func sample
def sample(qf: PF, u: Int, v: Int, res: Int) -> Float32:
    var i = max(0, min(res - 1, u))
    var j = max(0, min(res - 1, v))
    return qf.unsafe_load(i * res + j)


def sample_v2(vf: PF, u: Int, v: Int, res: Int) -> V2:
    var i = max(0, min(res - 1, u))
    var j = max(0, min(res - 1, v))
    var o = (i * res + j) * 2
    return V2(vf.unsafe_load(o), vf.unsafe_load(o + 1))


def sample_v3(vf: PF, u: Int, v: Int, res: Int) -> V3:
    var i = max(0, min(res - 1, u))
    var j = max(0, min(res - 1, v))
    var o = (i * res + j) * 3
    return V3(vf.unsafe_load(o), vf.unsafe_load(o + 1), vf.unsafe_load(o + 2), 0.0)


# ------------------------------------------------------------------ ti.func lerp
def lerp_v2(vl: V2, vr: V2, frac: Float32) -> V2:
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


def lerp_v3(vl: V3, vr: V3, frac: Float32) -> V3:
    return vl + frac * (vr - vl)


# ---------------------------------------------------------------- ti.func bilerp
def bilerp_v2(vf: PF, p: V2, res: Int) -> V2:
    var u = p[0]
    var v = p[1]
    var s = u - 0.5
    var t = v - 0.5
    # floor
    var iu = floor(s)
    var iv = floor(t)
    # fract
    var fu = s - iu
    var fv = t - iv
    var a = sample_v2(vf, Int(iu), Int(iv), res)
    var b = sample_v2(vf, Int(iu + 1), Int(iv), res)
    var c = sample_v2(vf, Int(iu), Int(iv + 1), res)
    var d = sample_v2(vf, Int(iu + 1), Int(iv + 1), res)
    return lerp_v2(lerp_v2(a, b, fu), lerp_v2(c, d, fu), fv)


def bilerp_v3(vf: PF, p: V2, res: Int) -> V3:
    var u = p[0]
    var v = p[1]
    var s = u - 0.5
    var t = v - 0.5
    var iu = floor(s)
    var iv = floor(t)
    var fu = s - iu
    var fv = t - iv
    var a = sample_v3(vf, Int(iu), Int(iv), res)
    var b = sample_v3(vf, Int(iu + 1), Int(iv), res)
    var c = sample_v3(vf, Int(iu), Int(iv + 1), res)
    var d = sample_v3(vf, Int(iu + 1), Int(iv + 1), res)
    return lerp_v3(lerp_v3(a, b, fu), lerp_v3(c, d, fu), fv)


# ------------------------------------------------- 3rd order Runge-Kutta
def backtrace(vf: PF, p: V2, dt_: Float32, res: Int) -> V2:
    var v1 = bilerp_v2(vf, p, res)
    var p1 = p - 0.5 * dt_ * v1
    var v2 = bilerp_v2(vf, p1, res)
    var p2 = p - 0.75 * dt_ * v2
    var v3 = bilerp_v2(vf, p2, res)
    return p - dt_ * (
        (Float32(2.0) / Float32(9.0)) * v1
        + (Float32(1.0) / Float32(3.0)) * v2
        + (Float32(4.0) / Float32(9.0)) * v3
    )


# ------------------------------------------------------------------- ti.kernel
def advect_v2(
    vf: PF, qf: PF, new_qf: PF, res: Int, dt: Float32, dye_decay: Float32, nworkers: Int
):
    def one(i: Int, j: Int) {imm}:
        var p = V2(Float32(i), Float32(j)) + 0.5
        p = backtrace(vf, p, dt, res)
        var q = bilerp_v2(qf, p, res) * dye_decay
        var o = (i * res + j) * 2
        new_qf.unsafe_store(o, q[0])
        new_qf.unsafe_store(o + 1, q[1])

    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


def advect_v3(
    vf: PF, qf: PF, new_qf: PF, res: Int, dt: Float32, dye_decay: Float32, nworkers: Int
):
    def one(i: Int, j: Int) {imm}:
        var p = V2(Float32(i), Float32(j)) + 0.5
        p = backtrace(vf, p, dt, res)
        var q = bilerp_v3(qf, p, res) * dye_decay
        var o = (i * res + j) * 3
        new_qf.unsafe_store(o, q[0])
        new_qf.unsafe_store(o + 1, q[1])
        new_qf.unsafe_store(o + 2, q[2])

    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def apply_impulse(
    vf: PF,
    dyef: PF,
    imp_data: PF,
    res: Int,
    dt: Float32,
    f_strength: Float32,
    force_radius: Float32,
    nworkers: Int,
):
    var g_dir = -V2(0.0, 9.8) * 300.0
    var omx = imp_data.unsafe_load(2)
    var omy = imp_data.unsafe_load(3)
    var mdir = V2(imp_data.unsafe_load(0), imp_data.unsafe_load(1))
    var dye_color = V3(
        imp_data.unsafe_load(4),
        imp_data.unsafe_load(5),
        imp_data.unsafe_load(6),
        0.0,
    )
    # upstream writes `4 / (res / 15) ** 2` in Python (float64) before it
    # reaches the kernel; reproduce that float64 value, then round once.
    var dye_falloff = Float32(Float64(4.0) / (Float64(res) / 15.0) ** 2)
    var mdir_norm = sqrt(mdir[0] * mdir[0] + mdir[1] * mdir[1])

    def one(i: Int, j: Int) {imm}:
        var dx = Float32(i) + 0.5 - omx
        var dy = Float32(j) + 0.5 - omy
        var d2 = dx * dx + dy * dy
        # dv = F * dt
        var factor = exp(-d2 / force_radius)

        var c = (i * res + j) * 3
        var dc = V3(
            dyef.unsafe_load(c), dyef.unsafe_load(c + 1), dyef.unsafe_load(c + 2), 0.0
        )
        var a = sqrt(dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2])

        var momentum = (mdir * f_strength * factor + g_dir * a / (1.0 + a)) * dt

        var o = (i * res + j) * 2
        var v = V2(vf.unsafe_load(o), vf.unsafe_load(o + 1))
        vf.unsafe_store(o, v[0] + momentum[0])
        vf.unsafe_store(o + 1, v[1] + momentum[1])
        # add dye
        if mdir_norm > 0.5:
            dc = dc + exp(-d2 * dye_falloff) * dye_color

        dyef.unsafe_store(c, dc[0])
        dyef.unsafe_store(c + 1, dc[1])
        dyef.unsafe_store(c + 2, dc[2])

    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def divergence(vf: PF, velocity_divs: PF, res: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        var vl = sample_v2(vf, i - 1, j, res)
        var vr = sample_v2(vf, i + 1, j, res)
        var vb = sample_v2(vf, i, j - 1, res)
        var vt = sample_v2(vf, i, j + 1, res)
        var vc = sample_v2(vf, i, j, res)
        if i == 0:
            vl = V2(-vc[0], vl[1])
        if i == res - 1:
            vr = V2(-vc[0], vr[1])
        if j == 0:
            vb = V2(vb[0], -vc[1])
        if j == res - 1:
            vt = V2(vt[0], -vc[1])
        velocity_divs.unsafe_store(i * res + j, (vr[0] - vl[0] + vt[1] - vb[1]) * 0.5)

    # Left scalar on purpose.  The velocity field is interleaved (x, y per cell),
    # so every term of this stencil -- `vl[0]` at `(i - 1) * res + c`, `vb[1]`
    # at `i * res + c - 1` -- is a stride-2 read of one component across
    # consecutive columns.  A `VEC_W`-wide contiguous load cannot produce it and
    # `SIMD.gather` does not return the addressed elements on this toolchain (see
    # `taichi.vector`), so there is nothing to vectorise here.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def vorticity(vf: PF, velocity_curls: PF, res: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        var vl = sample_v2(vf, i - 1, j, res)
        var vr = sample_v2(vf, i + 1, j, res)
        var vb = sample_v2(vf, i, j - 1, res)
        var vt = sample_v2(vf, i, j + 1, res)
        velocity_curls.unsafe_store(i * res + j, (vr[1] - vl[1] - vt[0] + vb[0]) * 0.5)

    # Left scalar on purpose.  The velocity field is interleaved (x, y per cell),
    # so every term of this stencil -- `vl[0]` at `(i - 1) * res + c`, `vb[1]`
    # at `i * res + c - 1` -- is a stride-2 read of one component across
    # consecutive columns.  A `VEC_W`-wide contiguous load cannot produce it and
    # `SIMD.gather` does not return the addressed elements on this toolchain (see
    # `taichi.vector`), so there is nothing to vectorise here.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def pressure_jacobi(pf: PF, new_pf: PF, velocity_divs: PF, res: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        var pl = sample(pf, i - 1, j, res)
        var pr = sample(pf, i + 1, j, res)
        var pb = sample(pf, i, j - 1, res)
        var pt = sample(pf, i, j + 1, res)
        var div = velocity_divs.unsafe_load(i * res + j)
        new_pf.unsafe_store(i * res + j, (pl + pr + pb + pt - div) * 0.25)

    def sweep(i: Int) {imm}:
        var base = i * res
        var up = max(0, i - 1) * res
        var dn = min(res - 1, i + 1) * res
        var j = 0
        while j < HEAD:
            one(i, j)
            j += 1
        while j + VEC_W <= res - HEAD:
            var c = base + j
            new_pf.unsafe_store(
                c,
                (
                    pf.unsafe_load[width=VEC_W](up + j)
                    + pf.unsafe_load[width=VEC_W](dn + j)
                    + pf.unsafe_load[width=VEC_W](c - 1)
                    + pf.unsafe_load[width=VEC_W](c + 1)
                    - velocity_divs.unsafe_load[width=VEC_W](c)
                )
                * FW(0.25),
            )
            j += VEC_W
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def subtract_gradient(vf: PF, pf: PF, res: Int, nworkers: Int):
    def one(i: Int, j: Int) {imm}:
        var pl = sample(pf, i - 1, j, res)
        var pr = sample(pf, i + 1, j, res)
        var pb = sample(pf, i, j - 1, res)
        var pt = sample(pf, i, j + 1, res)
        var o = (i * res + j) * 2
        var v = V2(vf.unsafe_load(o), vf.unsafe_load(o + 1))
        v = v - 0.5 * V2(pr - pl, pt - pb)
        vf.unsafe_store(o, v[0])
        vf.unsafe_store(o + 1, v[1])

    # Left scalar on purpose.  The velocity field is interleaved (x, y per cell),
    # so every term of this stencil -- `vl[0]` at `(i - 1) * res + c`, `vb[1]`
    # at `i * res + c - 1` -- is a stride-2 read of one component across
    # consecutive columns.  A `VEC_W`-wide contiguous load cannot produce it and
    # `SIMD.gather` does not return the addressed elements on this toolchain (see
    # `taichi.vector`), so there is nothing to vectorise here.
    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def enhance_vorticity(
    vf: PF, cf: PF, res: Int, curl_strength: Float32, dt: Float32, nworkers: Int
):
    def one(i: Int, j: Int) {imm}:
        var cl = sample(cf, i - 1, j, res)
        var cr = sample(cf, i + 1, j, res)
        var cb = sample(cf, i, j - 1, res)
        var ct = sample(cf, i, j + 1, res)
        var cc = sample(cf, i, j, res)
        # Vector.normalized(eps) is `1 / (norm + eps) * v`
        var f = V2(abs(ct) - abs(cb), abs(cl) - abs(cr))
        var invlen = 1.0 / (sqrt(f[0] * f[0] + f[1] * f[1]) + Float32(1e-3))
        f = invlen * f
        f = f * (curl_strength * cc)
        var o = (i * res + j) * 2
        var v = V2(vf.unsafe_load(o), vf.unsafe_load(o + 1))
        v = min(max(v + f * dt, -1e3), 1e3)
        vf.unsafe_store(o, v[0])
        vf.unsafe_store(o + 1, v[1])

    def sweep(i: Int) {imm}:
        var j = 0
        while j < res:
            one(i, j)
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def copy_divergence(div_in: PF, div_out: PF, res: Int, nworkers: Int):
    def sweep(i: Int) {imm}:
        var base = i * res
        var j = 0
        while j + VEC_W <= res:
            div_out.unsafe_store(base + j, -div_in.unsafe_load[width=VEC_W](base + j))
            j += VEC_W
        while j < res:
            div_out.unsafe_store(base + j, -div_in.unsafe_load(base + j))
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1


# ------------------------------------------------------------------- ti.kernel
def apply_pressure(p_in: PF, p_out: PF, res: Int, nworkers: Int):
    def sweep(i: Int) {imm}:
        var base = i * res
        var j = 0
        while j + VEC_W <= res:
            p_out.unsafe_store(base + j, p_in.unsafe_load[width=VEC_W](base + j))
            j += VEC_W
        while j < res:
            p_out.unsafe_store(base + j, p_in.unsafe_load(base + j))
            j += 1

    if nodes_per_worker(res * res, nworkers) > 0:
        parallelize(sweep, res, nworkers)
    else:
        var lo = 0
        while lo < res:
            sweep(lo)
            lo += 1