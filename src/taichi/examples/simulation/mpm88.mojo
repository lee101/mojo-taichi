"""Mojo port of `taichi/examples/simulation/mpm88.py`.

Material Point Method with quadratic B-spline transfer, 2D, APIC affine
velocity field.  Same names, same order, same arithmetic as upstream.

Forced divergences, documented in the README:

  * upstream's module-level scalars (`n_particles`, `n_grid`, `dx`, `dt`,
    `p_vol`, `p_mass`, `E`, `gravity`, `bound`) are arguments.
  * `ti.static(ti.ndrange(3, 3))` is a plain `for i in range(3): for j in
    range(3):` — Mojo has no unrolled field iteration.
  * `init` takes the uniform [0, 1) draws as a buffer instead of calling
    `ti.random()`, so the caller controls the seed and the kernel stays
    reproducible (upstream's `ti.random()` stream is not reproducible across
    architectures).
  * upstream's particle `C` field is a 2x2 matrix field; it is stored here as
    a flat (n_particles, 2, 2) C-contiguous buffer, same as `to_numpy()`.
  * upstream builds one 3-vector of weights and indexes it twice, `w[i].x` for
    the x axis and `w[j].y` for the y axis; here that is two vectors, `w` and
    `wy`, with the same values.
  * upstream's P2G scatter and G2P gather index `grid_v[base + offset]` with no
    clamp, so a particle outside [0, 1] addresses memory off the field.
    Upstream is GPU-only and the access is undefined; through a raw pointer
    here it would be a wild write.  The node index is clamped to the grid,
    which leaves in-domain arithmetic bit-identical and confines an escaped
    particle to the border node.
"""

from std.math import clamp

comptime PF = Pointer[Float32, AnyOrigin[mut=True]]
comptime V2 = SIMD[DType.float32, 2]

# 3-element weight vectors live in 4 lanes: Mojo SIMD widths must be powers
# of two and the fourth lane is never read.
comptime V3 = SIMD[DType.float32, 4]
comptime V4 = SIMD[DType.float32, 4]



def substep(
    x: PF,
    v: PF,
    C: PF,
    J: PF,
    grid_v: PF,
    grid_m: PF,
    n_particles: Int,
    n_grid: Int,
    dx: Float32,
    dt: Float32,
    p_vol: Float32,
    p_mass: Float32,
    E: Float32,
    gravity: Float32,
    bound: Int,
    nworkers: Int,
):
    # for i, j in grid_m:
    #
    # Left serial on purpose.  The grid is `n_grid` squared and at the size this
    # port is called with (128x128) it is 192 KiB, which stays in L2 across both
    # passes, so neither sweep is a bandwidth problem; a vector body measured no
    # faster, and this keeps the loop identical to upstream's.  The normalise
    # sweep additionally cannot be vectorised as written: `m` is zero on every
    # node no particle reached, so its `m > 0` guard is per node and clamping
    # the divide into a vector expression produces NaN where upstream leaves
    # the value alone.

    for i in range(n_grid):
        var base = i * n_grid
        for j in range(n_grid):
            grid_v.unsafe_store(base * 2 + j * 2, 0.0)
            grid_v.unsafe_store(base * 2 + j * 2 + 1, 0.0)
            grid_m.unsafe_store(base + j, 0.0)

    for p in range(n_particles):
        var Xp = V2(x.unsafe_load(p * 2), x.unsafe_load(p * 2 + 1)) / dx
        var base_i = Int(Xp[0] - 0.5)
        var base_j = Int(Xp[1] - 0.5)
        var fx = Xp - V2(Float32(base_i), Float32(base_j))
        # w is a 3-vector: w[i].x upstream, w[i] here
        var w = V3(
            0.5 * (1.5 - fx[0]) ** 2,
            0.75 - (fx[0] - 1) ** 2,
            0.5 * (fx[0] - 0.5) ** 2,
            0.0,
        )
        var wy = V3(
            0.5 * (1.5 - fx[1]) ** 2,
            0.75 - (fx[1] - 1) ** 2,
            0.5 * (fx[1] - 0.5) ** 2,
            0.0,
        )
        var stress = -dt * 4 * E * p_vol * (J.unsafe_load(p) - 1) / dx**2
        var c0 = C.unsafe_load(p * 4)
        var c1 = C.unsafe_load(p * 4 + 1)
        var c2 = C.unsafe_load(p * 4 + 2)
        var c3 = C.unsafe_load(p * 4 + 3)
        # affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        var a00 = stress + p_mass * c0
        var a01 = p_mass * c1
        var a10 = p_mass * c2
        var a11 = stress + p_mass * c3
        var vp = V2(v.unsafe_load(p * 2), v.unsafe_load(p * 2 + 1))
        for i in range(3):
            for j in range(3):
                var offset = V2(Float32(i), Float32(j))
                var dpos = (offset - fx) * dx
                var weight = w[i] * wy[j]
                # upstream indexes `grid_v[base + offset]` with no clamp, so a
                # particle outside [0, 1] walks off the field.  Upstream is GPU
                # only and the access is undefined; here it would be a wild
                # pointer write.  Clamp the node instead: in-domain arithmetic
                # is unchanged (see test_substep_parity), and an escaped
                # particle piles onto the border node rather than corrupting
                # the heap.
                var gi = clamp(base_i + i, 0, n_grid - 1)
                var gj = clamp(base_j + j, 0, n_grid - 1)
                var o = (gi * n_grid + gj) * 2
                var impulse = weight * (p_mass * vp + V2(a00 * dpos[0] + a01 * dpos[1], a10 * dpos[0] + a11 * dpos[1]))
                grid_v.unsafe_store(o, grid_v.unsafe_load(o) + impulse[0])
                grid_v.unsafe_store(o + 1, grid_v.unsafe_load(o + 1) + impulse[1])
                grid_m.unsafe_store(gi * n_grid + gj, grid_m.unsafe_load(gi * n_grid + gj) + weight * p_mass)


    for i in range(n_grid):
        for j in range(n_grid):
            var k = i * n_grid + j
            var m = grid_m.unsafe_load(k)
            var o = k * 2
            if m > 0:
                grid_v.unsafe_store(o, grid_v.unsafe_load(o) / m)
                grid_v.unsafe_store(o + 1, grid_v.unsafe_load(o + 1) / m)
            grid_v.unsafe_store(o + 1, grid_v.unsafe_load(o + 1) - dt * gravity)
            if i < bound and grid_v.unsafe_load(o) < 0:
                grid_v.unsafe_store(o, 0.0)
            if i > n_grid - bound and grid_v.unsafe_load(o) > 0:
                grid_v.unsafe_store(o, 0.0)
            if j < bound and grid_v.unsafe_load(o + 1) < 0:
                grid_v.unsafe_store(o + 1, 0.0)
            if j > n_grid - bound and grid_v.unsafe_load(o + 1) > 0:
                grid_v.unsafe_store(o + 1, 0.0)

    for p in range(n_particles):
        var Xp = V2(x.unsafe_load(p * 2), x.unsafe_load(p * 2 + 1)) / dx
        var base_i = Int(Xp[0] - 0.5)
        var base_j = Int(Xp[1] - 0.5)
        var fx = Xp - V2(Float32(base_i), Float32(base_j))
        var w = V3(
            0.5 * (1.5 - fx[0]) ** 2,
            0.75 - (fx[0] - 1) ** 2,
            0.5 * (fx[0] - 0.5) ** 2,
            0.0,
        )
        var wy = V3(
            0.5 * (1.5 - fx[1]) ** 2,
            0.75 - (fx[1] - 1) ** 2,
            0.5 * (fx[1] - 0.5) ** 2,
            0.0,
        )
        var new_v = V2(0.0, 0.0)
        var new_C = V4(0.0, 0.0, 0.0, 0.0)
        for i in range(3):
            for j in range(3):
                var offset = V2(Float32(i), Float32(j))
                var dpos = (offset - fx) * dx
                var weight = w[i] * wy[j]
                # same clamp as the P2G scatter, and for the same reason
                var gi = clamp(base_i + i, 0, n_grid - 1)
                var gj = clamp(base_j + j, 0, n_grid - 1)
                var o = (gi * n_grid + gj) * 2
                var g_v = V2(grid_v.unsafe_load(o), grid_v.unsafe_load(o + 1))
                new_v = new_v + weight * g_v
                # 4 * weight * g_v.outer_product(dpos) / dx**2
                new_C = new_C + (
                    4
                    * weight
                    * V4(
                        g_v[0] * dpos[0],
                        g_v[0] * dpos[1],
                        g_v[1] * dpos[0],
                        g_v[1] * dpos[1],
                    )
                ) / dx**2
        v.unsafe_store(p * 2, new_v[0])
        v.unsafe_store(p * 2 + 1, new_v[1])
        x.unsafe_store(p * 2, x.unsafe_load(p * 2) + dt * new_v[0])
        x.unsafe_store(p * 2 + 1, x.unsafe_load(p * 2 + 1) + dt * new_v[1])
        J.unsafe_store(p, J.unsafe_load(p) * (1 + dt * (new_C[0] + new_C[3])))
        C.unsafe_store(p * 4, new_C[0])
        C.unsafe_store(p * 4 + 1, new_C[1])
        C.unsafe_store(p * 4 + 2, new_C[2])
        C.unsafe_store(p * 4 + 3, new_C[3])


def init(x: PF, v: PF, J: PF, uniforms: PF, n_particles: Int):
    # Left scalar on purpose.  `x` is an element-wise map, but `v` is not: its
    # y component is -1 for every particle, which in this interleaved layout is
    # a per-lane sign mask rather than a splat.  `init` is one pass over
    # `n_particles` and is not in any benchmark.
    for i in range(n_particles):
        x.unsafe_store(i * 2, uniforms.unsafe_load(i * 2) * 0.4 + 0.2)
        x.unsafe_store(i * 2 + 1, uniforms.unsafe_load(i * 2 + 1) * 0.4 + 0.2)
        v.unsafe_store(i * 2, 0.0)
        v.unsafe_store(i * 2 + 1, -1.0)
        J.unsafe_store(i, 1.0)

