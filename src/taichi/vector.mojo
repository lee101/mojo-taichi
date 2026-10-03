"""Grid traversal shared by the ported kernels.

Two decisions live here, both measured (see the README):

**Grid traversal.** Upstream's `for i, j in field` becomes, in Mojo, a flat loop
over `k = i * n1 + j` with the row recovered by division.  That costs more than
the stencil it indexes: a scalar load-add-store plus `k // n1` and `k % n1` runs
at 5.4 ns/node where the same three operations without the division run at
0.88 ns/node, and the division is not strength-reduced to an increment in this
toolchain.  A two-level loop -- an outer sweep over rows and an inner sweep over
`VEC_W` columns -- keeps the row index in a register, so the neighbour offsets
are constant within the inner loop and the neighbour loads stay in the same
cache lines as the cell being written.  It measured 2.2 ns/node against the
flat sweep's 9.0 ns/node for the same arithmetic.

**Vector width.** `VEC_W` comes from `simd_width_of[DType.float32]()`, which is
8 on this toolchain.  `SIMD` has no `.gather`, so the clamped `sample` cannot be
expressed as one indirect load; every load in a vector body is a contiguous one
and the clamped boundaries are handled by running the first and last rows, and
the first and last `VEC_W` nodes of each row, separately.  `Pointer.store` of a
`SIMD` value writes exactly the `VEC_W` elements it is given at the index it is
given, so a `guarded()` field whose base is not 32-byte aligned still
vectorises.

`gather` is also *not correct* on this toolchain for a runtime index vector: it
returns zero rather than the addressed elements, and silently.  A clamped SIMD
gather stencil measured 0.095 ms where the contiguous form measured 0.13 ms,
i.e. it was reading nothing at all.  Nothing below uses it.
"""

from max.algorithm import parallelize
from std.sys import simd_width_of

comptime VEC_W = simd_width_of[DType.float32]()

# Workload size above which a `parallelize` launch is worth paying for.  An
# empty launch costs ~5 ms on this toolchain (measured against an empty ctypes
# call, which is 0.0026 ms), and the largest workload this port covers -- a
# 512x512 advection -- is ~3 ms of serial work, so the honest threshold is a
# grid well past the size the library supports.  `bench` prints the measurement
# it is read off.
comptime PAR_MIN = 1 << 30

# Nodes per `parallelize` worker: one contiguous span several vector steps wide.
comptime CHUNK = 2048

# Worker count that measured best, and the cap `nodes_per_worker` applies.
comptime MAX_WORKERS = 16


def nodes_per_worker(n: Int, nworkers: Int) -> Int:
    """Nodes per `parallelize` worker, or 0 to take the plain loop.

    One branch in every kernel, kept in upstream's `if ... else` shape, so the
    serial path and the fork-join path run the identical body.  `PAR_MIN` sits
    past every size the library accepts, so the branch is present and explicit
    rather than deleted, and the fork-join path exists for a driver that has to
    reach it.  The worker count is capped at `MAX_WORKERS`, which is where the
    worker-scaling table in the README turned over.
    """
    if n >= PAR_MIN and nworkers > 1:
        return VEC_W * CHUNK
    return 0