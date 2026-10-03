"""Mojo port of `taichi/examples/patterns.py`.

Only `taichi_logo` is used by the ported solvers, so only it is here.
"""

comptime V2 = SIMD[DType.float32, 2]


def norm_sqr(v: V2) -> Float32:
    return v[0] * v[0] + v[1] * v[1]


def taichi_logo(pos: V2, scale: Float32 = Float32(1.0 / 1.11)) -> Float32:
    var p = (pos - V2(0.5)) / scale + V2(0.5)
    var ret = -1
    if not norm_sqr(p - V2(0.50)) <= Float32(0.52) ** 2:
        if ret == -1:
            ret = 0
    if not norm_sqr(p - V2(0.50)) <= Float32(0.495) ** 2:
        if ret == -1:
            ret = 1
    if norm_sqr(p - V2(0.50, 0.25)) <= Float32(0.08) ** 2:
        if ret == -1:
            ret = 1
    if norm_sqr(p - V2(0.50, 0.75)) <= Float32(0.08) ** 2:
        if ret == -1:
            ret = 0
    if norm_sqr(p - V2(0.50, 0.25)) <= Float32(0.25) ** 2:
        if ret == -1:
            ret = 0
    if norm_sqr(p - V2(0.50, 0.75)) <= Float32(0.25) ** 2:
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return Float32(1 - ret)
