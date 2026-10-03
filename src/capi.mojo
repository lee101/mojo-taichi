"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric.  Callers own all
memory, including scratch — nothing here allocates, so nothing here can leak.

This is the single compilation unit: `mojo build --emit shared-lib -I src
src/capi.mojo` pulls in the ported kernels under `src/taichi/`.
"""

from taichi.examples.patterns import taichi_logo
from taichi.examples.simulation.eulerfluid2d import (
    advection_v2,
    advection_v3,
    apply_c_bc,
    apply_p_bc,
    apply_vel_bc,
    curl,
    divergence as e_divergence,
    init_field,
    mouse_addspeed,
    pressure_iteration,
    pressure_projection,
    vorticity_projection,
)
from taichi.examples.simulation.mpm88 import init as mpm88_init, substep as mpm88_substep
from taichi.examples.simulation.stable_fluid import (
    advect_v2,
    advect_v3,
    apply_impulse,
    apply_pressure,
    copy_divergence,
    divergence,
    enhance_vorticity,
    pressure_jacobi,
    subtract_gradient,
    vorticity,
)

from taichi.vector import VEC_W

comptime PF = Pointer[Float32, AnyOrigin[mut=True]]


def p(addr: Int) -> PF:
    return PF(unsafe_from_address=addr)


@export("mt_simd_width")
def mt_simd_width() abi("C") -> Int:
    """The float32 SIMD width the kernels were built with.

    The test suite reads this rather than hard-coding a number, so a toolchain
    that reports a different width fails there instead of quietly skipping every
    vector loop in the parity tests.
    """
    return VEC_W


# ------------------------------------------------------- taichi.examples.patterns
@export("mt_taichi_logo")
def mt_taichi_logo(pos: Int, dst: Int, n: Int, scale: Float32) abi("C"):
    var src = p(pos)
    var out = p(dst)
    for i in range(n):
        out.unsafe_store(
            i,
            taichi_logo(
                SIMD[DType.float32, 2](src.unsafe_load(i * 2), src.unsafe_load(i * 2 + 1)),
                scale,
            ),
        )


# ------------------------------------------------------ taichi.examples.simulation
# --------------------------------------------------------------- stable_fluid
@export("mt_sf_advect_v2")
def mt_sf_advect_v2(
    vf: Int, qf: Int, new_qf: Int, res: Int, dt: Float32, dye_decay: Float32, nw: Int
) abi("C"):
    advect_v2(p(vf), p(qf), p(new_qf), res, dt, dye_decay, nw)


@export("mt_sf_advect_v3")
def mt_sf_advect_v3(
    vf: Int, qf: Int, new_qf: Int, res: Int, dt: Float32, dye_decay: Float32, nw: Int
) abi("C"):
    advect_v3(p(vf), p(qf), p(new_qf), res, dt, dye_decay, nw)


@export("mt_sf_apply_impulse")
def mt_sf_apply_impulse(
    vf: Int,
    dyef: Int,
    imp_data: Int,
    res: Int,
    dt: Float32,
    f_strength: Float32,
    force_radius: Float32,
    nw: Int,
) abi("C"):
    apply_impulse(p(vf), p(dyef), p(imp_data), res, dt, f_strength, force_radius, nw)


@export("mt_sf_divergence")
def mt_sf_divergence(vf: Int, divs: Int, res: Int, nw: Int) abi("C"):
    divergence(p(vf), p(divs), res, nw)


@export("mt_sf_vorticity")
def mt_sf_vorticity(vf: Int, curls: Int, res: Int, nw: Int) abi("C"):
    vorticity(p(vf), p(curls), res, nw)


@export("mt_sf_pressure_jacobi")
def mt_sf_pressure_jacobi(pf: Int, new_pf: Int, divs: Int, res: Int, nw: Int) abi("C"):
    pressure_jacobi(p(pf), p(new_pf), p(divs), res, nw)


@export("mt_sf_subtract_gradient")
def mt_sf_subtract_gradient(vf: Int, pf: Int, res: Int, nw: Int) abi("C"):
    subtract_gradient(p(vf), p(pf), res, nw)


@export("mt_sf_enhance_vorticity")
def mt_sf_enhance_vorticity(
    vf: Int, cf: Int, res: Int, curl_strength: Float32, dt: Float32, nw: Int
) abi("C"):
    enhance_vorticity(p(vf), p(cf), res, curl_strength, dt, nw)


@export("mt_sf_copy_divergence")
def mt_sf_copy_divergence(div_in: Int, div_out: Int, res: Int, nw: Int) abi("C"):
    copy_divergence(p(div_in), p(div_out), res, nw)


@export("mt_sf_apply_pressure")
def mt_sf_apply_pressure(p_in: Int, p_out: Int, res: Int, nw: Int) abi("C"):
    apply_pressure(p(p_in), p(p_out), res, nw)


# -------------------------------------------------------------- eulerfluid2d
@export("mt_ef_init_field")
def mt_ef_init_field(press: Int, vel: Int, color: Int, n0: Int, n1: Int) abi("C"):
    init_field(p(press), p(vel), p(color), n0, n1)


@export("mt_ef_advection_v2")
def mt_ef_advection_v2(
    vf: Int, qf: Int, new_qf: Int, n0: Int, n1: Int, dt: Float32, nw: Int
) abi("C"):
    advection_v2(p(vf), p(qf), p(new_qf), n0, n1, dt, nw)


@export("mt_ef_advection_v3")
def mt_ef_advection_v3(
    vf: Int, qf: Int, new_qf: Int, n0: Int, n1: Int, dt: Float32, nw: Int
) abi("C"):
    advection_v3(p(vf), p(qf), p(new_qf), n0, n1, dt, nw)


@export("mt_ef_curl")
def mt_ef_curl(vf: Int, cf: Int, n0: Int, n1: Int, nw: Int) abi("C"):
    curl(p(vf), p(cf), n0, n1, nw)


@export("mt_ef_vorticity_projection")
def mt_ef_vorticity_projection(
    cf: Int, vf: Int, vf_new: Int, n0: Int, n1: Int, dt: Float32, curl_param: Float32, nw: Int
) abi("C"):
    vorticity_projection(p(cf), p(vf), p(vf_new), n0, n1, dt, curl_param, nw)


@export("mt_ef_divergence")
def mt_ef_divergence(vf: Int, divf: Int, n0: Int, n1: Int, nw: Int) abi("C"):
    e_divergence(p(vf), p(divf), n0, n1, nw)


@export("mt_ef_pressure_iteration")
def mt_ef_pressure_iteration(
    divf: Int, pf: Int, new_pf: Int, n0: Int, n1: Int, nw: Int
) abi("C"):
    pressure_iteration(p(divf), p(pf), p(new_pf), n0, n1, nw)


@export("mt_ef_pressure_projection")
def mt_ef_pressure_projection(pf: Int, vf: Int, vf_new: Int, n0: Int, n1: Int, nw: Int) abi("C"):
    pressure_projection(p(pf), p(vf), p(vf_new), n0, n1, nw)


@export("mt_ef_apply_vel_bc")
def mt_ef_apply_vel_bc(vf: Int, n0: Int, n1: Int, nw: Int) abi("C"):
    apply_vel_bc(p(vf), n0, n1, nw)


@export("mt_ef_apply_p_bc")
def mt_ef_apply_p_bc(pf: Int, n0: Int, n1: Int, nw: Int) abi("C"):
    apply_p_bc(p(pf), n0, n1, nw)


@export("mt_ef_apply_c_bc")
def mt_ef_apply_c_bc(cf: Int, n0: Int, n1: Int, nw: Int) abi("C"):
    apply_c_bc(p(cf), n0, n1, nw)


@export("mt_ef_mouse_addspeed")
def mt_ef_mouse_addspeed(
    cur_posx: Int,
    cur_posy: Int,
    prev_posx: Int,
    prev_posy: Int,
    mouseRadius: Float32,
    vf: Int,
    new_vf: Int,
    n0: Int,
    n1: Int,
    mouse_speed: Float32,
    mouse_respondDistance: Float32,
    nw: Int,
) abi("C"):
    mouse_addspeed(
        cur_posx,
        cur_posy,
        prev_posx,
        prev_posy,
        mouseRadius,
        p(vf),
        p(new_vf),
        n0,
        n1,
        mouse_speed,
        mouse_respondDistance,
        nw,
    )


# -------------------------------------------------------------------- mpm88
@export("mt_mpm88_substep")
def mt_mpm88_substep(
    x: Int,
    v: Int,
    C: Int,
    J: Int,
    grid_v: Int,
    grid_m: Int,
    n_particles: Int,
    n_grid: Int,
    dx: Float32,
    dt: Float32,
    p_vol: Float32,
    p_mass: Float32,
    E: Float32,
    gravity: Float32,
    bound: Int,
    nw: Int,
) abi("C"):
    mpm88_substep(
        p(x), p(v), p(C), p(J), p(grid_v), p(grid_m),
        n_particles, n_grid, dx, dt, p_vol, p_mass, E, gravity, bound, nw,
    )


@export("mt_mpm88_init")
def mt_mpm88_init(x: Int, v: Int, J: Int, uniforms: Int, n_particles: Int) abi("C"):
    mpm88_init(p(x), p(v), p(J), p(uniforms), n_particles)
