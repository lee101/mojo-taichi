"""mojo-taichi against upstream taichi on the same inputs.

Both sides run the same source: the Mojo kernels from `src/`, and the taichi
kernels exec'd out of the installed `taichi` package (see `load_upstream`), so
this is a like-for-like comparison of the two implementations, not of two
different algorithms.

Run it with `pixi run bench` (the task holds a machine-wide lock so a concurrent
factory job cannot distort the numbers).  Not `python bench/bench.py`.
"""

import ast
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mojotaichi  # noqa: E402
from mojotaichi import eulerfluid2d as mte  # noqa: E402
from mojotaichi import mpm88 as mtm  # noqa: E402
from mojotaichi import stable_fluid as mtsf  # noqa: E402
from mojotaichi._lib import guarded  # noqa: E402

# Both sides get the same number of worker threads so the comparison is about
# the kernels, not about who grabbed more cores.  Override with
# MOJOTAICHI_BENCH_THREADS.
THREADS = int(
    os.environ.get("MOJOTAICHI_BENCH_THREADS", min(os.cpu_count() or 1, 16))
)
os.environ["MOJOTAICHI_THREADS"] = str(THREADS)

ti.init(
    arch=ti.cpu,
    default_fp=ti.f32,
    cpu_max_num_threads=THREADS,
    offline_cache=False,
    log_level=ti.ERROR,
)


def load_upstream(relative: str, names, namespace: dict) -> dict:
    """Exec the named top-level defs of an installed taichi file into `namespace`."""
    path = Path(ti.__file__).parent / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    keep = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.Module(body=keep, type_ignores=[])
    namespace["__file__"] = str(path)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


SF = load_upstream(
    "examples/simulation/stable_fluid.py",
    [
        "sample", "lerp", "bilerp", "backtrace", "TexPair", "advect", "apply_impulse",
        "divergence", "vorticity", "pressure_jacobi", "subtract_gradient",
        "solve_pressure_jacobi", "step",
    ],
    {
        "ti": ti, "np": np, "res": 512, "dt": 0.03, "dye_decay": 1 - 1 / (60 * 2),
        "f_strength": 10000.0, "force_radius": 256.0, "curl_strength": 0,
        "p_jacobi_iters": 20, "use_sparse_matrix": False, "debug": False,
    },
)

EF = load_upstream(
    "examples/simulation/eulerfluid2d.py",
    ["sample", "lerp", "bilerp", "TexPair", "advection", "curl", "divergence",
     "pressure_iteration", "apply_vel_bc"],
    {
        "ti": ti, "np": np,
        "eulerSimParam": {
            "shape": [512, 512], "dt": 1 / 60.0, "iteration_step": 20,
            "mouse_radius": 0.01, "mouse_speed": 125.0,
            "mouse_respondDistance": 0.5, "curl_param": 15,
        },
    },
)


def timeit(fn, repeat: int = 3, budget: float = 0.05) -> float:
    """Fastest of `repeat` timed calls, each repeated until `budget` has elapsed.

    The fast kernels here run in under a millisecond and this box carries other
    tenants, so a fixed small repeat count lands on a scheduler hiccup more often
    than not and made `advect (vec2)` alternate between 7ms and 17ms across runs.
    Spending a fixed time budget per call instead makes the reported figure the
    best case over many samples, which is the number the scheduling decisions
    need: the question is the fastest a kernel can go, not what it does on a
    contended sample.
    """
    fn()
    best = math.inf
    for _ in range(repeat):
        reps = 1
        t0 = time.perf_counter()
        fn()
        one = time.perf_counter() - t0
        while one * reps < budget and reps < 10_000:
            reps += 1
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def tfield(shape, comps=0):
    f = ti.Vector.field(comps, float, shape=shape) if comps else ti.field(float, shape=shape)
    f.from_numpy(np.zeros(tuple(shape) + ((comps,) if comps else ()), dtype=np.float32))
    return f


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


RES = 512
DT = 0.03
DYE = 1 - 1 / (60 * 2)
RNG = np.random.default_rng(0)


@case("stable_fluid advect (512x512, vec2)")
def _():
    v = (RNG.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    q = (RNG.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    out = np.zeros_like(q)
    vf, qf, nq = tfield((RES, RES), 2), tfield((RES, RES), 2), tfield((RES, RES), 2)
    vf.from_numpy(v), qf.from_numpy(q)

    def ours():
        mtsf.advect(v, q, out, dt=DT, dye_decay=DYE)

    def theirs():
        SF["advect"](vf, qf, nq)

    return ours, theirs


@case("stable_fluid advect (512x512, vec3 dye)")
def _():
    v = (RNG.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    q = (RNG.normal(size=(RES, RES, 3)) * 0.3).astype(np.float32)
    out = np.zeros_like(q)
    vf, qf, nq = tfield((RES, RES), 2), tfield((RES, RES), 3), tfield((RES, RES), 3)
    vf.from_numpy(v), qf.from_numpy(q)

    def ours():
        mtsf.advect(v, q, out, dt=DT, dye_decay=DYE)

    def theirs():
        SF["advect"](vf, qf, nq)

    return ours, theirs


@case("stable_fluid divergence (512x512)")
def _():
    v = (RNG.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    out = np.zeros((RES, RES), dtype=np.float32)
    vf, df = tfield((RES, RES), 2), tfield((RES, RES))
    vf.from_numpy(v)
    SF["velocity_divs"] = df  # upstream writes to this module global

    def ours():
        mtsf.divergence(v, out)

    def theirs():
        SF["divergence"](vf)

    return ours, theirs


@case("stable_fluid pressure_jacobi x20 (512x512)")
def _():
    p = (RNG.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    d = (RNG.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    ours_p = np.zeros((RES, RES), dtype=np.float32)
    ours_n = np.zeros((RES, RES), dtype=np.float32)
    pf, npf, df = tfield((RES, RES)), tfield((RES, RES)), tfield((RES, RES))
    pf.from_numpy(p), df.from_numpy(d)

    def ours():
        a, b = ours_p, ours_n
        for _ in range(20):
            mtsf.pressure_jacobi(a, b, d)
            a, b = b, a

    def theirs():
        for _ in range(20):
            SF["pressure_jacobi"](pf, npf)

    return ours, theirs


@case("stable_fluid vorticity (512x512)")
def _():
    v = (RNG.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    out = np.zeros((RES, RES), dtype=np.float32)
    vf, cf = tfield((RES, RES), 2), tfield((RES, RES))
    vf.from_numpy(v)
    SF["velocity_curls"] = cf  # upstream writes to this module global

    def ours():
        mtsf.vorticity(v, out)

    def theirs():
        SF["vorticity"](vf)

    return ours, theirs


@case("stable_fluid step() 512x512, 20 jacobi iters")
def _():
    v = (RNG.normal(size=(RES, RES, 2)) * 0.3).astype(np.float32)
    q = (RNG.normal(size=(RES, RES, 3)) * 0.3).astype(np.float32)
    imp = mojotaichi.impulse(mouse_dir=(1.0, 0.0), mouse_pos=(256.0, 256.0), color=(1.0, 0.3, 0.1))
    sim = mojotaichi.StableFluid(res=RES, p_jacobi_iters=20)
    sim.velocities_pair.cur[...] = v
    sim.dyes_pair.cur[...] = q
    pv = SF["TexPair"](tfield((RES, RES), 2), tfield((RES, RES), 2))
    pp = SF["TexPair"](tfield((RES, RES)), tfield((RES, RES)))
    pd = SF["TexPair"](tfield((RES, RES), 3), tfield((RES, RES), 3))
    pv.cur.from_numpy(v), pd.cur.from_numpy(q)
    divs, curls = tfield((RES, RES)), tfield((RES, RES))
    SF.update(velocities_pair=pv, pressures_pair=pp, dyes_pair=pd, velocity_divs=divs,
              velocity_curls=curls)

    def ours():
        sim.step(imp)

    def theirs():
        SF["step"](imp)

    return ours, theirs


@case("eulerfluid2d advection (512x512, vec3 colour)")
def _():
    v = (RNG.normal(size=(RES, RES, 2)) * 0.05).astype(np.float32)
    q = (RNG.normal(size=(RES, RES, 3)) * 0.3).astype(np.float32)
    out = np.zeros_like(q)
    vf, qf, nq = tfield((RES, RES), 2), tfield((RES, RES), 3), tfield((RES, RES), 3)
    vf.from_numpy(v), qf.from_numpy(q)

    def ours():
        mte.advection(v, q, out, dt=1 / 60.0)

    def theirs():
        EF["advection"](vf, qf, nq)

    return ours, theirs


@case("eulerfluid2d pressure_iteration x20 (512x512, mojo only)")
def _():
    d = (RNG.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    p = (RNG.normal(size=(RES, RES)) * 0.3).astype(np.float32)
    a, b = np.zeros((RES, RES), np.float32), np.zeros((RES, RES), np.float32)
    df, pf, npf = tfield((RES, RES)), tfield((RES, RES)), tfield((RES, RES))
    df.from_numpy(d), pf.from_numpy(p)

    def ours():
        x, y = a, b
        for _ in range(20):
            mte.pressure_iteration(d, x, y)
            x, y = y, x

    def theirs():
        EF["pressure_iteration"](df, pf, npf)

    return ours, None  # reads i + 1 off the last row: upstream refuses on CPU


@case("eulerfluid2d full frame (advect+curl+20 jacobi+project, mojo only)")
def _():
    sim = mojotaichi.EulerFluid(shape=(RES, RES))
    px = py = 0

    def ours():
        nonlocal px, py
        px, py = sim.step(px, py, 0.4, 0.6)

    return ours, None  # divergence/pressure_iteration go off the grid upstream


@case("mpm88 substep (8192 particles, 128x128 grid)")
def _():
    n_particles, n_grid = 8192, 128
    dx = 1 / n_grid
    p_vol = (dx * 0.5) ** 2
    sim = mojotaichi.MPM88(n_particles=n_particles, n_grid=n_grid).init(seed=0)
    up = load_upstream(
        "examples/simulation/mpm88.py", ["substep"],
        {"ti": ti, "n_particles": n_particles, "n_grid": n_grid, "dx": dx, "dt": 2e-4,
         "p_vol": p_vol, "p_rho": 1, "p_mass": p_vol, "E": 400, "gravity": 9.8, "bound": 3},
    )
    fx = ti.Vector.field(2, float, shape=(n_particles,))
    fv = ti.Vector.field(2, float, shape=(n_particles,))
    fC = ti.Matrix.field(2, 2, float, shape=(n_particles,))
    fJ = ti.field(float, shape=(n_particles,))
    fgv = ti.Vector.field(2, float, shape=(n_grid, n_grid))
    fgm = ti.field(float, shape=(n_grid, n_grid))
    # taichi needs its own fields; both sides evolve independently from the
    # same initial state, which is all a timing comparison needs
    fx.from_numpy(sim.x), fv.from_numpy(sim.v), fC.from_numpy(sim.C), fJ.from_numpy(sim.J)
    up.update(x=fx, v=fv, C=fC, J=fJ, grid_v=fgv, grid_m=fgm)

    def ours():
        sim.substep()

    def theirs():
        up["substep"]()

    return ours, theirs


def main() -> None:
    print(f"machine: {platform.platform()}", flush=True)
    print(f"python {platform.python_version()}, numpy {np.__version__}, taichi {ti.__version__}")
    print(f"cpus visible: {os.cpu_count()}, worker threads per kernel: {THREADS}\n", flush=True)
    print(f"| {'case':<52}| {'mojo-taichi':>12}| {'taichi':>12}| {'ratio':>8}|")
    print(f"|{'-' * 52}|{'-' * 12}:|{'-' * 12}:|{'-' * 8}:|")
    rows = []
    for name, build in CASES:
        print(f"running {name} ...", flush=True)
        ours_fn, theirs_fn = build()
        ours = timeit(ours_fn)
        rows.append((name, ours, None if theirs_fn is None else timeit(theirs_fn)))
        theirs = rows[-1][2]
        if theirs is None:
            print(f"| {name:<52}| {ours * 1e3:>10.2f}ms | {'n/a':>12} | {'-':>8} |")
        else:
            flag = "" if ours <= theirs else "  (mojo slower)"
            print(f"| {name:<52}| {ours * 1e3:>10.2f}ms | {theirs * 1e3:>10.2f}ms | "
                  f"{theirs / ours:>7.2f}x |{flag}")
    compared = [(a, b) for _, a, b in rows if b is not None]
    fast = sum(1 for a, b in compared if a <= b)
    print(f"\nmojo-taichi is at least as fast on {fast} of {len(compared)} comparable cases; "
          f"{len(rows) - len(compared)} case(s) are mojo-taichi only.")

    scheduling_report()


def scheduling_report() -> None:
    """Why each kernel picks its serial or parallel path.

    `parallelize` builds and tears down a worker pool on every call.  This
    measures what that costs and whether it is worth paying for each kernel, so
    the `PAR_MIN` thresholds in `src/taichi/runtime.mojo` are reproducible from
    the benchmark rather than asserted in a comment.
    """
    print("\n## scheduling: serial vs parallel per kernel (512x512, ms)", flush=True)
    res = 512
    rng = np.random.default_rng(0)
    v = (rng.normal(size=(res, res, 2)) * 0.5).astype(np.float32)
    q = (rng.normal(size=(res, res, 3)) * 0.1).astype(np.float32)
    q1 = np.zeros_like(q)
    s1 = np.zeros((res, res), np.float32)
    pf = np.zeros((res, res), np.float32)
    v2 = np.zeros((res, res, 2), np.float32)
    d1 = np.zeros((res, res), np.float32)

    kernels = [
        ("stable_fluid.advect (vec3)", lambda nw: mtsf.advect(v, q, q1, nworkers=nw)),
        ("stable_fluid.divergence", lambda nw: mtsf.divergence(v, s1, nworkers=nw)),
        ("stable_fluid.vorticity", lambda nw: mtsf.vorticity(v, s1, nworkers=nw)),
        ("stable_fluid.pressure_jacobi", lambda nw: mtsf.pressure_jacobi(pf, s1, s1, nworkers=nw)),
        ("stable_fluid.subtract_gradient", lambda nw: mtsf.subtract_gradient(v, pf, nworkers=nw)),
        ("eulerfluid2d.advection", lambda nw: mte.advection(v2, q, q1, nworkers=nw)),
        ("eulerfluid2d.divergence", lambda nw: mte.divergence(v2, d1, nworkers=nw)),
    ]
    print(f"| {'kernel':<34}| {'serial':>9}| {'parallel':>9}| {'worth it':>9}|")
    print(f"|{'-' * 34}|{'-' * 9}:|{'-' * 9}:|{'-' * 9}:|")
    for name, f in kernels:
        ser = timeit(lambda: f(1)) * 1e3
        par = timeit(lambda: f(THREADS)) * 1e3
        # A parallel launch costs ~2.5 ms on this toolchain; call it a win only
        # when the parallel path is faster by more than that launch cost, so a
        # noise-level difference does not read as "parallelise this".
        worth = "yes" if ser - par > 2.0 else "no"
        print(f"| {name:<34}| {ser:>7.2f}ms | {par:>7.2f}ms | {worth:>9} |", flush=True)

    print("\n## worker scaling, stable_fluid.advect (512x512, ms)", flush=True)
    print(f"| {'workers':>8}| {'advect':>9}|")
    print(f"|{'-' * 8}:|{'-' * 9}:|")
    for nw in (1, 4, 8, 16, 32, 48, 64, os.cpu_count() or 1):
        print(f"| {nw:>8}| {timeit(lambda: mtsf.advect(v, q, q1, nworkers=nw)) * 1e3:>7.2f}ms |",
              flush=True)


if __name__ == "__main__":
    main()
