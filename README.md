# mojo-taichi

Taichi's parallel spatial-simulation kernels, ported to Mojo.

Upstream is the `taichi` package (v1.7.4). The ported algorithms are its
compute-bound grid and particle kernels: the Stam stable-fluids solver, the 2D
Euler fluid solver, and the MPM88 material-point solver, plus the `taichi_logo`
pattern the Euler solver paints its colour field with. They are the parts of
Taichi a host application actually spends its time in — the rest of Taichi is a
compiler, a type system and a renderer, which is not what a Mojo port is for.

Everything is float32 and keeps upstream's names, argument order and
arithmetic, so a call that reads

```python
advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
```

reads the same here.

## Coverage

Ported, in upstream's source order, in `src/taichi/`:

| upstream module | kernels / funcs ported |
| --- | --- |
| `taichi/examples/patterns.py` | `taichi_logo` |
| `taichi/examples/simulation/stable_fluid.py` | `sample`, `lerp`, `bilerp`, `backtrace`, `advect`, `apply_impulse`, `divergence`, `vorticity`, `pressure_jacobi`, `subtract_gradient`, `enhance_vorticity`, `copy_divergence`, `apply_pressure`, `solve_pressure_jacobi`, `step`, `reset` |
| `taichi/examples/simulation/eulerfluid2d.py` | `sample`, `lerp`, `bilerp`, `init_field`, `advection`, `curl`, `vorticity_projection`, `divergence`, `pressure_iteration`, `pressure_solve`, `pressure_projection`, `vel_with_boundary`, `p_with_boundary`, `c_with_boundary`, `apply_vel_bc`, `apply_p_bc`, `apply_c_bc`, `mouse_addspeed`, `mouse_interaction`, `advaction_step`, `voricity_step`, `pressure_step` |
| `taichi/examples/simulation/mpm88.py` | `substep`, `init` |

The Python-side steps (`solve_pressure_jacobi`, `pressure_solve`, `step`,
`advaction_step`, `voricity_step`, `pressure_step`, `mouse_interaction`) stay
where upstream put them: in Python, as loops over the kernels. They are
`StableFluid.step`, `EulerFluid.pressure_step` and friends, and the tests
compare them against the upstream functions exec'd from the installed package.

### Not covered, and why

- **`eulerfluid2d.pressure_projection`, and what that says about its four
  neighbours.** Its `p_with_boundary` calls read and write `i + 1` / `j - 1` on
  the border, and Taichi's CPU backend **segfaults** on it rather than raising
  (measured here, taichi 1.7.4 on x64). The port keeps the arithmetic and puts
  those border accesses in the field's guard, so the interior is well defined
  and is what the test compares.
  `curl`, `divergence`, `pressure_iteration` and `vorticity_projection` read a
  neighbour one cell outside the field in the same unclamped way, and *those* do
  run on Taichi's CPU backend — the out-of-bounds read lands in Taichi's own
  surrounding allocation instead of faulting. They are checked against a real
  Taichi run over the interior, where the value read is defined for both sides.
- **`stable_fluid`'s sparse-matrix pressure path** (`-S`, `fill_laplacian_matrix`
  + `ti.linalg.SparseSolver`). It is a wrapper around Taichi's own sparse
  factorisation, not a kernel; the Jacobi path is the one the example uses by
  default and is ported.
- **The GUI** (`ti.GUI`, `MouseDataGen`, `show`). Not a kernel, and the port has
  no window system. `impulse()` builds the same 8-float `mouse_data` layout and
  `EulerFluid.mouse_interaction` applies the same impulse maths.
- **The rest of the `taichi` package** — the compiler, the sparse and dense
  linear algebra, `taichi.math`, `taichi.ad`, the renderers. Out of scope for a
  spatial-kernel port; the port is the kernels, not the language.
- **`examples/simulation/mpm99.py` and the `taichi_mpm` solvers.** Their
  substep needs a 2x2 SVD whose float32 result is not reproducible against
  Taichi's, so a parity test against upstream would be meaningless. MPM88 is
  the SVD-free sibling and is ported in full.

## Install

```bash
pixi install
pixi run build      # mojo build --emit shared-lib -> dist/libmojo-taichi.so
pixi run test       # pytest, parity against the installed upstream taichi
pixi run bench      # benchmarks, prints a markdown table
```

`pixi.toml` pins `mojo 1.2.0.dev2026092905` and the matching `max 26.7.0.dev2026092905`
package (which provides `max.algorithm.parallelize`; the `std` path no longer has
it), and pulls `taichi 1.7.4` from PyPI as a test and benchmark dependency.

## Usage

A 256x256 stable-fluid solve, ten steps of it, with one stir:

```python
import numpy as np
import mojotaichi

# 256x256 stable fluid, 50 Jacobi iterations per pressure solve
sim = mojotaichi.StableFluid(res=256, p_jacobi_iters=50)

# a stir: a unit impulse at the centre, injecting an orange dye
impulse = mojotaichi.impulse(
    mouse_dir=(1.0, 0.0), mouse_pos=(128.0, 128.0), color=(1.0, 0.35, 0.05)
)
for _ in range(10):
    sim.step(impulse)

dye = sim.dyes_pair.cur             # (256, 256, 3) float32
velocity = sim.velocities_pair.cur  # (256, 256, 2) float32
print(dye.shape, float(dye.sum()), np.abs(velocity).max())
```

```
(256, 256, 3) 2503.9931640625 544.2346
```

The 2D Euler fluid solver and the MPM88 solver work the same way:

```python
euler = mojotaichi.EulerFluid(shape=(256, 256))
px = py = 0
for _ in range(20):
    px, py = euler.step(px, py, 0.4, 0.6)
print(euler.colorField.shape, float(euler.colorField.sum()))

mpm = mojotaichi.MPM88(n_particles=8192, n_grid=128).init(seed=0)
mpm.run(50)
print(mpm.x.shape, mpm.J.min(), mpm.J.max())
```

```
(256, 256, 3) 122472.9375
(8192, 2) 1.0 1.0
```

Individual kernels are exposed as module-level functions with upstream's names.
`dt` and `dye_decay` default to upstream's values:

```python
from mojotaichi import stable_fluid

v = np.zeros((64, 64, 2), dtype=np.float32)
dye = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
next_dye = np.zeros_like(dye)
stable_fluid.advect(v, dye, next_dye)
stable_fluid.divergence(v, np.zeros((64, 64), dtype=np.float32))
print(next_dye.shape, float(next_dye.sum()))
```

```
(64, 64, 3) 6098.083984375
```

Every snippet above runs top to bottom in one session from the repository root
with `PYTHONPATH=python`.
## Benchmarks

Every figure below is the **median of four `pixi run bench` runs** on the machine
named in the block below it, taken on the day this README was written. Both
sides run the same source: the Mojo kernels from `src/`, and the Taichi kernels
exec'd out of the installed `taichi` package, so this compares two
implementations of one algorithm rather than two algorithms. Within a run each
figure is already the fastest of three timed windows, each window repeating the
call until 50 ms has elapsed.

```
machine: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
python 3.13.14, numpy 2.5.1, taichi 1.7.4, mojo 1.2.0.dev2026092905
cpus visible: 72, worker threads per kernel: 16
```

| case | mojo-taichi | taichi | ratio |
| --- | ---: | ---: | ---: |
| stable_fluid advect (512x512, vec2) | 20.5ms | 2.8ms | 0.14x |
| stable_fluid advect (512x512, vec3 dye) | 23.1ms | 2.9ms | 0.13x |
| stable_fluid divergence (512x512) | 2.6ms | 0.56ms | 0.21x |
| stable_fluid pressure_jacobi x20 (512x512) | 4.0ms | 10.8ms | **2.7x** |
| stable_fluid vorticity (512x512) | 1.8ms | 0.43ms | 0.24x |
| stable_fluid step() 512x512, 20 jacobi iters | 56.1ms | 20.3ms | 0.36x |
| eulerfluid2d advection (512x512, vec3 colour) | 5.8ms | 1.05ms | 0.18x |
| eulerfluid2d pressure_iteration x20 (512x512) | 15.7ms | n/a | - |
| eulerfluid2d full frame (advect+curl+20 jacobi+project, 512x512) | 38.1ms | n/a | - |
| mpm88 substep (8192 particles, 128x128 grid) | 0.89ms | 1.14ms | 1.28x |

`ratio` is `taichi / mojo-taichi`, so above 1.0 means the port is faster.
`mojo-taichi` wins two of the eight cases both sides can run, and the two wins
are the two kernels that vectorise: the five-point stencil on a scalar field,
and MPM88's whole-field grid clear.

`n/a` means upstream Taichi's CPU backend cannot run that kernel: it indexes one
cell past the last row and column (see [Not covered](#not-covered-and-why)).

**Variance on this box is high**, so this table is medians rather than a single
run and the ratios matter more than the absolutes. `pixi run bench` holds a
machine-wide `flock` so a concurrent job cannot distort a run, but it cannot stop
other tenants. Across the four runs the `mojo-taichi` column moved by 3-5% on
most rows, while the `taichi` column moved by up to 40% — `pressure_jacobi`
measured 6.42, 9.61, 10.53 and 11.18 ms for the same binary, which is why its
ratio is quoted as 2.7x from the median rather than from whichever run looked
best. One run had a 4 ms scheduler hiccup land on `stable_fluid.advect` and flip
its serial-vs-parallel row to "yes"; the other three said "no", and the two
configurations are within 1% of each other.

### Reading the table

**`pressure_jacobi` is the one kernel shaped to vectorise.** It is the only one
that reads a scalar field on all five of its taps, so its row interior is four
contiguous `VEC_W`-wide loads at constant offsets. Every other kernel is either
per-node arithmetic over a node-dependent address (`advect`, four bilinear
samplings each) or a stride-2 read of one component of an interleaved vector
field (`divergence`, `vorticity`, `subtract_gradient`), and neither has a
contiguous load that produces it on this toolchain. That is the split between
the rows that beat Taichi and the rows that do not.

**The scalar stencils are four to eight times slower than Taichi.** Their bodies
are upstream's, and the two-level loop removed the flat-loop index recovery, but
the arithmetic is per-node with no vector form, so what is left is a scalar loop
against Taichi's vectorised one.

**`mpm88` is a modest win, and it is the grid clear that wins.** Its P2G scatter
is a read-modify-write over a 3x3 neighbourhood, which has no vector form either
way; what the port does better is the whole-field clears around it.

### Vectorisation

`VEC_W` is `simd_width_of[DType.float32]()`, which is **8** on the pinned
toolchain. Each vectorisable kernel sweeps a row `VEC_W` columns at a time with
`unsafe_load[width=VEC_W]` / `unsafe_store`, and finishes the remainder in a
scalar tail. `tests/test_vector_paths.py` and `tests/test_exports.py` both read
`VEC_W` from the compiled library through `mt_simd_width` and build grid widths
around it rather than hard-coding 8, so a toolchain reporting a different width
fails there instead of silently skipping every vector body.

Only three kernels have a vector body at all: `pressure_jacobi`,
`copy_divergence` and `apply_pressure`. The first is the five-point stencil on
a scalar field; the other two are whole-field copies with no clamp, so every
node in a row is eligible. `pressure_jacobi` additionally routes a row's first
and last `VEC_W - 1` columns through the scalar body, because upstream's
`sample` clamps `j - 1` and `j + 1` to the grid and a vector of consecutive
columns would re-derive that clamp per lane.

Everything else is scalar, and the reasons are specific:

- `advect` (both solvers) and `apply_impulse` do per-node arithmetic at a
  node-dependent address: `advect` takes four bilinear samplings per node, each
  four scalar loads, and `apply_impulse` two exponentials. There is no load to
  widen.
- `divergence`, `vorticity` and `subtract_gradient` read the *interleaved*
  velocity field, so each of their terms is one component at a stride of two
  across consecutive columns.
- `curl` and `divergence` in `eulerfluid2d` read one velocity component with the
  same stride-2 shape.
- `vorticity_projection` and `pressure_projection` are guarded by a per-node
  predicate or write to the field they read.

Two properties of this toolchain were verified by compiling and running probes
rather than read out of the docs:

- **`Pointer.store` of a `SIMD` value writes exactly the `VEC_W` elements it is
  given, at the index it is given, and does not require a 32-byte-aligned
  base.** `guarded_field` hands the kernels a view offset by `(2 * stride + 2) *
  comps` floats, which is not 32-byte aligned. This is what lets the guarded
  fields vectorise at all.
- **`SIMD.gather` is broken for a runtime index vector.** It returns zero rather
  than the addressed elements, silently. Nothing in this port uses `gather`;
  every load is contiguous, and where a clamp cannot be expressed as a constant
  offset the kernel runs its scalar body. `tests/test_exports.py` asserts that
  every node of `pressure_jacobi`, `copy_divergence` and `apply_pressure` is
  written at a width that is not a multiple of `VEC_W`, so a gather that read
  nothing could not pass.

### Scheduling: what `parallelize` costs

`max.algorithm.parallelize` is **not** a cheap fork-join on this toolchain. The
fork-join branch stays in every kernel in upstream's `if ... else` shape,
guarded by `PAR_MIN` in `src/taichi/vector.mojo`, which is set past every size
the library accepts — so the branch is present and explicit rather than deleted,
and `nworkers` is documented as inert at that setting rather than silently
ignored.

`pixi run bench` prints the measurement the policy is read off. Medians of the
same four runs, at 512x512:

| kernel | serial | parallel (16) | parallel wins? |
| --- | ---: | ---: | :--- |
| stable_fluid.advect (vec3) | 22.5ms | 22.5ms | no |
| stable_fluid.divergence | 2.2ms | 2.2ms | no |
| stable_fluid.vorticity | 1.5ms | 1.5ms | no |
| stable_fluid.pressure_jacobi | 0.19ms | 0.18ms | no |
| stable_fluid.subtract_gradient | 1.52ms | 1.53ms | no |
| eulerfluid2d.advection | 5.4ms | 5.4ms | no |
| eulerfluid2d.divergence | 0.65ms | 0.65ms | no |

**Nothing in this port is parallelised, and that is the measured answer.** No
covered kernel gains anything from the worker pool at 512x512 — the two columns
agree to within 1% on every row. The largest remaining kernel, `advect`, is a
body with no vector form, and the largest vectorised kernel,
`pressure_jacobi`, takes 0.19 ms, which is far too little work to pay for a
fork-join launch.

A parallel launch that loses to the serial path is worse than no parallelism, so
this is the honest outcome: **a library of memory-bound stencils at these sizes
does not benefit from a `parallelize` launch on this toolchain.** If a future
toolchain makes the pool cheap, `PAR_MIN` is the one number that has to move.
`nworkers=` and `$MOJOTAICHI_THREADS` are still read and passed through to every
kernel, so lowering `PAR_MIN` is a one-number change rather than a rework of the
bindings.

### What was tried and reverted

Recording these because the negative results are as load-bearing as the positive
ones, and because each was a real change that had to be undone:

- **Vectorising MPM88's grid clear and normalise.** Reverted. The grid is
  `n_grid` squared and at the size this port is called with (128x128) it is
  192 KiB, which stays in L2 across both passes, so neither sweep is a
  bandwidth problem. The normalise sweep additionally cannot be vectorised as
  written: `m` is zero on every node no particle reached, so its `m > 0` guard
  is per node, and clamping the divide into a vector expression produced NaN
  where upstream leaves the value alone — which showed up immediately as 11
  failing parity tests.
- **Vectorising `mpm88.init`.** Reverted. `x` is an element-wise map, but `v` is
  not: its y component is -1 for every particle, which in the interleaved layout
  is a per-lane sign mask rather than a splat. `init` is one pass over
  `n_particles` and is in no benchmark.
- **Vectorising `eulerfluid2d`'s `curl`, `divergence` and `pressure_iteration`.**
  Reverted after they failed parity. `curl`'s y term is a stride-2 read of the
  interleaved velocity field; `divergence` and `pressure_iteration` read one cell
  past the field on the last row and column, which the two-cell guard makes safe
  in memory but which is not the same computation as a vector body that stays
  inside the view.
- **A clamped SIMD `gather` form of the five-point stencils.** Reverted on the
  measurement: it was faster only because it read nothing.
- **`parallelize` for `advect` and every other kernel.** Reverted, per the
  scheduling table above.
- **No GPU path.** Every covered kernel is a five-point stencil or an
  element-wise map over node-dependent addresses, so there is no arithmetic
  intensity to exploit and no contiguous-load formulation to feed a GPU. A
  `max.gpu.host` path would need the field copied in, the kernel launched and the
  result copied back for every kernel in a step, and the solvers call several
  kernels per frame on buffers that stay on the host. `max.gpu.host` is present
  and compiles on this toolchain; it is not the right tool for these kernels.
  The box has an RTX 5090 with 32607 MiB total and 17056 MiB in use at the time
  of writing, so nothing here was skipped for lack of a GPU.

## How it works

**FFI.** `src/capi.mojo` is the single compilation unit. It is a thin layer of
`@export("mt_...")` functions with `abi("C")`; each one rebuilds pointers from
`Int` addresses and calls the ported kernel, which is a plain `def` taking
`Pointer[Float32, AnyOrigin[mut=True]]` and extents. `@export` rejects parametric
functions and a pointer with an inferred origin is parametric, so the pointer
type is written out explicitly and every buffer crosses as an integer address.
Nothing in the library allocates: the caller owns every buffer, including
scratch, so there is nothing to leak and nothing to keep alive.

The Python side (`python/mojotaichi/_lib.py`) loads `dist/libmojo-taichi.so`
with `ctypes`, rebuilding it when `src/` is newer, and passes
`array.ctypes.data` directly. A C-contiguous float32 array is never copied.

**Argument validation.** Every entry point checks its buffers before crossing:
they must be numpy arrays, float32, C-contiguous, and of the shape the kernel
indexes. Nothing is converted on the way in. That is deliberate: these kernels
write in place, so silently converting a float64 or strided array would send
the result into a temporary copy and leave the caller's array untouched, with
no indication that anything had happened. A wrong buffer raises `TypeError`; a
wrong shape raises `ValueError` naming both shapes. MPM88 derives its particle
count from `len(x)`, so `v`, `C` and `J` are checked against it rather than
trusted.

**Memory layout.** Everything is float32, exactly like the `ti.field(float, ...)`
it mirrors, so no precision is added or lost at the boundary. A field of shape
`(n0, n1)` is row-major, element `(i, j)` at `i * n1 + j`; a vector field of
rank `r` interleaves components, element `(i, j)` at `(i * n1 + j) * r`. That is
what `field.to_numpy()` returns, so a field can be handed to the port and back
without a repack.

**Guards.** `eulerfluid2d`'s `curl`, `vorticity_projection`, `divergence`,
`pressure_iteration` and `pressure_projection` index `i + 1` / `j + 1` on the
first and last row and column without clamping. Upstream only defines that on a
GPU. On Taichi's CPU backend `pressure_projection` segfaults on it and the other
four read into whatever adjoins the field.
`mojotaichi.guarded_field(shape, comps)` allocates the same allocation with two
cells of slack on every side and returns the logical `(n0, n1, r)` view, so the
arithmetic is unchanged and those accesses land in slack instead of in the next
allocation. `EulerFluid` allocates its fields that way. Pass plain arrays if you
want to, but then a border access is your problem — through a raw pointer here
that is a wild read or write, not a Taichi-style refusal.

**MPM88's grid node is clamped.** Upstream's P2G scatter and G2P gather index
`grid_v[base + offset]` with no clamp, so a particle that has left `[0, 1]`
addresses memory off the field — fine to leave undefined on a GPU, a wild
write through a raw pointer here. The port clamps the node to the grid. For
in-domain particles this is a no-op and the parity tests are unaffected; an
escaped particle piles onto the border node instead of corrupting the heap.

**Parallelism.** None of it, and that is measured rather than assumed — see
[Scheduling](#scheduling-what-parallelize-costs). The fork-join branch stays in
every kernel in upstream's `if ... else` shape, guarded by `PAR_MIN` in
`src/taichi/vector.mojo`, which is set past every size the library accepts. The
scatter in MPM's P2G stays serial regardless, as it is upstream: it is a
read-modify-write over a 3x3 neighbourhood.

**Grid traversal.** Upstream's `for i, j in field` becomes a two-level loop: an
outer sweep over rows and an inner sweep of `VEC_W` columns. The obvious
transliteration is a flat loop over `k = i * n1 + j` with `i = k // n1`,
`j = k % n1`, and that costs more than the stencil it indexes, because the
division is not strength-reduced to an increment on this toolchain. Keeping the
row index in a register leaves the neighbour offsets constant within the inner
loop, which is also what makes a vector load legal there.

## Testing

```bash
pixi run test
```

122 tests in seven modules. Most of them are real parity tests:
`tests/conftest.py` pulls the actual kernel source out of the *installed*
`taichi` package (the examples are scripts that would run `ti.init`, build a
512x512 field set and open a GUI, so their top-level defs are exec'd with the
caller supplying the module globals they read) and run it on Taichi's CPU
backend next to the port on the same inputs.

The exceptions, and what they do instead:

- `eulerfluid2d.pressure_projection` is checked against a NumPy transcription
  of its own source expressions over the interior, because Taichi's CPU backend
  segfaults on it (see [Not covered](#not-covered-and-why)). `curl`,
  `divergence`, `pressure_iteration` and `vorticity_projection` *can* run on
  Taichi's CPU backend, so they are checked against a real upstream run **and**
  against the transcription, over the interior where every tap is in-field for
  all three implementations.
- `mpm88.init` takes its uniform draws as an argument instead of calling
  `ti.random()` (whose stream is not reproducible), so it is checked against the
  upstream formula, not against upstream.
- `tests/test_memory_safety.py` holds the FFI-boundary contracts (28 tests).
- `tests/test_exports.py` calls every `@export`ed symbol by name (29 tests).

Those four Euler kernels also get an `isfinite` assertion over the *whole*
field. They read one cell past the edge, so a test that passes a plain array
instead of a `guarded_field` reads off the end of the allocation; the assertion
is what makes that a failure rather than a silent garbage value that happens to
match the reference over the interior.

The Python-side steps are differential too, since the coverage table above lists
them as ported: `advaction_step` and `voricity_step` run against the upstream
functions exec'd from the installed package, from the same starting state, and
`pressure_solve` runs against upstream's — covering the part of `pressure_step`
that Taichi's CPU backend can execute. `pressure_step`'s projection half is
checked against the transcription, because upstream's own `pressure_projection`
segfaults. The rest of the suite is behavioural rather than differential: Jacobi
converging to the Poisson fixed point, projection making the field
divergence-free, mass conservation through P2G, boundary clamping, and a stable
multi-frame Euler step.

### Every exported kernel is called by name

`tests/test_exports.py` (29 tests) bypasses the Python wrappers and calls each of
the 26 `mt_*` symbols in `src/capi.mojo` directly through `ctypes`, asserting on
real output. The rest of the suite reaches the kernels through the wrappers,
which is the right way to test behaviour but the wrong way to test wiring: a
transposed signature, a swapped component stride or a crossed argument pair
inside `capi.mojo` still produces a plausible result through a wrapper, because
the wrapper is what supplies the arguments in the first place. Calling the
symbols directly makes each kernel's reachability, argument order and output
independent of the wrapper that owns it. A transposed pair in `capi.mojo` was
injected to confirm the module fails on it.

The module also asserts that `capi.mojo` and `_SIGNATURES` export the same set
of names and that every one of them is present in the shared library, so a
symbol added on one side and forgotten on the other is a test failure rather
than an unchecked ctypes call.

### FFI-boundary contracts

`tests/test_memory_safety.py` (28 tests) holds the argument contracts that are
the difference between a clear Python error and silent memory corruption:

- a float64, strided, wrong-shaped or non-array buffer is rejected rather than
  silently converted, which would have sent the result into a copy and left the
  caller's array untouched;
- **component width is checked**, not just the square extent every field shares.
  A kernel fixes its component stride at compile time, so `apply_impulse` on a
  scalar or vec2 dye field, or `advect` with a vec3 `qf` and a vec2 `new_qf`,
  writes past the end of the buffer. Both were accepted and both corrupted the
  heap; the tests now pin the rejection and, through a canary allocation, that
  the accepted path writes exactly the field it was given and no further;
- the scalar stencils reject a vector field, which they would otherwise stride
  over and touch a third of;
- an MPM88 particle outside `[0, 1]` still scatters and gathers inside the
  allocation, and MPM88's particle arrays are checked against each other's
  length;
- `taichi_logo` refuses a position that is not a 2-vector.

### Vector paths

`tests/test_vector_paths.py` (15 tests) covers the parts of the kernels that a
single even grid size never reaches:

- `VEC_W` is read from the compiled library through `mt_simd_width` rather than
  hard-coded, so a toolchain reporting a different width fails there instead of
  silently skipping every vector body.
- The scalar tail is exercised at widths on both sides of a `VEC_W` boundary
  (8, 9, 15, 16, 17, 31, 32, 33, 64, 65) and against a NumPy transcription of
  upstream, so a tail that skips, repeats or mis-clamps a node cannot pass.
- A grid narrower than one vector is checked on its own, since that is the case
  where the vector loop never runs and only the scalar body can be right.
- The vec2 stencils, which are scalar by design because their terms are stride-2
  reads of an interleaved field, are pinned against the same transcription at
  widths straddling a `VEC_W` boundary.
- The worker count is varied across a grid far larger than `PAR_MIN`, so the
  fork-join and serial drivers are held to bit-identical output.

## Fidelity notes

Documented divergences, all forced by the language or by upstream's own
undefined behaviour:

- `ti.template()` does not exist, so a function upstream instantiates per field
  rank becomes one function per rank: `sample`, `lerp`, `bilerp` and `advect`
  gain `_v2` / `_v3` siblings.
- Upstream's module-level constants are arguments, with upstream's values as
  defaults (`res=512`, `dt=0.03`, `dye_decay=1 - 1/(60*2)`, `f_strength=10000`,
  `force_radius=res/2`, `curl_strength=0`, `p_jacobi_iters=500`,
  `eulerSimParam`, and MPM88's `dx`, `E`, `gravity`, `bound`).
- `ti.static(ti.ndrange(3, 3))` is a plain nested `for` loop: Mojo has no
  unrolled field iteration.
- The `for i, j in field` iteration becomes the two-level loop described under
  Grid traversal.
- `mpm88.init` takes its uniforms as an argument.
- Guarded fields, as described above.
- Every kernel takes the serial driver rather than the worker pool, because no
  covered kernel gains from the pool at the sizes the library accepts.
  The body is upstream's and the branch is upstream's; only the threshold moved.
  This is a performance decision, not a semantic one, and it is measured in
  [Scheduling](#scheduling-what-parallelize-costs).
- The stencils whose terms are stride-2 reads of an interleaved vector field keep
  upstream's scalar body, because there is no contiguous load that produces such
  a term on this toolchain. Which kernels those are is listed under
  [Vectorisation](#vectorisation).
- Mojo contracts multiply-add into FMA where Taichi's backend does not always;
  results agree to float32 rounding, not bit for bit. The tests assert
  numerical parity with tolerances at that level rather than equality.

## Licence

MIT, (c) Lee Penkman. See `LICENSE`. Upstream Taichi is Apache-2.0; this port
is not a derivative of its source, which is read at test and benchmark time
only, to check parity.
