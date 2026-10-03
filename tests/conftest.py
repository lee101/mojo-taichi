import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np  # noqa: E402

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="session", autouse=True)
def _taichi_runtime():
    """One CPU runtime for the whole session: `ti.init` cannot be called twice."""
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f32,
        cpu_max_num_threads=int(os.environ.get("MOJOTAICHI_TEST_THREADS", "4")),
        offline_cache=False,
        log_level=ti.ERROR,
    )


def _upstream_path(relative: str) -> Path:
    return Path(ti.__file__).parent / relative


def load_upstream(relative: str, names, namespace: dict) -> dict:
    """Exec the named top-level defs / assignments of an installed taichi file.

    The upstream examples are scripts: importing them would run `ti.init`, build
    a 512x512 field set and open a GUI.  Taking the AST nodes instead gives the
    *actual installed kernel source*, unmodified, with the caller supplying the
    module globals it would have read.

    Call this per test, not once per session: a no-argument upstream kernel such
    as mpm88's `substep` captures its fields from these globals when Taichi
    compiles it, so a second call with different fields would silently reuse the
    first call's kernel.
    """
    path = _upstream_path(relative)
    tree = ast.parse(path.read_text(), filename=str(path))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in names for t in node.targets
        ):
            keep.append(node)
    missing = set(names) - {
        getattr(n, "name", None)
        or next((t.id for t in getattr(n, "targets", []) if hasattr(t, "id")), None)
        for n in keep
    }
    assert not missing, f"{relative} is missing {sorted(missing)}"
    module = ast.Module(body=keep, type_ignores=[])
    namespace["__file__"] = str(path)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def field(shape, comps: int = 0):
    """A float32 taichi field with `shape`; `comps` makes it a vector field."""
    if comps:
        f = ti.Vector.field(comps, float, shape=shape)
        f.from_numpy(np.zeros(tuple(shape) + (comps,), dtype=np.float32))
    else:
        f = ti.field(float, shape=shape)
        f.from_numpy(np.zeros(tuple(shape), dtype=np.float32))
    return f


def set_field(f, a):
    f.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def get_field(f):
    return f.to_numpy()


def assert_same(a, b, *, rtol=2e-4, atol=2e-5, name=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    err = np.abs(a - b)
    tol = atol + rtol * np.abs(b)
    bad = err > tol
    if bad.any():
        i = np.unravel_index(int(np.argmax(err - tol)), err.shape)
        raise AssertionError(
            f"{name}: {bad.sum()} of {err.size} values outside tolerance; "
            f"worst at {i}: ours={a[i]!r} upstream={b[i]!r} "
            f"(abs {err[i]:.3e}, tol {tol[i]:.3e})"
        )
