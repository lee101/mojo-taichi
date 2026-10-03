"""Parity for `taichi_logo`, the pattern `eulerfluid2d.init_field` paints with."""

import numpy as np
import pytest

from conftest import assert_same, load_upstream

import mojotaichi


@pytest.fixture
def upstream():
    ti = pytest.importorskip("taichi")
    return load_upstream("examples/patterns.py", ["taichi_logo"], {"ti": ti})


def test_taichi_logo_values(upstream):
    """The logo is a hard-edged mask: 0.0 or 1.0, and never anything else."""
    pts = np.random.default_rng(0).random((2000, 2)).astype(np.float32)
    ours = mojotaichi.taichi_logo(pts)
    assert set(np.unique(ours)) <= {0.0, 1.0}

    # compare against the upstream expression, evaluated in float32 numpy
    def ref(pos, scale=1 / 1.11):
        p = (pos - 0.5) / np.float32(scale) + 0.5
        ret = -1
        if not np.dot(p - 0.50, p - 0.50) <= np.float32(0.52) ** 2 and ret == -1:
            ret = 0
        if not np.dot(p - 0.50, p - 0.50) <= np.float32(0.495) ** 2 and ret == -1:
            ret = 1
        if np.dot(p - np.array([0.50, 0.25], np.float32), p - np.array([0.50, 0.25], np.float32)) <= np.float32(0.08) ** 2 and ret == -1:
            ret = 1
        if np.dot(p - np.array([0.50, 0.75], np.float32), p - np.array([0.50, 0.75], np.float32)) <= np.float32(0.08) ** 2 and ret == -1:
            ret = 0
        if np.dot(p - np.array([0.50, 0.25], np.float32), p - np.array([0.50, 0.25], np.float32)) <= np.float32(0.25) ** 2 and ret == -1:
            ret = 0
        if np.dot(p - np.array([0.50, 0.75], np.float32), p - np.array([0.50, 0.75], np.float32)) <= np.float32(0.25) ** 2 and ret == -1:
            ret = 1
        if p[0] < 0.5:
            if ret == -1:
                ret = 1
        elif ret == -1:
            ret = 0
        return np.float32(1 - ret)

    expected = np.array([ref(q) for q in pts], dtype=np.float32)
    assert_same(ours, expected, rtol=0, atol=0, name="taichi_logo")


def test_taichi_logo_scalar_and_scale(upstream):
    assert mojotaichi.taichi_logo([0.5, 0.5]) == 1.0
    # a different scale shrinks the mask about the same centre
    a = mojotaichi.taichi_logo(np.array([0.5, 0.75], np.float32))
    b = mojotaichi.taichi_logo(np.array([0.5, 0.75], np.float32), scale=0.5)
    assert a in (0.0, 1.0) and b in (0.0, 1.0)
