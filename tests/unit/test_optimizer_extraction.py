"""Canonical extraction preserves absent values and shared path traffic."""

from types import SimpleNamespace

import numpy as np

from service_placement.utils.ilp_solvers import arc_model_arrays, path_model_arrays


def test_absent_optimizer_values_remain_unusable():
    x, y, q = arc_model_arrays(
        {(0, 0): SimpleNamespace(value=None)},
        {(0, 0): SimpleNamespace(value=1)},
        {0: SimpleNamespace(value=None)},
        1,
        1,
        1,
        1,
    )
    assert np.isnan(x[0, 0]) and np.isnan(q[0])
    np.testing.assert_array_equal(y, [[1.0]])


def test_path_extraction_combines_weights_by_demand_and_arc():
    x, y, q = path_model_arrays(
        {0: 0.25, 1: 0.75, 2: 1.0},
        {(0, 0): 0, (0, 1): 0, (0, 2): 1},
        {0: 2, 1: 3},
        [[0, 1, 2], [0, 2], [0, 1, 2]],
        [0, 0, 1],
        {(0, 1): 0, (1, 2): 1, (0, 2): 2},
        2,
        3,
        1,
        3,
    )
    np.testing.assert_array_equal(x, [[0.25, 0.25, 0.75], [1, 1, 0]])
    np.testing.assert_array_equal(x.T @ q, [3.5, 3.5, 1.5])
    np.testing.assert_array_equal(y, [[0, 0, 1]])
    np.testing.assert_array_equal(q, [2, 3])


def test_empty_extraction_retains_matrix_dimensions():
    x, y, q = arc_model_arrays({}, {}, {}, 0, 2, 0, 3)
    assert x.shape == (0, 2) and y.shape == (0, 3) and q.shape == (0,)
