"""Diagnostics exposed to API users use the same reference checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from service_placement import Solution, SolutionValidationError, validate
from tests.helpers import make_problem, route_solution


def test_shared_traffic_latency_report_explains_counterexample():
    problem = make_problem()
    problem.latencies = [3, 3]
    solution = Solution(*route_solution(problem, [[0, 1], [0, 1]], [1], [2, 2]))
    report = validate(problem, solution)
    assert not report.feasible
    assert [(v.constraint, v.demand, v.observed, v.limit) for v in report.violations] == [
        ("latency", 0, 4, 3),
        ("latency", 1, 4, 3),
    ]
    with pytest.raises(SolutionValidationError, match="latency"):
        report.raise_if_infeasible()


def test_collects_independent_capacity_resource_and_flow_violations():
    problem = make_problem()
    problem.network_graph.available_resources = [1, 1]
    problem.network_graph.capacities = [1, 1]
    problem.min_flows = [3, 3]
    problem.max_flows = [3, 3]
    solution = Solution(*route_solution(problem, [[0, 1], [0, 1]], [1], [2, 4]))
    violations = validate(problem, solution).violations
    by_constraint = {v.constraint: v for v in violations}
    assert {"capacity", "resources", "minimum_flow", "maximum_flow"} <= by_constraint.keys()
    assert by_constraint["capacity"].observed == 6
    assert by_constraint["resources"].node == 1
    assert by_constraint["minimum_flow"].demand == 0
    assert by_constraint["maximum_flow"].demand == 1


def test_dimension_errors_are_reported_before_matrix_operations():
    report = validate(make_problem(), Solution([], [[1]], []))
    assert len(report.violations) == 3
    assert all(v.constraint == "dimensions" for v in report.violations)
    assert report.violations[0].observed == (0,)
    assert report.violations[0].limit == (2, 2)


def test_malformed_numeric_arrays_are_reported():
    candidate = SimpleNamespace(x="bad", y=[[0, 1]], q=[2, 2])
    assert validate(make_problem(), candidate).violations[0].constraint == "numeric"


def test_absence_of_solution_has_a_report():
    assert validate(make_problem(), None).violations[0].constraint == "no_solution"


@pytest.mark.parametrize("atol", [-1, float("nan"), float("inf"), "bad", None])
def test_invalid_tolerance_rejected_even_without_candidate(atol):
    with pytest.raises(ValueError, match="atol"):
        validate(make_problem(), None, atol=atol)


def test_wrong_solution_type_has_clear_error():
    with pytest.raises(TypeError, match="x, y and q"):
        validate(make_problem(), object())


def test_solution_owns_arrays_and_can_be_validated_without_mutation():
    problem = make_problem()
    x, y, q = route_solution(problem, [[0, 1], [0, 1]], [1], [2, 2])
    solution = Solution(x, y, q)
    x[:] = 0
    y[:] = 0
    q[:] = 99
    assert validate(problem, solution).feasible
    for array in (solution.x, solution.y, solution.q):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        solution.q[0] = 8
    np.testing.assert_array_equal(solution.q, [2, 2])
