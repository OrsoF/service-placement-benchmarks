import numpy as np
import pytest

from service_placement.domain.solution import SolvingInstance
from service_placement.domain.solution import validate_arc_solution as legacy_validate_arc_solution
from service_placement.domain.validation import SolutionValidationError, validate_arc_solution
from tests.helpers import make_problem, route_solution


def test_legacy_entrypoint_is_the_same_validator():
    assert legacy_validate_arc_solution is validate_arc_solution


def test_valid_solution_is_not_modified(shared_arc):
    problem, solution = shared_arc
    originals = [v.copy() for v in solution]
    assert validate_arc_solution(problem, *solution) is None
    for actual, original in zip(solution, originals, strict=False):
        np.testing.assert_array_equal(actual, original)


def test_exact_latency_and_capacity_boundaries(shared_arc):
    problem, solution = shared_arc
    problem.latencies = [4, 4]
    problem.network_graph.capacities = [4, 10]
    validate_arc_solution(problem, *solution)
    assert SolvingInstance(problem).test_solution([1], [0, 0], [2, 2])


@pytest.mark.parametrize("constraint", ["resources", "capacity", "minimum", "maximum"])
def test_infeasible_constraints(shared_arc, constraint):
    problem, solution = shared_arc
    if constraint == "resources":
        problem.network_graph.available_resources = [10, 1]
    elif constraint == "capacity":
        problem.network_graph.capacities = [3, 10]
    elif constraint == "minimum":
        problem.min_flows = [3, 0]
    else:
        problem.max_flows = [1, 10]
    with pytest.raises(SolutionValidationError):
        validate_arc_solution(problem, *solution)
    assert not SolvingInstance(problem).test_solution([1], [0, 0], [2, 2])


def test_resources_accumulate_across_services():
    problem = make_problem(services=2)
    problem.network_graph.available_resources = [10, 3]
    solution = route_solution(problem, [[0, 1], [0, 1]], [1, 1], [1, 1])
    with pytest.raises(SolutionValidationError, match="resources"):
        validate_arc_solution(problem, *solution)


@pytest.mark.parametrize(
    "index,values",
    [
        (0, [[1, 0]]),
        (0, [1, 0]),
        (0, [[[1, 0], [1, 0]]]),
        (1, [[0, 1, 0]]),
        (1, [0, 1]),
        (2, [2]),
        (2, [[2, 2]]),
    ],
)
def test_wrong_arc_dimensions(shared_arc, index, values):
    problem, original = shared_arc
    solution = list(original)
    solution[index] = values
    with pytest.raises(SolutionValidationError, match="shape"):
        validate_arc_solution(problem, *solution)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_rejected(shared_arc, index, value):
    problem, solution = shared_arc
    solution[index].flat[0] = value
    with pytest.raises(SolutionValidationError, match="finite"):
        validate_arc_solution(problem, *solution)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("value", [-1, 0.5, 2])
def test_routing_and_placement_are_binary(shared_arc, index, value):
    problem, solution = shared_arc
    solution[index].flat[0] = value
    with pytest.raises(SolutionValidationError, match="binary"):
        validate_arc_solution(problem, *solution)


@pytest.mark.parametrize("placement", [[[0, 0]], [[1, 1]]])
def test_service_must_be_placed_exactly_once(shared_arc, placement):
    problem, (x, _, q) = shared_arc
    with pytest.raises(SolutionValidationError, match="exactly one"):
        validate_arc_solution(problem, x, placement, q)


def test_continuous_quantities_are_valid_for_milp(shared_arc):
    problem, (x, y, _) = shared_arc
    validate_arc_solution(problem, x, y, [1.25, 2.5])
    assert SolvingInstance(problem).test_solution([1], [0, 0], [1.25, 2.5])


@pytest.mark.parametrize("quantities", [[-1, 2], [0, 0]])
def test_empty_route_to_remote_placement_is_infeasible(shared_arc, quantities):
    problem, (x, y, _) = shared_arc
    with pytest.raises(SolutionValidationError):
        validate_arc_solution(problem, np.zeros_like(x), y, quantities)


def test_reverse_route_violates_flow_conservation(shared_arc):
    problem, (_, y, q) = shared_arc
    with pytest.raises(SolutionValidationError, match="conservation"):
        validate_arc_solution(problem, [[0, 1], [0, 1]], y, q)


def test_two_routes_with_correct_endpoints_still_violate_conservation():
    problem = make_problem(edges=((0, 1), (0, 2), (1, 3), (2, 3)), nodes=4, demands=1)
    x, y, q = route_solution(problem, [[0, 1, 3]], [3], [1])
    for edge in [(0, 2), (2, 3)]:
        x[0, problem.network_graph.arc_index_dictionary[edge]] = 1
    with pytest.raises(SolutionValidationError, match="conservation"):
        validate_arc_solution(problem, x, y, q)


def test_local_service_needs_no_arc_even_with_positive_quantity():
    problem = make_problem(edges=(), nodes=1)
    solution = route_solution(problem, [[0], [0]], [0], [2, 2])
    problem.latencies = [0, 0]
    validate_arc_solution(problem, *solution)
    assert SolvingInstance(problem).test_solution([0], [0, 0], [2, 2])


@pytest.mark.parametrize(
    "placement,routing,quantities",
    [
        ([], [0, 0], [2, 2]),
        ([1], [0], [2, 2]),
        ([1], [0, 0], [2]),
        ([[1]], [0, 0], [2, 2]),
        ([1], [[0, 0]], [2, 2]),
        ([1], [0, 0], [[2, 2]]),
        ([0.5], [0, 0], [2, 2]),
        ([1], [0.5, 0], [2, 2]),
        ([1], [-1, 0], [2, 2]),
        ([2], [0, 0], [2, 2]),
        ([1], [1, 0], [2, 2]),
        ([np.nan], [0, 0], [2, 2]),
        ([1], [0, 0], [np.inf, 2]),
        (None, [0, 0], [2, 2]),
        ([1], [0, 0], ["bad", 2]),
    ],
)
def test_malformed_compact_solution_returns_false(shared_arc, placement, routing, quantities):
    problem, _ = shared_arc
    checker = SolvingInstance(problem)
    assert not checker.test_compact_solution(placement, routing, quantities)
    assert not checker.validate_graph_attributes(checker.graph, placement, routing, quantities)


def test_unreachable_compact_route_returns_false():
    problem = make_problem(edges=(), nodes=2)
    assert not SolvingInstance(problem).test_solution([1], [0, 0], [1, 1])


def test_tolerance_handles_solver_noise_without_modifying_inputs(shared_arc):
    problem, solution = shared_arc
    solution[0][:] += 1e-8
    solution[1][:] += 1e-8
    problem.latencies = [4 - 1e-8, 4 - 1e-8]
    validate_arc_solution(problem, *solution)
    with pytest.raises(SolutionValidationError):
        validate_arc_solution(problem, *solution, atol=0)


@pytest.mark.parametrize("atol", [-1, np.nan, np.inf])
def test_invalid_tolerance(shared_arc, atol):
    problem, solution = shared_arc
    with pytest.raises(ValueError, match="atol"):
        validate_arc_solution(problem, *solution, atol=atol)
