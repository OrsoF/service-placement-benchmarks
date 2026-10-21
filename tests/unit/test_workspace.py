"""Solvers own mutable work state; shared input data stays unchanged."""

import numpy as np
import pytest

from service_placement import make_tree_instance, solve
from service_placement.domain.conversions import compact_to_arc
from service_placement.domain.solution import SolvingInstance
from service_placement.domain.solution import compact_to_arc as legacy_compact_to_arc
from service_placement.utils.heuristic_solvers import baseline_solution
from service_placement.utils.solver_workspace import SolverWorkspace


def test_public_solve_does_not_create_legacy_problem_state():
    problem = make_tree_instance(2, neighbors=0)
    assert problem._legacy_state is None
    assert solve(problem, seed=42).validation.feasible
    assert problem._legacy_state is None


def test_workspace_owns_result_graph_and_path_cache():
    problem = make_tree_instance(2, neighbors=0)
    first, second = SolverWorkspace(problem), SolverWorkspace(problem)
    candidate = baseline_solution(problem)
    # This small instance has a feasible cloud baseline.
    first.set_variable_solution(candidate.x, candidate.y, candidate.q)
    assert problem._legacy_state is None
    assert second.current_solution is second.current_arc_solution is None
    assert first.path_cache is not second.path_cache
    first.compute_nth_shortest_path(0, 1, 0)
    assert first.saved_shortest_path and not second.saved_shortest_path
    first.graph.nodes[0]["remaining_resources"] = -1
    assert second.graph.nodes[0]["remaining_resources"] >= 0
    assert "remaining_resources" not in problem.network_graph.nx_graph.nodes[0]
    assert not first.current_arc_solution[0].flags.writeable


def test_invalid_workspace_result_does_not_replace_current_state():
    workspace = SolverWorkspace(make_tree_instance(2, neighbors=0))
    candidate = baseline_solution(workspace.problem_instance)
    with pytest.raises(ValueError):
        workspace.set_variable_solution(candidate.x, candidate.y, [-1, -1])
    assert workspace.current_arc_solution is None


def test_legacy_imports_and_mutable_access_remain_available():
    assert SolvingInstance is SolverWorkspace
    assert legacy_compact_to_arc is compact_to_arc
    problem = make_tree_instance(2, neighbors=0)
    original = problem.q_variable.copy()
    replacement = original + 1
    problem.q_variable = replacement
    replacement[:] = 0
    np.testing.assert_array_equal(problem.q_variable, original + 1)


def test_workspace_random_generators_are_independent():
    problem = make_tree_instance(2)
    first, second = SolverWorkspace(problem, seed=42), SolverWorkspace(problem, seed=42)
    expected = second.rng.integers(1000, size=10)
    np.testing.assert_array_equal(first.rng.integers(1000, size=10), expected)
    first.rng.integers(1000, size=100)
    reference = SolverWorkspace(problem, seed=42)
    reference.rng.integers(1000, size=10)
    np.testing.assert_array_equal(
        second.rng.integers(1000, size=10), reference.rng.integers(1000, size=10)
    )
