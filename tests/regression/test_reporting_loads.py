"""Reporting and graph state must agree with the reference shared-load latency."""

import numpy as np
import pytest

from service_placement.domain.loads import residual_arrays
from service_placement.domain.solution import Solution
from service_placement.domain.validation import validate
from service_placement.utils.solver_workspace import SolverWorkspace
from tests.helpers import make_problem, route_solution


def test_shared_arc_loads_match_reference_violations(shared_arc):
    problem, (x, y, q) = shared_arc
    problem.latencies = [3, 3]
    original = [values.copy() for values in (x, y, q)]
    loads = residual_arrays(problem, x, y, q)
    np.testing.assert_array_equal(loads["demand_latency"], [4, 4])
    report = validate(problem, Solution(x, y, q))
    assert [v.observed for v in report.violations if v.constraint == "latency"] == [4, 4]
    for before, after in zip(original, (x, y, q), strict=False):
        np.testing.assert_array_equal(before, after)


def test_working_graph_preserves_separate_directed_arc_loads():
    problem = make_problem(services=2)
    problem.origins = [0, 1]
    x, y, q = route_solution(problem, [[0, 1], [1, 0]], [1, 0], [2, 3])
    workspace = SolverWorkspace(problem)
    workspace.update_graph_attributes_from_arc_variables(workspace.graph, x, y, q)
    assert workspace.graph.is_directed()
    assert workspace.graph.number_of_edges() == len(problem.network_graph.arc_list)
    assert workspace.graph.edges[0, 1]["latency"] == 2
    assert workspace.graph.edges[1, 0]["latency"] == 3
    assert workspace.graph.edges[0, 1]["remaining_capacity"] == 8
    assert workspace.graph.edges[1, 0]["remaining_capacity"] == 7
    assert "remaining_capacity" not in problem.network_graph.nx_graph.edges[0, 1]


@pytest.mark.requires("matplotlib")
def test_plot_summary_displays_total_traffic_latency(shared_arc):
    import matplotlib.pyplot as plt

    from service_placement.benchmarks.reporting import plot_instance_solution

    plt.switch_backend("Agg")
    problem, (x, y, q) = shared_arc
    problem.problem_name = "shared_arc"
    problem.demand_per_service = [2]
    problem.latencies = [3, 3]
    fig, ax = plot_instance_solution(problem, x, y, q, show=False)
    try:
        summary = next(text.get_text() for text in ax.texts if "solution" in text.get_text())
        assert "d0: L 4/3" in summary and "d1: L 4/3" in summary
    finally:
        plt.close(fig)
