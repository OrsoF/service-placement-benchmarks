from itertools import product

import numpy as np
import pytest

from service_placement.domain.solution import SolvingInstance
from service_placement.domain.validation import SolutionValidationError, validate_arc_solution
from service_placement.solvers.fast_heuristic import Solver as FastHeuristic
from tests.helpers import make_problem, route_solution


def test_shared_load_counterexample_rejected_by_every_entrypoint(shared_arc):
    problem, solution = shared_arc
    problem.latencies = [3, 3]
    with pytest.raises(SolutionValidationError, match="latency"):
        validate_arc_solution(problem, *solution)
    checker = SolvingInstance(problem)
    assert not checker.test_compact_solution([1], [0, 0], [2, 2])
    assert not checker.test_solution([1], [0, 0], [2, 2])
    assert not checker.validate_graph_attributes(checker.graph, [1], [0, 0], [2, 2])
    assert not FastHeuristic(problem)._feasible([1], [0, 0], [2, 2])


def test_multihop_latency_uses_all_shared_loads_and_beta():
    problem = make_problem(edges=((0, 1), (1, 2)), nodes=3)
    problem.origins = [0, 1]
    problem.network_graph.betas = [1] * 4
    solution = route_solution(problem, [[0, 1, 2], [1, 2]], [2], [2, 2])
    # First arc: 1 + 2, second arc: 1 + 4. First demand latency: 8.
    problem.latencies = [7, 5]
    with pytest.raises(SolutionValidationError, match="latency"):
        validate_arc_solution(problem, *solution)
    problem.latencies = [8, 5]
    validate_arc_solution(problem, *solution)


def test_balanced_cycle_is_allowed_by_arc_milp():
    problem = make_problem(edges=((0, 1), (2, 3)), nodes=4, demands=1)
    x, y, q = route_solution(problem, [[0, 1]], [1], [1])
    for edge in [(2, 3), (3, 2)]:
        x[0, problem.network_graph.arc_index_dictionary[edge]] = 1
    validate_arc_solution(problem, x, y, q)
    problem.latencies = [2]
    with pytest.raises(SolutionValidationError, match="latency"):
        validate_arc_solution(problem, x, y, q)


def test_fast_heuristic_lift_respects_other_demands_latency(shared_arc):
    problem, _ = shared_arc
    problem.latencies = [3, 100]
    solver = FastHeuristic(problem)
    p, r = np.array([1]), np.array([0, 0])
    q = solver._lift(p, r, np.array([1, 1]))
    assert sum(q) == 3
    assert solver._feasible(p, r, q)
    validate_arc_solution(problem, *solver.solving_instance.solution_to_arc_variables(p, r, q))


def test_remaining_quantity_respects_other_demands_latency(shared_arc):
    problem, _ = shared_arc
    problem.latencies = [3, 100]
    checker = SolvingInstance(problem)
    assert checker._get_remaining_quantity_per_demand([1], [0, 0], [1, 1], 1) == 1


def test_remaining_quantity_sums_latency_slopes_over_entire_path():
    problem = make_problem(edges=((0, 1), (1, 2)), nodes=3)
    # Both demands traverse both arcs. Load 2 on each arc gives latency 4.
    problem.latencies = [7, 7]
    checker = SolvingInstance(problem)
    assert checker._get_remaining_quantity_per_demand([2], [0, 0], [1, 1], 0) == 1


def test_remaining_quantity_with_zero_latency_slope(shared_arc):
    problem, _ = shared_arc
    problem.network_graph.alphas = [0, 0]
    checker = SolvingInstance(problem)
    assert checker._get_remaining_quantity_per_demand([1], [0, 0], [1, 1], 0) == 8


def arc_milp_model(problem):
    """Build the repository's actual constraints; no optimizer is required."""
    import pyomo.environ as pyo

    from service_placement.utils.arc_formulation import add_arc_constraints

    model = pyo.ConcreteModel()
    network = problem.network_graph
    model.D = pyo.Set(initialize=range(problem.total_number_of_demands))
    model.S = pyo.Set(initialize=range(problem.number_of_service))
    model.U = pyo.Set(initialize=range(network.nx_graph.number_of_nodes()))
    model.A = pyo.Set(initialize=range(len(network.arc_list)))
    model.x = pyo.Var(model.D, model.A, within=pyo.Binary)
    model.y = pyo.Var(model.S, model.U, within=pyo.Binary)
    model.p = pyo.Var(model.D, within=pyo.NonNegativeReals)
    model.P = pyo.Var(model.D, model.A, within=pyo.NonNegativeReals)
    model.L = pyo.Var(model.D, model.A, within=pyo.NonNegativeReals)
    for name, index, values in [
        ("Pdmin", model.D, problem.min_flows),
        ("Pdmax", model.D, problem.max_flows),
        ("latencies", model.D, problem.latencies),
        ("resources", model.S, problem.needed_resources),
        ("serv_number", model.D, problem.demand_to_service),
        ("node_resources", model.U, network.available_resources),
        ("arc_capacity", model.A, network.capacities),
        ("alpha", model.A, network.alphas),
        ("beta", model.A, network.betas),
    ]:
        setattr(model, name, pyo.Param(index, initialize=dict(enumerate(values))))
    model.inc_arcs = pyo.Param(
        model.U,
        within=pyo.Any,
        initialize={u: [a for a, (_, v) in enumerate(network.arc_list) if v == u] for u in model.U},
    )
    model.out_arcs = pyo.Param(
        model.U,
        within=pyo.Any,
        initialize={u: [a for a, (v, _) in enumerate(network.arc_list) if v == u] for u in model.U},
    )
    model.ind_u_origin = pyo.Param(
        model.U, model.D, initialize=lambda m, u, d: int(u == problem.origins[d])
    )
    model.M = pyo.Param(initialize=max(problem.max_flows))
    model.N = pyo.Param(initialize=max(problem.latencies))
    add_arc_constraints(model, model.D, model.A, model.U, model.S, model.x)
    return model


def milp_accepts(model, problem, x, y, q):
    import pyomo.environ as pyo

    # For binary x, P and L are fixed to these values by the linearizations.
    for d in model.D:
        model.p[d].value = q[d]
        for a in model.A:
            model.x[d, a].value = x[d, a]
            model.P[d, a].value = x[d, a] * q[d]
            model.L[d, a].value = x[d, a] * (
                problem.network_graph.betas[a]
                + problem.network_graph.alphas[a] * sum(x[dp, a] * q[dp] for dp in model.D)
            )
    for s in model.S:
        for u in model.U:
            model.y[s, u].value = y[s, u]
    for variable in model.component_data_objects(pyo.Var):
        if variable.value not in variable.domain:
            return False
    for constraint in model.component_data_objects(pyo.Constraint):
        body = pyo.value(constraint.body)
        if constraint.has_lb() and body < pyo.value(constraint.lower) - 1e-6:
            return False
        if constraint.has_ub() and body > pyo.value(constraint.upper) + 1e-6:
            return False
    return True


def validator_accepts(problem, solution):
    try:
        validate_arc_solution(problem, *solution)
    except SolutionValidationError:
        return False
    return True


@pytest.mark.requires("pyomo")
def test_validator_agrees_with_arc_milp_on_1024_assignments(shared_arc):
    problem, _ = shared_arc
    problem.latencies = [3, 3]
    problem.max_flows = [3, 3]
    problem.network_graph.capacities = [3, 3]
    problem.network_graph.available_resources = [1, 10]
    model = arc_milp_model(problem)
    accepted = rejected = 0
    for routing, placement, quantities in product(
        product((0, 1), repeat=4),
        product((0, 1), repeat=2),
        product((0, 1.25, 2, 4), repeat=2),
    ):
        solution = np.reshape(routing, (2, 2)), np.reshape(placement, (1, 2)), quantities
        expected = milp_accepts(model, problem, *solution)
        assert validator_accepts(problem, solution) == expected, solution
        accepted += expected
        rejected += not expected
    assert accepted > 0 and rejected > 0
    assert accepted + rejected == 1024


@pytest.mark.parametrize("latency,expected", [(2, False), (3, True)])
@pytest.mark.requires("pyomo")
def test_cycle_validation_matches_actual_milp_constraints(latency, expected):
    problem = make_problem(edges=((0, 1), (2, 3)), nodes=4, demands=1)
    solution = route_solution(problem, [[0, 1]], [1], [1])
    # Add a disconnected circulation; conservation alone allows it.
    for edge in [(2, 3), (3, 2)]:
        solution[0][0, problem.network_graph.arc_index_dictionary[edge]] = 1
    problem.latencies = [latency]
    assert milp_accepts(arc_milp_model(problem), problem, *solution) == expected
    assert validator_accepts(problem, solution) == expected
