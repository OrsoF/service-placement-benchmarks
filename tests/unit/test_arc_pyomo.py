"""Arc parameter contracts, including the corrected latency upper bound."""

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from service_placement.domain.network import NetworkSnapshot
from service_placement.domain.problem_arrays import ProblemArrays

pytestmark = pytest.mark.requires("pyomo")
SOLVERS = ("ilp_arc", "lp_arc", "milp_arc", "milp_arc_full")
EXPECTED = json.loads(
    (Path(__file__).parents[1] / "regression/fixtures/arc_parameters.json").read_text()
)


def build_solver(name, data):
    problem = SimpleNamespace(
        data=data, network_graph=NetworkSnapshot.from_data(data, nx.DiGraph())
    )
    solver = importlib.import_module(f"service_placement.solvers.{name}").Solver(problem)
    solver._build_model()
    solver._build_sets()
    solver._build_variables()
    solver._build_params()
    return solver


@pytest.fixture(params=SOLVERS)
def solver(request):
    # Deliberately distinct service/node resources, ungrouped demands and
    # an arc ordering that differs from NetworkX's edge iteration order.
    data = ProblemArrays(
        origins=np.array([2, 0, 3, 1]),
        demand_to_service=np.array([1, 0, 1, 0]),
        min_flows=np.array([1, 2, 3, 4]),
        max_flows=np.array([5, 6, 7, 8]),
        latency_limits=np.array([11.5, 17, 23.25, 29]),
        service_resources=np.array([2, 7]),
        node_resources=np.array([17, 23, 31, 47]),
        arc_capacities=np.array([11, 13, 17, 19, 23, 29]),
        latency_slopes=np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5]),
        latency_offsets=np.array([1, 2, 0, 4, 3, 5]),
        arcs=((2, 1), (0, 1), (1, 3), (1, 0), (1, 2), (3, 1)),
    )
    return build_solver(request.param, data)


def test_parameters_match_expected_snapshot(solver):
    import pyomo.environ as pyo

    params = solver.model.component_map(pyo.Param)
    assert list(params) == list(EXPECTED)
    for name, expected in EXPECTED.items():
        param = params[name]
        assert str(param.domain) == expected["domain"], name
        assert param.mutable == expected["mutable"], name
        assert param.index_set().dimen == expected["dimen"], name
        indices = [list(key) if isinstance(key, tuple) else key for key in param]
        assert indices == expected["indices"], name
        assert list(param.extract_values().values()) == expected["values"], name


def test_parameters_keep_formulation_index_sets(solver):
    model = solver.model
    labels = (
        ("total_number_of_demands", "number_of_services", "node_number", "arc_number")
        if hasattr(model, "total_number_of_demands")
        else ("D", "S", "U", "A")
    )
    sets = dict(zip("DSUA", (getattr(model, name) for name in labels), strict=True))
    axes = {
        "origins": "D",
        "Pdmin": "D",
        "Pdmax": "D",
        "latencies": "D",
        "resources": "S",
        "alpha": "A",
        "beta": "A",
        "serv_number": "D",
        "out_arcs": "U",
        "inc_arcs": "U",
        "ind_u_origin": "UD",
        "node_resources": "U",
        "arc_capacity": "A",
    }
    for name, dimensions in axes.items():
        actual = list(getattr(model, name).index_set().subsets())
        expected = [sets[label] for label in dimensions]
        assert len(actual) == len(expected), name
        assert all(a is b for a, b in zip(actual, expected, strict=True)), name


def test_resources_constraint_uses_service_demand_and_node_capacity(solver):
    import pyomo.environ as pyo
    from pyomo.repn import generate_standard_repn

    solver._build_constraints()
    for node in range(solver.data.n_nodes):
        constraint = solver.model.resources_constraint[node]
        expression = generate_standard_repn(constraint.body)
        coefficients = {
            variable.name: coefficient
            for variable, coefficient in zip(
                expression.linear_vars, expression.linear_coefs, strict=True
            )
        }
        assert coefficients == {
            solver.model.y[s, node].name: solver.data.service_resources[s]
            for s in range(solver.data.n_services)
        }
        assert pyo.value(constraint.upper) == solver.data.node_resources[node]


def test_service_resources_do_not_depend_on_node_count(solver):
    data = replace(
        solver.data,
        service_resources=np.array([2, 7, 11, 13, 19]),
        demand_to_service=np.array([4, 0, 3, 1]),
    )
    name = type(solver).__module__.rsplit(".", 1)[1]
    model = build_solver(name, data).model
    assert list(model.resources.values()) == [2, 7, 11, 13, 19]
    assert list(model.node_resources.values()) == [17, 23, 31, 47]
