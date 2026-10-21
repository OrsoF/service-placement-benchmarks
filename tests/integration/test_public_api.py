"""Public API, reproducibility, lazy dependencies and solver result semantics."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from service_placement import (
    MissingSolverDependencyError,
    SolverExecutionError,
    SolveStatus,
    UnknownSolverError,
    api,
    list_solvers,
    make_star_instance,
    make_tree_instance,
    solve,
    validate,
)
from service_placement.utils import solver_registry as registry


def test_user_quickstart_and_repeatability():
    problem = make_tree_instance(n_users=20, seed=42)
    before = problem.q_variable.copy()
    result = solve(problem, solver="fast_heuristic", seed=42)
    report = validate(problem, result.solution)
    repeated = solve(make_tree_instance(n_users=20, seed=42), solver="fast_heuristic", seed=42)
    assert report.feasible and result.validation.feasible
    assert result.objective == np.sum(result.solution.q) == repeated.objective
    assert result.status == SolveStatus.FEASIBLE
    assert result.solver == "fast_heuristic" and result.seed == 42
    assert result.runtime >= result.build_time >= 0
    for name in ("x", "y", "q"):
        np.testing.assert_array_equal(
            getattr(result.solution, name), getattr(repeated.solution, name)
        )
    np.testing.assert_array_equal(before, problem.q_variable)


def test_tree_seed_is_metadata_for_deterministic_generator():
    first, second = make_tree_instance(2, seed=0), make_tree_instance(2, seed=9)
    assert first.network_graph.arc_list == second.network_graph.arc_list
    np.testing.assert_array_equal(first.min_flows, second.min_flows)
    assert (first.seed, second.seed) == (0, 9)


def test_star_single_user_group_has_no_self_loops():
    from service_placement import Solution
    from tests.helpers import route_solution

    problem = make_star_instance(n_groups=2, users_per_group=1, neighbors=3, seed=42)
    assert all(u != v for u, v in problem.network_graph.arc_list)
    candidate = Solution(
        *route_solution(problem, [[0, 2], [4, 2], [1, 3], [4, 3]], [2, 3], [1] * 4)
    )
    assert validate(problem, candidate).feasible


def test_bad_baseline_does_not_prevent_a_feasible_instance_from_being_solved():
    from service_placement import Solution

    problem = make_tree_instance(1, neighbors=0)
    assert not validate(
        problem, Solution(problem.x_variable, problem.y_variable, problem.q_variable)
    ).feasible
    assert solve(problem, seed=42).validation.feasible


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.requires("cma")
def test_cma_reproducibility_including_zero_seed(seed):
    options = {"max_iter": 2, "max_pcc": 1}
    before = np.random.get_state()
    first = solve(make_tree_instance(2, 0), solver="cma", seed=seed, solver_options=options)
    second = solve(make_tree_instance(2, 0), solver="cma", seed=seed, solver_options=options)
    assert first.validation.feasible and second.validation.feasible
    for name in ("x", "y", "q"):
        np.testing.assert_array_equal(getattr(first.solution, name), getattr(second.solution, name))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_users": 0},
        {"n_users": -1},
        {"n_users": True},
        {"n_users": 1.5},
        {"neighbors": -1},
        {"neighbors": 0.5},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_tree_builder_parameters_fail_clearly(kwargs):
    with pytest.raises(ValueError, match="must be an integer"):
        make_tree_instance(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n_groups": 0}, {"users_per_group": 0}, {"seed": -1}])
def test_star_builder_parameters_fail_clearly(kwargs):
    options = {"n_groups": 2, "users_per_group": 2, **kwargs}
    with pytest.raises(ValueError, match="must be an integer"):
        make_star_instance(**options)


def test_unknown_solver_is_typed_and_lists_available_names():
    with pytest.raises(UnknownSolverError, match="fast_heuristic"):
        solve(make_tree_instance(2), solver="missing")
    assert "fast_heuristic" in list_solvers()
    assert all(name == name.lower() for name in list_solvers())


@pytest.mark.parametrize("time_limit", [0, -1, float("nan"), float("inf"), "bad", True])
def test_bad_time_limit_rejected(time_limit):
    with pytest.raises(ValueError, match="time_limit"):
        solve(make_tree_instance(2), time_limit=time_limit)


def test_unsupported_time_limit_and_duplicate_seed_rejected():
    problem = make_tree_instance(2)
    with pytest.raises(ValueError, match="does not support time_limit"):
        solve(problem, time_limit=1)
    with pytest.raises(ValueError, match="Pass seed directly"):
        solve(problem, solver_options={"seed": 42})


def test_missing_optional_import_has_installation_command(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'pyomo'", name="pyomo")

    monkeypatch.setattr(registry.importlib, "import_module", missing)
    with pytest.raises(
        MissingSolverDependencyError, match=r"pip install 'service-placement-benchmarks\[exact\]'"
    ):
        solve(make_tree_instance(2), solver="ilp_arc")


def test_missing_dependency_during_run_is_also_actionable(monkeypatch):
    def missing():
        raise ModuleNotFoundError("No module named 'cma'", name="cma")

    monkeypatch.setattr(api, "make_solver", lambda *args, **kwargs: SimpleNamespace(run=missing))
    with pytest.raises(MissingSolverDependencyError, match="evolutionary"):
        solve(make_tree_instance(2), solver="cma")


def candidate_backend(problem, termination=None, quantities=None):
    return SimpleNamespace(
        run=lambda **kwargs: None,
        termination_condition=termination,
        x_var=problem.x_variable,
        y_var=problem.y_variable,
        q_var=problem.q_variable if quantities is None else quantities,
        score=999999,
        build_time=0,
    )


def test_invalid_heuristic_output_does_not_claim_problem_is_infeasible(monkeypatch):
    problem = make_tree_instance(2)
    backend = candidate_backend(problem, quantities=np.asarray(problem.max_flows) + 100)
    monkeypatch.setattr(api, "make_solver", lambda *args, **kwargs: backend)
    result = solve(problem)
    assert result.status == SolveStatus.INVALID_SOLUTION
    assert result.solution is not None and result.objective is None
    assert not result.validation.feasible


def test_objective_is_recomputed_from_validated_quantities(monkeypatch):
    problem = make_tree_instance(2)
    monkeypatch.setattr(api, "make_solver", lambda *args, **kwargs: candidate_backend(problem))
    result = solve(problem)
    assert result.objective == float(np.sum(problem.q_variable))


@pytest.mark.parametrize(
    "termination,with_candidate,status",
    [
        ("optimal", True, SolveStatus.OPTIMAL),
        ("maxTimeLimit", True, SolveStatus.TIME_LIMIT),
        ("maxTimeLimit", False, SolveStatus.TIME_LIMIT),
        ("infeasible", True, SolveStatus.INFEASIBLE),
        ("unbounded", True, SolveStatus.UNBOUNDED),
        ("infeasibleOrUnbounded", False, SolveStatus.NO_SOLUTION),
        ("optimal", False, SolveStatus.NO_SOLUTION),
        (None, False, SolveStatus.NO_SOLUTION),
    ],
)
def test_exact_termination_and_incumbent_semantics(
    monkeypatch, termination, with_candidate, status
):
    problem = make_tree_instance(2)
    backend = candidate_backend(problem, termination)
    kwargs_received = {}

    def run(options):
        kwargs_received.update(options)

    backend.run = run
    if not with_candidate:
        backend.x_var = backend.y_var = backend.q_var = None
    monkeypatch.setattr(api, "make_solver", lambda *args, **kwargs: backend)
    result = solve(problem, solver="ilp_arc", seed=42, time_limit=3)
    assert result.status == status
    assert result.termination == termination
    assert kwargs_received == {"Seed": 42, "TimeLimit": 3.0}
    usable = with_candidate and termination not in ("infeasible", "unbounded")
    assert result.validation.feasible == usable
    assert (result.solution is not None) == usable
    assert (result.objective is not None) == usable


def test_license_failure_is_not_disguised_as_a_candidate(monkeypatch):
    def run(options):
        raise RuntimeError("license expired")

    monkeypatch.setattr(api, "make_solver", lambda *args, **kwargs: SimpleNamespace(run=run))
    with pytest.raises(SolverExecutionError, match="Gurobi installation and license"):
        solve(make_tree_instance(2), solver="ilp_arc")


@pytest.mark.parametrize(
    "termination,incumbent", [("optimal", True), ("infeasible", False), ("maxTimeLimit", False)]
)
@pytest.mark.requires("pyomo")
def test_real_arc_adapter_records_structured_termination(monkeypatch, termination, incumbent):
    from pyomo.opt import SolverResults, TerminationCondition

    import service_placement.solvers.ilp_arc as module

    problem = make_tree_instance(2)

    class FakeOptimizer:
        options = {}

        def solve(self, model):
            if incumbent:
                for index in model.demand_arc_transit:
                    model.demand_arc_transit[index].value = problem.x_variable[index]
                for index in model.y:
                    model.y[index].value = problem.y_variable[index]
                for d in model.p:
                    model.p[d].value = problem.q_variable[d]
            result = SolverResults()
            result.solver.termination_condition = getattr(TerminationCondition, termination)
            result.solver.time = 0
            return result

    monkeypatch.setattr(module.pyo, "SolverFactory", lambda *args: FakeOptimizer())
    result = solve(problem, solver="ilp_arc", seed=42, time_limit=1)
    assert result.termination == termination
    assert result.validation.feasible == incumbent
    assert (result.solution is not None) == incumbent


@pytest.mark.parametrize(
    "name", [name for name, spec in registry.REGISTRY.items() if spec.mathematical]
)
@pytest.mark.requires("pyomo")
def test_mathematical_adapters_accept_optimizer_options(name):
    import importlib
    import inspect

    cls = importlib.import_module(registry.REGISTRY[name].module).Solver
    assert "options" in inspect.signature(cls.run).parameters


@pytest.mark.parametrize("name", ["a2c", "ppo", "ppo_best_state_reset"])
@pytest.mark.requires("stable_baselines3")
@pytest.mark.rl
def test_rl_seed_and_explicit_training_options_reach_base_solver(monkeypatch, name):
    from service_placement.utils import rl_solvers

    def env(self, problem, max_step, reuse):
        return SimpleNamespace(max_step=max_step)

    monkeypatch.setattr(rl_solvers.RLSolver, "_make_env", env)
    monkeypatch.setattr(
        rl_solvers.RLSolver, "_make_model", lambda self: SimpleNamespace(seed=self.seed)
    )
    backend = registry.make_solver(name, make_tree_instance(2), seed=42, max_step=12, train_step=32)
    assert backend.seed == backend.model.seed == 42
    assert backend.env.max_step == 12 and backend.train_step == 32


def test_base_api_and_cli_never_import_optional_dependencies(tmp_path):
    source = str(Path(__file__).resolve().parents[2] / "src")
    code = f"""
import sys
import importlib.abc
sys.path.insert(0, {source!r})
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'pyomo', 'gurobipy', 'cma', 'pandas', 'matplotlib', 'tqdm', 'gymnasium', 'stable_baselines3', 'torch'}}:
            raise RuntimeError('optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from service_placement import make_tree_instance, solve, validate
p = make_tree_instance(n_users=20, seed=42)
r = solve(p, solver='fast_heuristic', seed=42)
assert validate(p, r.solution).feasible
from service_placement.cli import main
assert main(['--start-users', '2', '--stop-users', '3', '--seed', '42']) == 0
print(r.objective)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "140.0" in result.stdout
