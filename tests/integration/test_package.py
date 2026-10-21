"""Imports, CLI execution, and benchmark validation after the src migration."""

import ast
import csv
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from service_placement import SolveStatus
from service_placement.benchmarks import runner
from service_placement.cli import main
from service_placement.instances.synthetic import create_tree_problem_instance
from tests.optional import is_rl_module, module_requirements

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
MODULES = sorted(
    ".".join(path.relative_to(SOURCE).with_suffix("").parts).removesuffix(".__init__")
    for path in (SOURCE / "service_placement").rglob("*.py")
)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            module,
            marks=[
                pytest.mark.requires(*module_requirements(module)),
                *([pytest.mark.rl] if is_rl_module(module) else []),
            ],
        )
        for module in MODULES
    ],
)
def test_public_modules_import(module):
    importlib.import_module(module)


def test_local_imports_and_dynamic_registry_paths_exist():
    for path in (SOURCE / "service_placement").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("service_placement.") and " " not in node.value:
                    modules = [node.value]
            for module in modules:
                if module.startswith("service_placement."):
                    assert importlib.util.find_spec(module) is not None, (path, module)
                else:
                    assert module.split(".")[0] not in {
                        "core",
                        "instances",
                        "solvers",
                        "rl",
                        "data_io",
                        "plotting",
                    }, (path, module)


def test_solver_aliases_and_package_directory():
    from service_placement.utils.solver_registry import SOLVER_FILE_ALIASES, SOLVER_MODULES

    assert (SOURCE / "service_placement/solvers").is_dir()
    assert set(SOLVER_FILE_ALIASES.values()) <= set(SOLVER_MODULES)


def run_cli(tmp_path, *args):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE) + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "service_placement", *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_help_outside_repository(tmp_path):
    result = run_cli(tmp_path, "--help")
    assert result.returncode == 0, result.stderr
    assert "--instance" in result.stdout
    assert "--solvers" in result.stdout


@pytest.mark.parametrize("instance,expected_users", [("tree", "2"), ("star", "4")])
def test_cli_creates_benchmark_results_outside_repository(tmp_path, instance, expected_users):
    result = run_cli(
        tmp_path,
        "--instance",
        instance,
        "--groups",
        "2",
        "--solvers",
        "fast_heuristic",
        "--start-users",
        "2",
        "--stop-users",
        "3",
        "--neighbors",
        "0",
        "--no-plots",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    with (tmp_path / "artifacts/data" / f"exp_{instance}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["users"] == expected_users
    assert rows[0]["solver"] == "fast_heuristic"
    assert float(rows[0]["score"]) >= 0


def test_cli_rejects_invalid_step_before_creating_output(tmp_path):
    result = run_cli(tmp_path, "--step-users", "0")
    assert result.returncode == 2
    assert not (tmp_path / "artifacts").exists()


def test_benchmark_runner_rejects_infeasible_solver_output(monkeypatch):
    from service_placement import api

    problem = create_tree_problem_instance(2, 0)
    candidate = SimpleNamespace(
        run=lambda: None,
        x_var=problem.x_variable,
        y_var=problem.y_variable,
        q_var=[flow + 1000 for flow in problem.max_flows],
    )
    monkeypatch.setattr(api, "make_solver", lambda *args, **kwargs: candidate)
    result = runner.run_solver("heuristic", problem, 1)
    assert result.status == SolveStatus.INVALID_SOLUTION
    assert not result.validation.feasible
    assert result.objective is None


def test_failed_benchmark_returns_nonzero_status_and_records_failure(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("simulated solver failure")

    monkeypatch.setattr(runner, "run_solver", fail)
    output = tmp_path / "failed.csv"
    assert (
        main(
            [
                "--solvers",
                "fast_heuristic",
                "--start-users",
                "2",
                "--stop-users",
                "3",
                "--neighbors",
                "0",
                "--no-plots",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["score"] == ""
