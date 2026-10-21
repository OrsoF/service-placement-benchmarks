# Development checks

Run commands from the repository root with Python 3.10 or later.

## Minimal tests

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m build
```

This profile installs NumPy, NetworkX, pytest, and the build frontend. Optional
tests declare their dependencies with `@pytest.mark.requires("pyomo")`, for
example. A genuinely absent module produces an explicit skip. A dependency
that is installed but cannot import produces a failure, including when its
transitive dependency is missing. Test collection does not require every extra.

## Optional dependency coverage

```bash
python -m pip install -e ".[benchmarks,exact,environments,evolutionary,test]"
python -m pytest -m "not rl" --require-extras
```

`--require-extras` turns missing declared dependencies into failures. CI runs
both this profile and the minimal profile on Linux and Windows, with Python
3.10 and 3.12. Pyomo model comparisons do not require a Gurobi license.

The separate Linux RL job installs CPU PyTorch and all required extras, then
executes the full suite:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[benchmarks,exact,evolutionary,rl,test]"
python -m pytest --require-extras
```

Mark tests requiring Stable-Baselines3 with both `requires("stable_baselines3")`
and `rl`, so the non-RL profile can exclude them deliberately. Import smoke
tests use the declarations in `tests/optional.py`; update these when adding a
module that imports an optional dependency.

## Static checks

CI runs these checks with Python 3.10, matching the minimum supported version
and mypy's target. Use Python 3.10 to reproduce that environment locally; newer
Python versions may install dependency stubs that require newer Python syntax.

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy
```

Ruff checks imports, syntax, unused names, modernization, and common mistakes.
Formatting is checked separately. Historical files under `archive/` are outside
this active-code scope. To apply formatting locally, use `python -m ruff format
src tests examples`. Tool versions are pinned in the `dev` extra for repeatable CI.

Typing is strict within the explicit `tool.mypy.files` list in `pyproject.toml`:
the public API and errors, immutable problem arrays, solution, load and validation contracts, solver base and
registry, mathematical helpers, heuristic helpers, and legacy state adapter.
`domain.contracts` and `utils.solver_base` define typed array aliases and
protocols for these boundaries. Other algorithm and environment modules remain
outside strict coverage; add them to this list as their annotations mature.
Following their imports silently allows this incremental approach without
globally suppressing errors in checked files. NetworkX's missing external stubs
have a named override. Dynamic solver options and indexed optimizer objects use
`Any` at the adapter boundary. The distribution includes `service_placement/py.typed`.

## Adding a solver

Create `src/service_placement/solvers/<name>.py` with a `Solver` class. Subclass
`GenericSolver` from `service_placement.utils.solver_base` and call
`super().__init__(problem)` explicitly.
Implement `run`, populate outputs through `_set_output_variables`, and record
metrics and termination information. Register the flat module in
`utils.solver_registry`, explicitly declaring its optional extra and whether
it is mathematical. Shared logic belongs in the appropriate `utils/` module.
Keep mutable candidate state in a `SolverWorkspace`; treat input problems as
shared data. Read numerical inputs from `self.data` rather than copying problem
attributes into the solver. Copy vectors only when creating mutable candidates;
`problem.with_data(...)` creates an independent problem variant. Convert indexed
optimizer outputs with the extraction helpers.
The public facade copies outputs into `Solution`, validates them, and recomputes
the objective. Constructor contract tests cover every registered backend.
