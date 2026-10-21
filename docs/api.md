# Using service_placement

Install from the repository with `python -m pip install -e .`. Python 3.10 or
newer, NumPy and NetworkX are sufficient for the quickstart and the default CLI.

```python
from service_placement import make_tree_instance, solve, validate

problem = make_tree_instance(n_users=20, seed=42)
result = solve(problem, solver="fast_heuristic", seed=42)
report = validate(problem, result.solution)
print(result.objective, report.feasible)  # 140.0 True
```

`examples/quickstart.py` is executable from any directory after installation.
`make_star_instance(n_groups=4, users_per_group=20, seed=42)` builds the other
synthetic topology. Counts must be positive integers; `neighbors` and `seed`
must be nonnegative integers. Tree and star construction is deterministic;
the instance seed is recorded as metadata and does not change these generators.
The seed passed to `solve` controls the solver search, including RL training and
exact optimizer options. Reproducibility assumes the same software and runtime
environment; runtimes themselves are not reproducible measurements.

## Problem data

`problem.data` is a `ProblemArrays` snapshot, created once and shared by solvers,
workspaces, environments and validation. It contains `origins`,
`demand_to_service`, `min_flows`, `max_flows`, `latency_limits`,
`service_resources`, `node_resources`, `arc_capacities`, `latency_slopes` and
`latency_offsets`. Floating-point vectors use NumPy `float64`; indices use
`int64`. Dimensions are `n_services`, `n_demands`, `n_nodes` and `n_arcs`.
`data.arcs` fixes the order of arc columns and `data.arc_index` maps each arc
to its column.

These arrays and mappings are read-only and independent of the original
construction inputs. The problem's copied graph has a frozen topology;
workspaces own their mutable graph and candidate arrays. To change numerical
inputs, create another problem:

```python
larger_network = problem.with_data(arc_capacities=problem.data.arc_capacities * 2)
```

`with_data` checks dimensions, indices and numerical domains, preserves the name
and seed, and resets historical solution state. All coupled vectors must remain
dimensionally consistent. Historical attributes such as `problem.min_flows` and
`problem.network_graph.capacities` expose the same read-only arrays. The retained
`service_package` describes construction inputs; changing it does not change
`problem.data`. For a mutable candidate, explicitly copy a vector, for example
`quantities = problem.data.min_flows.copy()`.

## Solving

`list_solvers()` returns canonical names without importing optional packages.
For example:

```python
from service_placement import list_solvers, solve

print(list_solvers())
result = solve(problem, solver="fast_heuristic", seed=42,
               solver_options={"max_iter": 100, "patience": 12})
# Requires the exact extra and a working Gurobi installation/license:
result = solve(problem, solver="ilp_arc", seed=42, time_limit=30,
               run_options={"options": {"MIPGap": 0.01}})
```

`solver_options` is passed to the solver constructor; `run_options` is passed
to its `run` method. Unsupported keywords raise `TypeError`. Pass `seed` and
`time_limit` directly: they take precedence over corresponding optimizer options.
Time limits apply to mathematical solvers; heuristics reject this option rather
than silently ignoring it. Historical names such as `PPO` and `HRL` remain aliases
of the canonical lowercase names `ppo` and `hierarchical_ppo`.

Solver names and compatibility:

Solver names match their module filenames. Historical names remain accepted as
aliases; `list_solvers()` and `SolveResult.solver` use the new canonical names:

| Historical name | Canonical name |
| --- | --- |
| `ilp_arc_95_latency` | `ilp_arc_gap_5pct` |
| `heuristicprecise` | `heuristic_unit_step` |
| `discrete` | `discrete_evolutionary` |
| `cma` | `cma_es` |
| `hrl` | `hierarchical_ppo` |
| `ppo_options` | `ppo_best_state_reset` |

`ilp_arc_gap_5pct` defaults to `MIPGap=0.05`; it does not change latency constraints.
`heuristic_unit_step` defaults to a quantity increment of one. These defaults
remain configurable through the existing solver options.

Optional installations:

| Solver | Extra |
| --- | --- |
| `heuristic`, `heuristic_unit_step`, `fast_heuristic`, `discrete_evolutionary` | Base installation |
| `cma_es` | `python -m pip install -e ".[evolutionary]"` |
| `ilp_*`, `milp_*`, `lp_*` | `python -m pip install -e ".[exact]"`, plus Gurobi/license |
| `a2c`, `ppo`, `ppo_best_state_reset`, `hierarchical_ppo` | `python -m pip install -e ".[rl]"` |
| `ilp_arc_rl_warmstart` | `python -m pip install -e ".[exact,rl]"`, plus Gurobi/license |

`SolveResult` contains a copied `Solution` (or `None`), `objective`, `status`,
`runtime`, `build_time`, canonical `solver`, `seed`, `termination` and `validation`.
`runtime` measures elapsed seconds for construction, solving and validation.
`build_time` uses the backend's estimate when available (otherwise zero).
The objective is recomputed as `sum(solution.q)` only after feasibility succeeds.
It is `None` when no valid candidate exists. Always consult `validation.feasible`
before using a candidate.

| Status | Meaning |
| --- | --- |
| `feasible` | The reference validator accepted a candidate; no optimality claim |
| `optimal` | Backend reports optimal termination and the candidate validates |
| `time_limit` | Backend stopped at its time limit; may contain a valid incumbent |
| `infeasible` | Backend reports infeasibility; no candidate is returned |
| `unbounded` | Backend reports unboundedness; no candidate is returned |
| `no_solution` | No usable candidate, without an infeasibility proof |
| `invalid_solution` | Candidate retained for diagnosis; reference validation failed |

Optimality/infeasibility refers to the selected backend model, its tolerances and
any restrictions (for example a limited set of paths or fixed placements).
A failed heuristic does not prove that the problem is infeasible. Linear
relaxations may return fractional variables; these fail the binary solution
validator and are retained with `invalid_solution`.

Configuration errors raise `UnknownSolverError` or `MissingSolverDependencyError`
(with an installation command). Execution failures raise `SolverExecutionError`,
preserving the original cause; exact-solver failures include an installation and
license hint. These exceptions are exported from `service_placement`.

## Validation

`Solution(x, y, q)` copies numeric arrays and marks them read-only. Shapes are
`(demands, arcs)`, `(services, nodes)` and `(demands,)`. Arc columns follow
`problem.network_graph.arc_list`; node columns follow integer node indices.

```python
from service_placement import Solution, validate

candidate = Solution(x, y, q)
report = validate(problem, candidate)
for violation in report.violations:
    print(violation.constraint, violation.demand, violation.arc,
          violation.observed, violation.limit, violation.message)
report.raise_if_infeasible()  # Optional strict check
```

Reports collect independent violations with zero-based demand/service/node/arc
indices where applicable. Invalid shapes, nonnumeric/nonfinite arrays or binary
domains stop dependent physical calculations. `validate(problem, None)` returns
an infeasible report with `no_solution`. The default absolute tolerance is `1e-6`.

The checks cover dimensions, finite/binary variables, exactly one placement,
flow bounds, node resources, arc capacities, conservation and latency. Latency
uses each arc's total traffic: `x @ (beta + alpha * (x.T @ q))`. Continuous
quantities and balanced routing circulations are permitted, matching the arc
MILP. The historical strict `validate_arc_solution` uses these same checks.

Constructing a problem validates its components. A historical baseline is
initialized lazily when requested. This particular placement does not prove
feasibility; solve and validate determine the status of actual candidates.

## Command line and results

```bash
service-placement --seed 42
service-placement --instance tree --solvers fast_heuristic heuristic --seed 42 --start-users 20 --stop-users 61 --step-users 20 --output artifacts/data/api_tree.csv
```

Without arguments the CLI solves one tree instance with 20 users and the fast
heuristic, and writes CSV without plots. `--stop-users` is an exclusive bound.
`--plots` requests figures and requires the `benchmarks` extra. API and CLI use
the same registry, result objects and validator.

CSV columns are `instance,solver,users,neighbors,groups,score,runtime,buildtime,`
`status,feasible,seed,error`. Invalid candidates have an empty score; execution
errors have status `error`. A campaign returns exit code 1 when a run has no
feasible candidate or an execution error, and 2 for CLI/configuration errors.
Existing CSVs with a different header are rejected with an actionable message;
choose a new `--output` path to retain older experiment results.
