# Architecture and migration

`service_placement` is installed from `src/`. Runtime imports use that namespace;
there is no need to add the repository root to `sys.path`.

| Previous module | Current module |
| --- | --- |
| `core.problem_classes.*` | `service_placement.domain.problem` |
| `core.solution`, `core.problem_classes.solving_instance` | `service_placement.domain.solution` |
| `core.feasibility` | `service_placement.domain.validation` |
| `core.generic_solver` | `service_placement.utils.solver_base` |
| `core.pyomo_model` | `service_placement.utils.ilp_solvers` |
| `instances.builder` | `service_placement.instances.synthetic` |
| `instances.graph_build` | `service_placement.instances.topology` |
| `solvers.heuristic`, `service_placement.solvers.heuristics.*` | `service_placement.solvers.*` |
| `solvers.exact`, `solvers.relaxations`, `service_placement.solvers.exact.*`, `service_placement.solvers.relaxations.*` | `service_placement.solvers.*` |
| `solvers.evolutionary`, `solvers.rl`, `service_placement.solvers.evolutionary.*`, `service_placement.solvers.rl.*` | `service_placement.solvers.*` |
| `service_placement.solvers.base`, `service_placement.solvers.contracts` | `service_placement.utils.solver_base` |
| `service_placement.solvers.registry` | `service_placement.utils.solver_registry` |
| `service_placement.solvers.workspace`, `service_placement.solvers.graph_state` | `service_placement.utils.solver_workspace` |
| `service_placement.solvers.paths` | `service_placement.utils.solver_paths` |
| `service_placement.solvers.search`, `service_placement.solvers.initialization` | `service_placement.utils.heuristic_solvers` |
| `service_placement.solvers.model_utils`, `service_placement.solvers.exact.extraction` | `service_placement.utils.ilp_solvers` |
| `service_placement.solvers.rl.base_solver` | `service_placement.utils.rl_solvers` |
| `service_placement.solvers.arc_formulation`, `service_placement.solvers.path_formulation` | `service_placement.utils.arc_formulation`, `service_placement.utils.path_formulation` |
| `rl.*` | `service_placement.environments.*` |
| `data_io.experiment_io`, `service_placement.benchmarks.io` | `service_placement.compat.experiment_io` |
| `service_placement.benchmarks.pickle_io` | `service_placement.compat.pickle_io` |
| `plotting.result_plots` | `service_placement.benchmarks.reporting` |
| `exp.py` | `service_placement.cli`, `service_placement.benchmarks.runner` |

The domain defines demands, networks, services, packages, and problems.
`domain.network` owns graph data, `domain.services` owns demands and services,
and `domain.problem` assembles a problem. `domain.problem_arrays.ProblemArrays`
owns immutable numerical vectors, dimensions, demand-to-service indices and the
arc ordering. `ProblemInstance.data` builds this snapshot once, and every solver
receives that same object through `GenericSolver.data`. Validation, loads and
workspaces consume it too. Construction inputs cannot change the snapshot;
`ProblemInstance.with_data` creates a new validated problem. Compatibility
properties delegate to the snapshot. The historical `service_package` is
construction metadata, not a second numerical source. A private graph copy
preserves topology and metadata; its topology is frozen. `domain.solution.Solution` owns copied,
read-only arrays. Pure compact/arc conversion lives in `domain.conversions`;
Load, residual and total-traffic latency calculations live in `domain.loads`
and are shared by validation, reporting and graph updates.

`solvers/` is flat: each of the 21 registered solvers has one module defining
its `Solver` class and its specific constructor options. Historical solver names
remain accepted by `solve` and the CLI as aliases of the clearer canonical names.
A2C, PPO and the PPO reset variant
own their constructors; shared training lives in `utils.rl_solvers.RLSolver`.

`utils.solver_workspace.SolverWorkspace` owns a mutable graph, candidate state,
path cache, and independent random generator. Its module also applies graph
updates. Candidate search and baseline construction live in
`utils.heuristic_solvers`; optimizer options and indexed output extraction live
in `utils.ilp_solvers`. Its historical `ProblemArrays` and `problem_arrays`
imports reexport their canonical definitions from the domain. Mathematical
backends read shared parameters directly; only paths, formulation bounds and
optimizer state are stored per solver. Arc and path formulations retain their
own utility modules. Workspaces store their results without changing the
problem's historical solution state. RL environments use the workspace's copy
of the directed graph and its path cache, so observations never mutate the input
problem or another environment. `GenericSolver` initializes outputs through
an ordinary constructor; subclasses explicitly call `super().__init__`.

All feasibility decisions use `domain.validation`; its exception is
`SolutionValidationError`, a `ValueError`. Flow conservation and shared arc
latency follow the arc MILP, including its allowance of balanced circulations.

Historical imports from `domain.problem` and `domain.solution` remain available
as reexports. `SolvingInstance` aliases `SolverWorkspace`. Historical mutable
`ProblemInstance.x_variable`, `y_variable`, `q_variable`, `_set_arc_solution`, and
`validate_instance` live in `compat.problem_state`; they initialize a baseline
only on first access. `validate_instance` checks that particular baseline and
does not prove that the problem is infeasible. New code should use `solve` and
`validate`, and import helpers from their canonical modules.

The public facade in `api.py` exports instance builders and `solve`. Results
carry a copied `Solution`, a `ValidationReport`, termination status and objective
recomputed from validated quantities. `utils.solver_registry` provides the shared
lazy registry, including historical aliases and explicit optional-dependency and
mathematical-backend metadata. It no longer infers families from folder names.
The CLI and benchmark runner use
this facade. See [the API guide](api.md) for the public contracts.

Old solver module paths are replaced by the flat paths in the table above;
in-repository imports, archived notebook sources and typing targets have been
updated. `utils/__init__.py` does not import the optional solver families, so
using the base API does not require Pyomo or Stable-Baselines3.

Instance construction, environments, and solvers depend on the domain.
Benchmark orchestration depends on those components, validates solver outputs,
and records results through `metrics`. Reporting is imported only when plots
are requested. CLI imports do not require Pyomo, Gurobi, or RL packages.

Tests are organized by purpose: constraint tests in `unit`, component and CLI
tests in `integration`, and shared-load counterexamples plus comparison with
actual Pyomo MILP constraints in `regression`. The CI runs minimal and optional
dependency profiles on Windows and Linux with Python 3.10 and 3.12, runs a
separate RL profile, builds distributions, and checks Docker. Ruff covers active
Python sources, tests, and examples; strict mypy coverage grows from an explicit
list of public contracts and adapters. See [development](development.md).

`examples/` contains the quickstart, solver smoke runner and environment rendering
example. The former `exp.py`, `exp_tree.py`, `exp_star.py`, and `reset.py` wrappers
were removed; use the CLI for tree/star experiments. The historical experiment
API and its single pickle implementation live in `compat/`. Archived scripts
and notebooks reference those modules; the archived allocation helper lives
beside its only caller in `archive/legacy/service_placement.py`.

Unused `instances.utility`, `ToyInstance`, GML/path reconstruction helpers,
old graph-update functions and unused plotting/reward aliases were removed.
`data/`, `artifacts/`,
and `archive/` retain input topologies, experiment outputs, and historical work.
Archived imports were updated; historical algorithms and saved notebook outputs
are retained. Old module namespaces are removed. Previously pickled Python
objects that reference those namespaces must be regenerated or migrated before
loading; filenames alone cannot migrate pickle class references.

File names are lowercase except the standard project files `README.md`,
`LICENSE`, `Dockerfile`, and `MANIFEST.in`. The setuptools manifest includes
tests, examples, documentation, and topology files in source distributions.
