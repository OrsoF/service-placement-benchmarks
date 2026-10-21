# Benchmark experiments

Run commands from the repository root after installing `service_placement`.
For the full solver sweeps below, install the optional dependencies:

```bash
python -m pip install -e ".[benchmarks,exact,gurobi,evolutionary,rl]"
```

Exact solvers require a working Gurobi license. See the [API guide](api.md)
for installation options by solver and [development](development.md) for tests.

## Interactive walkthrough

The [benchmark notebook](../examples/benchmark_walkthrough.ipynb) compares two
base-installation heuristics on deterministic tree instances, repeats solver
search across three seeds, and plots served flow and elapsed time. It also
plots actual placements and directed routing on small tree/star networks and
runs a separate star campaign up to 400 users. To run it from a repository clone:

```bash
python -m pip install -e ".[benchmarks]" jupyterlab
jupyter lab examples/benchmark_walkthrough.ipynb
```

Outputs go to `artifacts/readme/`: the figures, a CSV using the CLI result
schema, and JSON recording configuration, package versions, host information,
and the Git revision and working-tree state when available. Rerunning replaces
these notebook outputs instead of appending measurements. Failed runs stay in
the CSV and the feasibility summary; only feasible runs contribute to score
and runtime curves. Bands show the observed minimum/maximum, not confidence
intervals. Solver runtimes include construction and validation; for RL, they
also include training and evaluation.

Edit the parameter cell to change sizes, seeds, topology, or registered solvers.
The default campaign requires no GPU or optimizer license. Additional solver
families need their optional dependencies and, for mathematical solvers, Gurobi
and a license. Changing search seeds does not randomize tree/star instances.

The larger campaign adds access groups of 20 users (40, 100, 200, 400 users).
It uses `fast_heuristic` with 8 iterations, patience 3, 4 candidate nodes, and
2 path candidates at every size, with search seeds 7, 21, and 42. These settings
are independent of the default-budget tree comparison. Edit `LARGE_*` parameters
to change them; the complete notebook can take several minutes on a CPU.

Network figures use validated `Solution.y` placements and `Solution.x` routing
with `Solution.q` flows. Orange node outlines mark hosts, directed arrows carry
actual traffic, arrow widths reflect traffic, and aggregate-view colors show
directional capacity usage. A second panel isolates one service's demands;
locally served demands have zero hops. Large-network labels are limited to
infrastructure nodes, and its service detail shows only actual route nodes.
Positions are schematic; topology, placements, and routes come from the solver.

Additional outputs include:

- `service-placement-routing.png`, `tree-placement-routing.png`, and
  `large-service-placement-routing.png`: solved network views.
- `large-benchmark-overview.png`, `large-benchmark-results.csv`, and
  `large-benchmark-protocol.json`: scaling figures, all outcomes, dimensions,
  served-flow fractions, and configuration.
- `network-placements.csv`, `network-routing.csv`, `network-demands.csv`, and
  `network-solution-protocol.json`: inspectable service hosts, directed arcs,
  flows, latency values/limits, and the source of each drawn solution.

To refresh the README, review feasibility and plots, then copy the reviewed
SVG/PNG figures and CSV/JSON snapshots into `docs/assets/`. Keep captions
consistent with the experiment; committed figures illustrate measured runs,
without a general performance claim.

## Tree and star sweeps

`service_placement` reproduces the tree and star benchmark sweeps. It appends rows to
`artifacts/data/*.csv` and writes objective/runtime plots under
`artifacts/figures/`.

```bash
service-placement --instance tree --solvers ilp_arc heuristic fast_heuristic cma_es ppo hierarchical_ppo --seed 42 --start-users 20 --stop-users 410 --step-users 10 --plots
service-placement --instance star --solvers ilp_arc heuristic fast_heuristic cma_es ppo hierarchical_ppo --seed 42 --groups 4 --start-users 15 --stop-users 125 --step-users 5 --plots
```

`python -m service_placement` invokes the same CLI. Use `--instance tree` or `--instance star`
to select the experiment topology.

The tree experiment builds instances with `create_tree_problem_instance(n_users,
neighbors)`: users are connected locally, then to one wireless access point and
one cloud node.

The star experiment builds instances with `create_star_problem_instance(n_groups,
users_per_group, neighbors)`: users are split into local groups, each group has
one wireless access point, and all groups share the cloud.

Common options:

- `--instance`: `tree` or `star`.
- `--solvers`: registered names; common choices are `ilp_arc`, `heuristic`, `fast_heuristic`, `cma_es`, `ppo`, `hierarchical_ppo`. Use `--help` for all names.
- `--seed`: search seed, recorded in CSV (default 0).
- `--neighbors`: local user-neighborhood parameter.
- `--ilp-time-limit`: Gurobi time limit for `ilp_arc`.
- `--output`: CSV path.
- `--figures-dir`: output directory for plots.
- `--plots`: generate figures (requires the `benchmarks` extra).
- `--no-plots`: append CSV rows without figures (default).

CSV schema:

```text
instance,solver,users,neighbors,groups,score,runtime,buildtime,status,feasible,seed,error
```

An invalid candidate has an empty score and its violations in `error`. Existing
CSV files with an older header require a new `--output` path.

Default figures:

- `artifacts/figures/tree/tree_objective_vs_graph_size.pdf`
- `artifacts/figures/tree/tree_runtime_vs_graph_size.pdf`
- `artifacts/figures/star/star_objective_vs_graph_size.pdf`
- `artifacts/figures/star/star_runtime_vs_graph_size.pdf`

## Docker

```bash
docker build -t service-placement .
docker run --rm service-placement --help
docker run --rm -v "${PWD}/artifacts:/app/artifacts" service-placement --instance tree --solvers fast_heuristic --start-users 2 --stop-users 3 --neighbors 0 --no-plots
```

The default image includes benchmark/reporting dependencies. To include exact
solvers, build with `--build-arg SERVICE_PLACEMENT_EXTRAS=benchmarks,exact,gurobi`; configure
your Gurobi license in the container. RL and evolutionary extras are available
through the same build argument.
