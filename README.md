# Service Placement Benchmarks

**Benchmark service placement strategies before integrating them into an edge network orchestrator.**

`service_placement` compares strategies for hosting services, routing traffic, and serving
demand under latency, compute, and link-capacity constraints. Train and evaluate
reinforcement learning alongside mathematical optimization and heuristics, using
one Python API and a shared feasibility validator.

Based on **IFIP Networking 2025** research, it gives telecom and industrial R&D
teams a runnable framework for evaluating orchestration strategies.

![Validated service placement and directed routing on an edge network](docs/assets/service-placement-routing.png)

*A solved synthetic edge scenario: orange outlines mark service hosts, arrows
show routing, and colors indicate link usage. The right panel follows one
service's demands.*

## From a network scenario to an engineering decision

- **Compare approaches:** mathematical optimization, heuristics, evolutionary
  search, and RL (PPO, A2C, hierarchical PPO).
- **Explore constraints:** vary network size, latency budgets, compute resources,
  and link capacities.
- **Check every candidate:** shared checks cover placement, routing, flows,
  capacity, resources, and latency. Invalid candidates receive no score.
- **Export results:** scores, runtimes, feasibility, status, seeds, and errors
  in CSV, with plots for technical reviews.

## Compare strategies on the same scenario

Compare traffic served and computation time on shared network inputs to select
strategies for further integration.

![Two heuristics compared on the same synthetic network scenarios](docs/assets/benchmark-overview.png)

*Two heuristics on synthetic trees with 10, 20, and 40 users, across three search
seeds. Lines: means; shading: observed range. All 18 runs passed validation;
timings depend on the host. See the
[measurements](docs/assets/benchmark-results.csv) and
[protocol](docs/assets/benchmark-protocol.json).*

The notebook also includes a separate
[scaling campaign from 40 to 400 users](docs/assets/large-benchmark-overview.png)
and a [400-user placement and routing view](docs/assets/large-service-placement-routing.png),
with [measurements](docs/assets/large-benchmark-results.csv) and a
[documented search budget](docs/assets/large-benchmark-protocol.json).

## Run your first comparison

From a clone of this repository, with **Python 3.10+**:

```bash
python -m pip install -e .
service-placement --instance tree --solvers fast_heuristic heuristic --seed 42 --start-users 10 --stop-users 41 --step-users 10 --output artifacts/data/first_comparison.csv
```

This compares 10, 20, 30, and 40 users and exports CSV results. Higher scores mean
more total served flow on the same instance. The base package needs NumPy and
NetworkX; no GPU or optimizer license.

Use the [notebook](examples/benchmark_walkthrough.ipynb) for network views,
solver comparisons, and the scaling campaign. It exports data and configuration.
Install its plotting and Jupyter dependencies with:

```bash
python -m pip install -e ".[benchmarks]" jupyterlab
jupyter lab examples/benchmark_walkthrough.ipynb
```

## Adapt the benchmark to your constraints

Adjust capacities, resources, and latency budgets through the Python API.
Here, both methods receive the same variant with double link capacity and
20% tighter latency limits:

```python
from service_placement import make_tree_instance, solve

problem = make_tree_instance(n_users=20)
scenario = problem.with_data(
    arc_capacities=problem.data.arc_capacities * 2,
    latency_limits=problem.data.latency_limits * 0.8,
)
for method in ("fast_heuristic", "heuristic"):
    result = solve(scenario, solver=method, seed=42)
    print(method, result.objective, result.runtime, result.validation.feasible)
```

See [input data and configuration](docs/api.md#problem-data) for additional
parameters to adapt the scenarios in your evaluation pipeline.

## Execute in your environment

Docker packages the benchmark as a CLI job, with results persisted to a mounted
directory:

```bash
docker build -t service-placement .
docker run --rm -v "${PWD}/artifacts:/app/artifacts" service-placement --instance tree --solvers fast_heuristic heuristic --seed 42 --start-users 10 --stop-users 41 --step-users 10
```

Deploy the toolkit and runner as evaluation jobs; connecting their decisions to
a live orchestrator requires an adapter for the target system. RL/evolutionary
dependencies are optional; mathematical backends require Gurobi and a license.
See [installation profiles](docs/experiments.md).

## Documentation and research

- [API, solver options, and validation](docs/api.md)
- [Benchmark campaigns and Docker](docs/experiments.md)
- [Development, tests, and adding a solver](docs/development.md)
- [Architecture](docs/architecture.md) · [Research directions](docs/research.md)

Research foundation: *Faster Latency-Constrained Service Placement in Edge
Computing with Deep Reinforcement Learning*, IFIP Networking 2025.
Project code is licensed under [MIT](LICENSE).
