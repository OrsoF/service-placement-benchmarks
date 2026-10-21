"""Run after installing service-placement-benchmarks; no optional extras needed."""

from service_placement import make_tree_instance, solve, validate


def main():
    problem = make_tree_instance(n_users=20, seed=42)
    result = solve(problem, solver="fast_heuristic", seed=42)
    report = validate(problem, result.solution)
    print(result.objective, report.feasible)


if __name__ == "__main__":
    main()
