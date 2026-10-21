"""Dependency declarations for flat solver and utility import smoke tests."""

from service_placement.utils.solver_registry import REGISTRY


def is_rl_module(module: str) -> bool:
    return module == "service_placement.utils.rl_solvers" or any(
        spec.module == module and spec.extra == "rl" for spec in REGISTRY.values()
    )


def module_requirements(module: str) -> tuple[str, ...]:
    if is_rl_module(module):
        return ("stable_baselines3",)
    if module in {
        "service_placement.utils.arc_formulation",
        "service_placement.utils.arc_pyomo",
        "service_placement.utils.path_formulation",
    } or any(spec.module == module and spec.mathematical for spec in REGISTRY.values()):
        return ("pyomo",)
    if module == REGISTRY["cma_es"].module:
        return ("cma",)
    if module in {
        "service_placement.environments.spp_env",
        "service_placement.environments.hrl_env",
        "service_placement.environments.observation",
    }:
        return ("gymnasium",)
    if module == "service_placement.compat.experiment_io":
        return ("pandas",)
    if module == "service_placement.benchmarks.reporting":
        return ("matplotlib",)
    return ()
