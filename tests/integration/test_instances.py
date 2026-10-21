"""
Minimal integration test for all instance builders.

Assumes:
- validate_components()
- validate_instance()

raise exceptions on failure.
"""

from pathlib import Path

import networkx as nx

from service_placement.instances.synthetic import (
    create_gml_problem_instance,
    create_large_problem_instance,
    create_star_problem_instance,
    create_tree_problem_instance,
)


def test_tree_instance():
    instance = create_tree_problem_instance(
        n_users=10,
        neighbors=2,
    )

    instance.validate_components()
    instance.validate_instance()

    print("Tree instance: OK")


def test_star_instance():
    instance = create_star_problem_instance(
        n_groups=3,
        users_per_group=8,
        neighbors=2,
    )

    instance.validate_components()
    instance.validate_instance()

    print("Star instance: OK")


def test_gml_instance():
    instance = create_gml_problem_instance(
        gml_path=Path(__file__).resolve().parents[2] / "data/topologies/france.gml",
        n_services=4,
        demands_per_service=4,
    )

    instance.validate_components()
    instance.validate_instance()

    print("GML instance: OK")


def test_large_instance():
    # Small synthetic base graph
    base_graph = nx.cycle_graph(5)

    instance = create_large_problem_instance(
        base_graph=base_graph,
        n_services=5,
        demands_per_service=5,
        edge_per_core=3,
        users_per_edge=2,
    )

    instance.validate_components()
    instance.validate_instance()

    print("Large instance: OK")


def run_all_tests():
    test_tree_instance()
    test_star_instance()
    test_gml_instance()
    test_large_instance()

    print("\nAll instance tests passed.")


if __name__ == "__main__":
    run_all_tests()
