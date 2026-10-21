"""Gym environments sharing a problem must own their graphs and routing caches."""

import copy

import numpy as np
import pytest

from service_placement import make_tree_instance
from service_placement.domain.loads import residual_arrays

pytestmark = pytest.mark.requires("gymnasium")


def assert_graph_attributes_equal(graph, snapshot):
    assert dict(graph.nodes(data=True)) == dict(snapshot.nodes(data=True))
    assert dict(((u, v), attributes) for u, v, attributes in graph.edges(data=True)) == dict(
        ((u, v), attributes) for u, v, attributes in snapshot.edges(data=True)
    )


def test_shared_problem_environments_are_isolated():
    from service_placement.environments.spp_env import SPPEnv

    problem = make_tree_instance(2, neighbors=0)
    original_graph = copy.deepcopy(problem.network_graph.nx_graph)
    first = SPPEnv(problem, warm_start_heuristic=False)
    second = SPPEnv(problem, warm_start_heuristic=False)
    try:
        first.reset(seed=42)
        second_observation, _ = second.reset(seed=42)
        second_graph = copy.deepcopy(second.graph)
        assert first.graph is first.solution_checker.graph
        assert first.path_cache is first.solution_checker.path_cache
        assert first.graph is not second.graph and first.path_cache is not second.path_cache
        assert first.graph is not problem.network_graph.nx_graph
        assert first.graph.number_of_edges() == len(problem.network_graph.arc_list)
        before = first.state.copy()
        observation, *_ = first.step((0, 0, 2, 0, 1, 0))
        assert not np.array_equal(before, first.state)
        assert first.observation_space.contains(observation)
        assert (
            first.graph.edges[2, 3]["remaining_capacity"]
            != second.graph.edges[2, 3]["remaining_capacity"]
        )
        loads = residual_arrays(problem, *first._state_to_variables(first.state))
        for a, arc in enumerate(problem.network_graph.arc_list):
            assert first.graph.edges[arc]["latency"] == loads["arc_latency"][a]
        assert_graph_attributes_equal(problem.network_graph.nx_graph, original_graph)
        assert_graph_attributes_equal(second.graph, second_graph)
        np.testing.assert_array_equal(second._observe(second.state), second_observation)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("name", ["PlacementRoutingEnv", "FlowSetupEnv"])
def test_hierarchical_environments_share_the_isolation_contract(name):
    from service_placement.environments import hrl_env

    problem = make_tree_instance(2, neighbors=0)
    original = copy.deepcopy(problem.network_graph.nx_graph)
    env = getattr(hrl_env, name)(problem, warm_start_heuristic=False)
    try:
        observation, _ = env.reset(seed=42)
        assert env.graph is env.solution_checker.graph
        assert env.path_cache is env.solution_checker.path_cache
        assert env.observation_space.contains(observation)
        assert_graph_attributes_equal(problem.network_graph.nx_graph, original)
    finally:
        env.close()


def test_reset_on_another_instance_rebuilds_an_isolated_workspace():
    from service_placement.environments.spp_env import SPPEnv

    first_problem = make_tree_instance(2, neighbors=0)
    second_problem = first_problem.with_data(arc_capacities=[20] * first_problem.data.n_arcs)
    first_graph = copy.deepcopy(first_problem.network_graph.nx_graph)
    second_graph = copy.deepcopy(second_problem.network_graph.nx_graph)
    env = SPPEnv([first_problem, second_problem], warm_start_heuristic=False)
    try:
        old_workspace = env.solution_checker
        # Seed 0 selects index 1 for two instances.
        observation, _ = env.reset(seed=0)
        assert env.problem_instance is second_problem
        assert env.solution_checker is not old_workspace
        assert env.graph is env.solution_checker.graph
        assert env.path_cache is env.solution_checker.path_cache
        assert env.observation_space.contains(observation)
        env.reset(seed=42, initial_instance=True)
        assert env.problem_instance is first_problem
        assert_graph_attributes_equal(first_problem.network_graph.nx_graph, first_graph)
        assert_graph_attributes_equal(second_problem.network_graph.nx_graph, second_graph)
    finally:
        env.close()
