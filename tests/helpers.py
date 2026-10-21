from types import SimpleNamespace

import networkx as nx
import numpy as np

from service_placement.domain.problem import NetworkGraph


def make_problem(edges=((0, 1),), nodes=2, demands=2, services=1):
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(edges)
    network = NetworkGraph(
        graph, [10] * nodes, [10] * len(edges), [1] * len(edges), [0] * len(edges)
    )
    return SimpleNamespace(
        network_graph=network,
        number_of_service=services,
        total_number_of_demands=demands,
        origins=[0] * demands,
        demand_to_service=[d % services for d in range(demands)],
        needed_resources=[2] * services,
        min_flows=[0] * demands,
        max_flows=[10] * demands,
        latencies=[10] * demands,
    )


def route_solution(problem, paths, placement, quantities):
    x = np.zeros((problem.total_number_of_demands, len(problem.network_graph.arc_list)))
    for d, path in enumerate(paths):
        for edge in zip(path, path[1:], strict=False):
            x[d, problem.network_graph.arc_index_dictionary[edge]] = 1
    y = np.zeros((problem.number_of_service, problem.network_graph.nx_graph.number_of_nodes()))
    y[np.arange(problem.number_of_service), placement] = 1
    return x, y, np.asarray(quantities, dtype=float)
