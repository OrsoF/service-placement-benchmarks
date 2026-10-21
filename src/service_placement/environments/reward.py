from service_placement.domain.problem import ProblemInstance
from service_placement.environments.state_codec import state_to_vectors


def served_quantity_score(problem_instance: ProblemInstance, state: list) -> float:
    """Score a state by its served demand quantities."""
    return float(sum(state_to_vectors(problem_instance, state)[2]))
