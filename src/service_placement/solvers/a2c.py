"""A2C reinforcement-learning solver."""

from service_placement.domain.problem import ProblemInstance
from service_placement.utils.rl_solvers import RLSolver


class Solver(RLSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int = 5000,
        train_step: int = 50000,
        evaluation_step: int = 10000,
        method: str = "A2C",
        seed: int = 0,
    ):
        """Initialize the default A2C RL solver."""
        super().__init__(
            problem,
            max_step,
            train_step,
            evaluation_step,
            method,
            seed=seed,
        )
