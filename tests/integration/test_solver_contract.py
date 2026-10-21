"""Every registered backend initializes the explicit base-class contract."""

import pytest

from service_placement import make_tree_instance
from service_placement.utils.solver_base import GenericSolver
from service_placement.utils.solver_registry import SOLVER_MODULES, make_solver
from tests.optional import is_rl_module, module_requirements


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=[
                pytest.mark.requires(*module_requirements(module)),
                *([pytest.mark.rl] if is_rl_module(module) else []),
            ],
        )
        for name, module in SOLVER_MODULES.items()
    ],
)
def test_backend_initializes_output_contract(name):
    problem = make_tree_instance(2, neighbors=0)
    backend = make_solver(name, problem, seed=42)
    assert backend.problem_instance is problem
    assert backend.x_var is backend.y_var is backend.q_var is None
    assert backend.score is backend.runtime is backend.termination_condition is None
    assert backend.build_time == 0.0


def test_subclass_constructor_is_not_replaced():
    def constructor(self, problem, *, option=1):
        GenericSolver.__init__(self, problem)
        self.option = option

    class Backend(GenericSolver):
        __init__ = constructor

    assert Backend.__init__ is constructor
    backend = Backend(make_tree_instance(2), option=7)
    assert backend.option == 7 and backend.q_var is None
