"""Mobile edge computing service placement benchmarks."""

__version__ = "0.1.0"

from service_placement.api import (
    SolveResult,
    SolveStatus,
    make_star_instance,
    make_tree_instance,
    solve,
)
from service_placement.domain.conversions import SolutionConversionError, to_arc_solution
from service_placement.domain.problem import ProblemInstance
from service_placement.domain.problem_arrays import ProblemArrays
from service_placement.domain.solution import ArcSolution, CompactSolution, Solution
from service_placement.domain.validation import (
    SolutionValidationError,
    ValidationReport,
    Violation,
    validate,
)
from service_placement.errors import (
    MissingSolverDependencyError,
    SolverExecutionError,
    UnknownSolverError,
)
from service_placement.utils.solver_registry import list_solvers

__all__ = [
    "make_tree_instance",
    "make_star_instance",
    "solve",
    "validate",
    "list_solvers",
    "ProblemInstance",
    "ProblemArrays",
    "Solution",
    "ArcSolution",
    "CompactSolution",
    "to_arc_solution",
    "SolutionConversionError",
    "SolveResult",
    "SolveStatus",
    "ValidationReport",
    "Violation",
    "SolutionValidationError",
    "UnknownSolverError",
    "MissingSolverDependencyError",
    "SolverExecutionError",
]
