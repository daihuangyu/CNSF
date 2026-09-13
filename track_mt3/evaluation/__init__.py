from .complexity import ComplexityEstimate, paper_complexity
from .evaluator import EvaluationResult, evaluate_model, load_trajectory

__all__ = [
    "BaselineEvaluation", "ComplexityEstimate", "EvaluationResult", "evaluate_model",
    "evaluate_set_baselines", "load_trajectory", "paper_complexity",
]


def __getattr__(name: str):
    """Load optional classical-baseline helpers only when requested."""
    if name in {"BaselineEvaluation", "evaluate_set_baselines"}:
        from .baseline_evaluator import BaselineEvaluation, evaluate_set_baselines

        return {
            "BaselineEvaluation": BaselineEvaluation,
            "evaluate_set_baselines": evaluate_set_baselines,
        }[name]
    raise AttributeError(name)
