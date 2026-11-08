"""Promotion gate: a candidate replaces Production only if it is strictly better on the holdout."""


def should_promote(
    candidate_rmse: float, production_rmse: float | None, min_improvement: float = 0.0
) -> bool:
    """Lower RMSE is better. No Production model yet -> promote. Otherwise require
    candidate < production * (1 - min_improvement), so ties and regressions never ship."""
    if production_rmse is None:
        return True
    return candidate_rmse < production_rmse * (1.0 - min_improvement)
