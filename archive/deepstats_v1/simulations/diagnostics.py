"""Overfitting and underfitting diagnostics.

This module provides tools to diagnose fitting issues from loss curves:
- Overfitting: validation loss diverges from training loss
- Underfitting: both losses remain high
- Good fit: both losses converge to similar low values
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class FittingDiagnosis(str, Enum):
    """Fitting diagnosis categories."""

    OVERFIT = "overfit"
    UNDERFIT = "underfit"
    GOOD_FIT = "good_fit"
    EARLY_STOPPING_NEEDED = "early_stopping_needed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class LossCurveAnalysis:
    """Analysis of training and validation loss curves."""

    diagnosis: FittingDiagnosis
    train_final: float
    val_final: float
    train_min: float
    val_min: float
    generalization_gap: float  # val_final - train_final
    relative_gap: float  # gap / train_final
    val_increasing: bool  # Is validation loss increasing at end?
    optimal_epoch: int  # Epoch with minimum validation loss
    convergence_epoch: int  # Epoch where training loss stabilized
    explanation: str


def analyze_loss_curves(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    gap_threshold: float = 0.3,
    underfit_threshold: float = 0.8,
    window_size: int = 10,
) -> LossCurveAnalysis:
    """Analyze training and validation loss curves.

    Parameters
    ----------
    train_loss : array-like
        Training loss at each epoch.
    val_loss : array-like
        Validation loss at each epoch.
    gap_threshold : float, default=0.3
        Relative gap threshold for overfitting detection.
    underfit_threshold : float, default=0.8
        Final loss threshold (relative to initial) for underfitting.
    window_size : int, default=10
        Window for detecting increasing validation loss.

    Returns
    -------
    LossCurveAnalysis
        Detailed analysis of the loss curves.
    """
    train_loss = np.asarray(train_loss)
    val_loss = np.asarray(val_loss)

    n_epochs = len(train_loss)

    # Basic statistics
    train_final = float(train_loss[-1])
    val_final = float(val_loss[-1])
    train_min = float(np.min(train_loss))
    val_min = float(np.min(val_loss))
    train_initial = float(train_loss[0])
    val_initial = float(val_loss[0])

    # Generalization gap
    gap = val_final - train_final
    relative_gap = gap / train_final if train_final > 0 else np.inf

    # Check if validation loss is increasing at the end
    if n_epochs > window_size:
        recent_val = val_loss[-window_size:]
        val_slope = np.polyfit(range(window_size), recent_val, 1)[0]
        val_increasing = val_slope > 0
    else:
        val_increasing = val_loss[-1] > val_loss[0]

    # Optimal epoch (minimum validation loss)
    optimal_epoch = int(np.argmin(val_loss))

    # Convergence epoch (when training loss stabilized)
    convergence_epoch = _find_convergence_epoch(train_loss)

    # Diagnosis logic
    diagnosis, explanation = _diagnose(
        train_final=train_final,
        val_final=val_final,
        train_initial=train_initial,
        val_initial=val_initial,
        relative_gap=relative_gap,
        val_increasing=val_increasing,
        optimal_epoch=optimal_epoch,
        n_epochs=n_epochs,
        gap_threshold=gap_threshold,
        underfit_threshold=underfit_threshold,
    )

    return LossCurveAnalysis(
        diagnosis=diagnosis,
        train_final=train_final,
        val_final=val_final,
        train_min=train_min,
        val_min=val_min,
        generalization_gap=gap,
        relative_gap=relative_gap,
        val_increasing=val_increasing,
        optimal_epoch=optimal_epoch,
        convergence_epoch=convergence_epoch,
        explanation=explanation,
    )


def _find_convergence_epoch(
    loss: np.ndarray,
    patience: int = 10,
    threshold: float = 0.01,
) -> int:
    """Find epoch where loss converged (stopped improving significantly)."""
    best_loss = loss[0]
    epochs_without_improvement = 0

    for i, l in enumerate(loss):
        if l < best_loss * (1 - threshold):
            best_loss = l
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            return max(0, i - patience)

    return len(loss) - 1


def _diagnose(
    train_final: float,
    val_final: float,
    train_initial: float,
    val_initial: float,
    relative_gap: float,
    val_increasing: bool,
    optimal_epoch: int,
    n_epochs: int,
    gap_threshold: float,
    underfit_threshold: float,
) -> tuple[FittingDiagnosis, str]:
    """Determine fitting diagnosis from loss curve statistics."""

    # Calculate relative improvement
    train_improvement = 1 - train_final / train_initial if train_initial > 0 else 0
    val_improvement = 1 - val_final / val_initial if val_initial > 0 else 0

    # Overfitting: large gap AND validation increasing
    if relative_gap > gap_threshold and val_increasing:
        return (
            FittingDiagnosis.OVERFIT,
            f"Overfitting detected: validation loss {relative_gap:.1%} higher than "
            f"training loss, and validation loss is increasing. Consider reducing "
            f"model capacity, adding regularization, or using early stopping at "
            f"epoch {optimal_epoch}.",
        )

    # Early stopping needed: validation minimum before final epoch
    if optimal_epoch < n_epochs - 1 and val_increasing:
        return (
            FittingDiagnosis.EARLY_STOPPING_NEEDED,
            f"Early stopping recommended: optimal validation loss at epoch "
            f"{optimal_epoch}, but training continued for {n_epochs - optimal_epoch} "
            f"more epochs with increasing validation loss.",
        )

    # Underfitting: poor improvement on both train and validation
    if train_improvement < (1 - underfit_threshold) and val_improvement < (
        1 - underfit_threshold
    ):
        return (
            FittingDiagnosis.UNDERFIT,
            f"Underfitting detected: training loss only improved by "
            f"{train_improvement:.1%} and validation by {val_improvement:.1%}. "
            f"Consider increasing model capacity, training longer, or checking "
            f"the learning rate.",
        )

    # Good fit: reasonable improvement, small gap
    if train_improvement > 0.3 and relative_gap < gap_threshold:
        return (
            FittingDiagnosis.GOOD_FIT,
            f"Good fit: training loss improved by {train_improvement:.1%}, "
            f"generalization gap is {relative_gap:.1%}. Model is learning "
            f"the signal without overfitting.",
        )

    # Inconclusive
    return (
        FittingDiagnosis.INCONCLUSIVE,
        f"Inconclusive: training improved by {train_improvement:.1%}, "
        f"validation improved by {val_improvement:.1%}, "
        f"generalization gap is {relative_gap:.1%}. "
        f"Consider running for more epochs or adjusting hyperparameters.",
    )


def diagnose_fitting(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
) -> FittingDiagnosis:
    """Simple interface to get fitting diagnosis.

    Parameters
    ----------
    train_loss : array-like
        Training loss at each epoch.
    val_loss : array-like
        Validation loss at each epoch.

    Returns
    -------
    FittingDiagnosis
        One of: OVERFIT, UNDERFIT, GOOD_FIT, EARLY_STOPPING_NEEDED, INCONCLUSIVE
    """
    analysis = analyze_loss_curves(train_loss, val_loss)
    return analysis.diagnosis


def compute_early_stopping_metrics(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    patience: int = 10,
) -> dict:
    """Compute metrics for early stopping analysis.

    Parameters
    ----------
    train_loss : array-like
        Training loss at each epoch.
    val_loss : array-like
        Validation loss at each epoch.
    patience : int, default=10
        Number of epochs to wait for improvement.

    Returns
    -------
    dict
        Dictionary with early stopping metrics.
    """
    val_loss = np.asarray(val_loss)
    train_loss = np.asarray(train_loss)

    optimal_epoch = int(np.argmin(val_loss))
    val_at_optimal = float(val_loss[optimal_epoch])
    train_at_optimal = float(train_loss[optimal_epoch])

    # How much worse is final vs optimal?
    val_degradation = (val_loss[-1] - val_at_optimal) / val_at_optimal if val_at_optimal > 0 else 0

    return {
        "optimal_epoch": optimal_epoch,
        "total_epochs": len(val_loss),
        "wasted_epochs": len(val_loss) - optimal_epoch - 1,
        "val_at_optimal": val_at_optimal,
        "val_at_final": float(val_loss[-1]),
        "val_degradation": float(val_degradation),
        "train_at_optimal": train_at_optimal,
        "gap_at_optimal": train_at_optimal - val_at_optimal,
    }
