"""Figure generation for simulation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class FigureGenerator:
    """Generate figures for simulation results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from SimulationResults.to_dataframe().
    output_dir : str | Path
        Directory to save figures.
    style : str
        Matplotlib style. Default: "seaborn-v0_8-whitegrid".
    """

    def __init__(
        self,
        results_df: pd.DataFrame,
        output_dir: str | Path = "figures",
        style: str | None = None,
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for figure generation")

        self.results_df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if style and style in plt.style.available:
            plt.style.use(style)

    def plot_ate_distribution(
        self,
        methods: list[str] | None = None,
        figsize: tuple[int, int] = (10, 6),
        save: bool = True,
    ) -> plt.Figure:
        """Plot ATE estimate distribution by method.

        Parameters
        ----------
        methods : list[str], optional
            Methods to include.
        figsize : tuple
            Figure size.
        save : bool
            Whether to save figure.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        if methods is None:
            methods = self.results_df["method"].unique().tolist()

        fig, ax = plt.subplots(figsize=figsize)

        true_ate = self.results_df["true_ate"].iloc[0]

        for i, method in enumerate(methods):
            method_df = self.results_df[self.results_df["method"] == method]
            estimates = method_df["ate_estimate"].values

            # Histogram
            ax.hist(
                estimates,
                bins=20,
                alpha=0.5,
                label=method,
                density=True,
            )

        # True value line
        ax.axvline(
            true_ate, color="black", linestyle="--", linewidth=2, label="True ATE"
        )

        ax.set_xlabel("ATE Estimate")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of ATE Estimates")
        ax.legend()

        if save:
            fig.savefig(self.output_dir / "ate_distribution.pdf", bbox_inches="tight")
            fig.savefig(self.output_dir / "ate_distribution.png", dpi=150, bbox_inches="tight")

        return fig

    def plot_coverage_comparison(
        self,
        figsize: tuple[int, int] = (8, 5),
        save: bool = True,
    ) -> plt.Figure:
        """Plot coverage rates by method.

        Parameters
        ----------
        figsize : tuple
            Figure size.
        save : bool
            Whether to save figure.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        coverage_by_method = (
            self.results_df.groupby("method")["ate_covered"].mean()
        )

        methods = coverage_by_method.index.tolist()
        coverages = coverage_by_method.values

        bars = ax.bar(methods, coverages, color="steelblue", alpha=0.7)

        # Nominal coverage line
        ax.axhline(0.95, color="red", linestyle="--", label="Nominal (95%)")

        # Acceptable range
        ax.axhspan(0.90, 0.98, alpha=0.1, color="green", label="Acceptable range")

        ax.set_xlabel("Method")
        ax.set_ylabel("Coverage Rate")
        ax.set_title("95% CI Coverage by Method")
        ax.set_ylim(0, 1.05)
        ax.legend()

        # Add value labels
        for bar, cov in zip(bars, coverages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{cov:.1%}",
                ha="center",
                fontsize=10,
            )

        if save:
            fig.savefig(self.output_dir / "coverage_comparison.pdf", bbox_inches="tight")
            fig.savefig(self.output_dir / "coverage_comparison.png", dpi=150, bbox_inches="tight")

        return fig

    def plot_ite_correlation(
        self,
        method: str,
        rep_id: int = 0,
        figsize: tuple[int, int] = (6, 6),
        save: bool = True,
    ) -> plt.Figure:
        """Plot ITE correlation scatter.

        Note: Requires access to true ITEs which are in full results.

        Parameters
        ----------
        method : str
            Method to plot.
        rep_id : int
            Replication to plot.
        figsize : tuple
            Figure size.
        save : bool
            Whether to save figure.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        method_df = self.results_df[
            (self.results_df["method"] == method) &
            (self.results_df["rep_id"] == rep_id)
        ]

        corr = method_df["ite_correlation"].iloc[0]

        # Without actual ITE values, just show correlation info
        ax.text(
            0.5, 0.5,
            f"ITE Correlation: {corr:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.set_title(f"{method} - ITE Estimation Quality")

        if save:
            fig.savefig(
                self.output_dir / f"ite_correlation_{method}.pdf",
                bbox_inches="tight",
            )

        return fig

    def plot_method_comparison(
        self,
        metric: str = "ite_rmse",
        figsize: tuple[int, int] = (10, 6),
        save: bool = True,
    ) -> plt.Figure:
        """Plot boxplot comparison of methods.

        Parameters
        ----------
        metric : str
            Metric to compare.
        figsize : tuple
            Figure size.
        save : bool
            Whether to save figure.

        Returns
        -------
        plt.Figure
            Generated figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        methods = self.results_df["method"].unique()
        data = [
            self.results_df[self.results_df["method"] == m][metric].values
            for m in methods
        ]

        bp = ax.boxplot(data, labels=methods, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Method")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Method")

        if save:
            fig.savefig(
                self.output_dir / f"comparison_{metric}.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                self.output_dir / f"comparison_{metric}.png",
                dpi=150,
                bbox_inches="tight",
            )

        return fig

    def generate_all(self) -> list[plt.Figure]:
        """Generate all standard figures."""
        figures = []

        figures.append(self.plot_ate_distribution())
        figures.append(self.plot_coverage_comparison())
        figures.append(self.plot_method_comparison("ite_rmse"))
        figures.append(self.plot_method_comparison("ate_bias"))

        return figures


def generate_all_figures(
    results_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate all figures from saved results.

    Parameters
    ----------
    results_dir : str | Path
        Directory containing result CSV files.
    output_dir : str | Path
        Directory to save figures.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    for csv_file in results_dir.glob("**/*.csv"):
        df = pd.read_csv(csv_file)

        # Create output subdirectory
        rel_path = csv_file.relative_to(results_dir)
        fig_dir = output_dir / rel_path.parent / rel_path.stem

        generator = FigureGenerator(df, fig_dir)
        generator.generate_all()

        print(f"Generated figures for {rel_path}")
