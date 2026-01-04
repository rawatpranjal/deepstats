#!/usr/bin/env python
"""Master script to run all simulation studies.

Usage:
    python simulations/run_all.py --all
    python simulations/run_all.py --tabular --image
    python simulations/run_all.py --tabular --dgp mixed sparse_nonlinear
    python simulations/run_all.py --n_reps 50 --seed 123

Examples:
    # Run all modalities with default settings (20 reps)
    python simulations/run_all.py --all

    # Run only tabular simulations
    python simulations/run_all.py --tabular

    # Run specific DGPs
    python simulations/run_all.py --tabular --dgp mixed high_dimensional

    # Run with more replications
    python simulations/run_all.py --all --n_reps 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.runners import (
    SimulationConfig,
    TabularSimulationRunner,
    ImageSimulationRunner,
    TextSimulationRunner,
    GraphSimulationRunner,
    TimeSeriesSimulationRunner,
    MultimodalSimulationRunner,
)
from simulations.dgp.tabular import list_tabular_dgps
from simulations.dgp.image import list_image_dgps
from simulations.dgp.text import list_text_dgps
from simulations.dgp.graph import list_graph_dgps
from simulations.dgp.timeseries import list_timeseries_dgps
from simulations.dgp.multimodal import list_multimodal_dgps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simulation studies for DeepHTE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Modality selection
    parser.add_argument(
        "--all", action="store_true", help="Run all modalities"
    )
    parser.add_argument(
        "--tabular", action="store_true", help="Run tabular simulations"
    )
    parser.add_argument(
        "--image", action="store_true", help="Run image simulations"
    )
    parser.add_argument(
        "--text", action="store_true", help="Run text simulations"
    )
    parser.add_argument(
        "--graph", action="store_true", help="Run graph simulations"
    )
    parser.add_argument(
        "--timeseries", action="store_true", help="Run time series simulations"
    )
    parser.add_argument(
        "--multimodal", action="store_true", help="Run multimodal (image+text) simulations"
    )

    # DGP selection
    parser.add_argument(
        "--dgp",
        nargs="+",
        default=None,
        help="Specific DGP names to run (default: all for selected modality)",
    )

    # Simulation parameters
    parser.add_argument(
        "--n_reps", type=int, default=20, help="Number of replications"
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Samples per replication"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["deephte", "causal_forest", "linear_dml"],
        help="Methods to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="simulations/results",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level"
    )

    # Utility flags
    parser.add_argument(
        "--list-dgps", action="store_true", help="List available DGPs"
    )

    return parser.parse_args()


def list_all_dgps():
    """Print all available DGPs."""
    print("\n=== Available DGPs ===\n")

    print("Tabular:")
    for dgp in list_tabular_dgps():
        print(f"  - {dgp}")

    print("\nImage:")
    for dgp in list_image_dgps():
        print(f"  - {dgp}")

    print("\nText:")
    for dgp in list_text_dgps():
        print(f"  - {dgp}")

    print("\nGraph:")
    for dgp in list_graph_dgps():
        print(f"  - {dgp}")

    print("\nTime Series:")
    for dgp in list_timeseries_dgps():
        print(f"  - {dgp}")

    print("\nMultimodal (Image+Text):")
    for dgp in list_multimodal_dgps():
        print(f"  - {dgp}")


def main():
    args = parse_args()

    if args.list_dgps:
        list_all_dgps()
        return

    # Determine which modalities to run
    run_tabular = args.tabular or args.all
    run_image = args.image or args.all
    run_text = args.text or args.all
    run_graph = args.graph or args.all
    run_timeseries = args.timeseries or args.all
    run_multimodal = args.multimodal or args.all

    if not any([run_tabular, run_image, run_text, run_graph, run_timeseries, run_multimodal]):
        print("No modality selected. Use --all or specify modalities.")
        print("Use --help for more information.")
        return

    # Create config
    config = SimulationConfig(
        n_reps=args.n_reps,
        n_samples=args.n_samples,
        seed=args.seed,
        methods=args.methods,
        output_dir=args.output,
        verbose=args.verbose,
    )

    print("=" * 60)
    print("DeepHTE Simulation Study")
    print("=" * 60)
    print(f"Replications: {config.n_reps}")
    print(f"Samples: {config.n_samples}")
    print(f"Seed: {config.seed}")
    print(f"Methods: {', '.join(config.methods)}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    all_results = {}

    # Run tabular simulations
    if run_tabular:
        print("\n>>> TABULAR SIMULATIONS <<<")
        dgps = args.dgp if args.dgp else list_tabular_dgps()
        runner = TabularSimulationRunner(config)
        all_results["tabular"] = runner.run_all_dgps(dgps)

    # Run image simulations
    if run_image:
        print("\n>>> IMAGE SIMULATIONS <<<")
        dgps = args.dgp if args.dgp else list_image_dgps()
        runner = ImageSimulationRunner(config)
        all_results["image"] = runner.run_all_dgps(dgps)

    # Run text simulations
    if run_text:
        print("\n>>> TEXT SIMULATIONS <<<")
        dgps = args.dgp if args.dgp else list_text_dgps()
        runner = TextSimulationRunner(config)
        all_results["text"] = runner.run_all_dgps(dgps)

    # Run graph simulations
    if run_graph:
        print("\n>>> GRAPH SIMULATIONS <<<")
        dgps = args.dgp if args.dgp else list_graph_dgps()
        runner = GraphSimulationRunner(config)
        all_results["graph"] = runner.run_all_dgps(dgps)

    # Run time series simulations
    if run_timeseries:
        print("\n>>> TIME SERIES SIMULATIONS <<<")
        dgps = args.dgp if args.dgp else list_timeseries_dgps()
        runner = TimeSeriesSimulationRunner(config)
        all_results["timeseries"] = runner.run_all_dgps(dgps)

    # Run multimodal simulations
    if run_multimodal:
        print("\n>>> MULTIMODAL (IMAGE+TEXT) SIMULATIONS <<<")
        dgps = args.dgp if args.dgp else list_multimodal_dgps()
        runner = MultimodalSimulationRunner(config)
        all_results["multimodal"] = runner.run_all_dgps(dgps)

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    for modality, results in all_results.items():
        print(f"\n{modality.upper()}")
        print("-" * 40)
        for dgp_name, result in results.items():
            summary = result.summary_by_method()
            print(f"\n  {dgp_name}:")
            for method in summary.index:
                row = summary.loc[method]
                print(
                    f"    {method:15s}: "
                    f"bias={row['bias_mean']:>7.3f}, "
                    f"cov={row['coverage']:>5.1%}, "
                    f"rmse={row['ite_rmse_mean']:>6.3f}"
                )

    print(f"\nResults saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
