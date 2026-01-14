"""
Report Generation Module for Evaluation Suite

Generates comprehensive reports in multiple formats:
- JSON: Machine-readable for programmatic analysis
- Markdown: Human-readable detailed report
- TXT: Quick summary table

Usage:
    from evals.report import generate_report
    generate_report(results, output_dir="evals/reports")
"""

import json
import os
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Standardized evaluation result."""
    name: str
    passed: bool
    metrics: Dict[str, Any]
    details: Optional[str] = None


def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def serialize_results(obj: Any) -> Any:
    """Convert non-serializable objects for JSON."""
    if hasattr(obj, '__dict__'):
        return {k: serialize_results(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_results(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def generate_json_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate JSON report for programmatic analysis."""
    report = {
        "meta": {
            "generated": datetime.datetime.now().isoformat(),
            "version": "1.0",
            "framework": "deep_inference evals"
        },
        "results": serialize_results(results),
        "summary": generate_summary(results)
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def generate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics from results."""
    summary = {
        "total_regimes": 0,
        "total_evals": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "by_regime": {}
    }

    for regime_name, regime_results in results.items():
        if not isinstance(regime_results, dict):
            continue

        summary["total_regimes"] += 1
        regime_summary = {"passed": 0, "failed": 0, "skipped": 0}

        for eval_name, eval_result in regime_results.items():
            if not isinstance(eval_result, dict):
                continue

            summary["total_evals"] += 1

            if eval_result.get("skipped") or eval_result.get("passed") is None:
                summary["skipped"] += 1
                regime_summary["skipped"] += 1
            elif eval_result.get("passed"):
                summary["passed"] += 1
                regime_summary["passed"] += 1
            else:
                summary["failed"] += 1
                regime_summary["failed"] += 1

        summary["by_regime"][regime_name] = regime_summary

    summary["pass_rate"] = summary["passed"] / summary["total_evals"] if summary["total_evals"] > 0 else 0
    return summary


def format_markdown_report(results: Dict[str, Any]) -> str:
    """Generate human-readable markdown report."""
    lines = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")

    # Summary section
    summary = generate_summary(results)
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total Regimes:** {summary['total_regimes']}")
    lines.append(f"- **Total Evaluations:** {summary['total_evals']}")
    lines.append(f"- **Passed:** {summary['passed']}")
    lines.append(f"- **Failed:** {summary['failed']}")
    lines.append(f"- **Skipped:** {summary['skipped']}")
    lines.append(f"- **Pass Rate:** {summary['pass_rate']*100:.1f}%")
    lines.append("")

    # Scorecard table
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Regime | Eval | Status |")
    lines.append("|--------|------|--------|")

    for regime_name, regime_results in results.items():
        if not isinstance(regime_results, dict):
            continue
        regime_display = regime_name.replace("_", " ").upper()
        first_row = True

        for eval_name, eval_result in regime_results.items():
            if not isinstance(eval_result, dict):
                continue

            if eval_result.get("skipped") or eval_result.get("passed") is None:
                status = "⏭️ SKIPPED"
            elif eval_result.get("passed"):
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            regime_col = regime_display if first_row else ""
            lines.append(f"| {regime_col} | {eval_name} | {status} |")
            first_row = False

    lines.append("")

    # Detailed results by regime
    for regime_name, regime_results in results.items():
        if not isinstance(regime_results, dict):
            continue

        lines.append(f"## {regime_name.replace('_', ' ').upper()}")
        lines.append("")

        for eval_name, eval_result in regime_results.items():
            if not isinstance(eval_result, dict):
                continue

            lines.append(f"### {eval_name}")
            lines.append("")

            # Status
            if eval_result.get("skipped") or eval_result.get("passed") is None:
                lines.append("**Status:** ⏭️ SKIPPED")
            elif eval_result.get("passed"):
                lines.append("**Status:** ✅ PASS")
            else:
                lines.append("**Status:** ❌ FAIL")
            lines.append("")

            # Metrics
            if "metrics" in eval_result:
                lines.append("**Metrics:**")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for metric_name, metric_value in eval_result["metrics"].items():
                    if isinstance(metric_value, float):
                        lines.append(f"| {metric_name} | {metric_value:.6f} |")
                    else:
                        lines.append(f"| {metric_name} | {metric_value} |")
                lines.append("")

            # Coverage-specific details
            if "coverage" in str(eval_name).lower() and "metrics" in eval_result:
                metrics = eval_result["metrics"]
                if "coverage" in metrics:
                    cov = metrics["coverage"]
                    lines.append(f"**Coverage:** {cov*100:.1f}%")
                if "se_ratio" in metrics:
                    lines.append(f"**SE Ratio:** {metrics['se_ratio']:.4f}")
                if "bias" in metrics:
                    lines.append(f"**Bias:** {metrics['bias']:.6f}")
                lines.append("")

    return "\n".join(lines)


def format_summary_table(results: Dict[str, Any]) -> str:
    """Generate quick summary table (TXT format)."""
    lines = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 70)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # Overall stats
    summary = generate_summary(results)
    lines.append(f"Total: {summary['total_evals']} evals across {summary['total_regimes']} regimes")
    lines.append(f"Pass: {summary['passed']}  Fail: {summary['failed']}  Skip: {summary['skipped']}")
    lines.append(f"Pass Rate: {summary['pass_rate']*100:.1f}%")
    lines.append("")

    # Table
    lines.append("-" * 70)
    lines.append(f"{'Regime':<15} {'Eval':<25} {'Status':<10} {'Key Metric':<15}")
    lines.append("-" * 70)

    for regime_name, regime_results in results.items():
        if not isinstance(regime_results, dict):
            continue
        regime_display = regime_name.replace("_", " ").upper()
        first_row = True

        for eval_name, eval_result in regime_results.items():
            if not isinstance(eval_result, dict):
                continue

            if eval_result.get("skipped") or eval_result.get("passed") is None:
                status = "SKIP"
            elif eval_result.get("passed"):
                status = "PASS"
            else:
                status = "FAIL"

            # Extract key metric if available
            key_metric = ""
            if "metrics" in eval_result:
                metrics = eval_result["metrics"]
                if "coverage" in metrics:
                    key_metric = f"cov={metrics['coverage']*100:.0f}%"
                elif "error" in metrics:
                    key_metric = f"err={metrics['error']:.4f}"
                elif "rmse" in metrics:
                    key_metric = f"rmse={metrics['rmse']:.4f}"

            regime_col = regime_display if first_row else ""
            lines.append(f"{regime_col:<15} {eval_name:<25} {status:<10} {key_metric:<15}")
            first_row = False

    lines.append("-" * 70)

    # Regime summary
    lines.append("")
    lines.append("BY REGIME:")
    for regime_name, regime_stats in summary["by_regime"].items():
        p = regime_stats["passed"]
        f = regime_stats["failed"]
        s = regime_stats["skipped"]
        lines.append(f"  {regime_name}: {p} pass, {f} fail, {s} skip")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_report(
    results: Dict[str, Any],
    output_dir: str = "evals/reports"
) -> Dict[str, str]:
    """
    Generate comprehensive validation reports.

    Args:
        results: Dictionary of evaluation results by regime
        output_dir: Directory for output files

    Returns:
        Dictionary with paths to generated files
    """
    ensure_output_dir(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # File paths
    json_path = os.path.join(output_dir, f"evals_report_{timestamp}.json")
    md_path = os.path.join(output_dir, f"evals_report_{timestamp}.md")
    txt_path = os.path.join(output_dir, f"evals_summary_{timestamp}.txt")

    # Also create latest versions (no timestamp)
    json_latest = os.path.join(output_dir, "evals_report_latest.json")
    md_latest = os.path.join(output_dir, "evals_report_latest.md")
    txt_latest = os.path.join(output_dir, "evals_summary_latest.txt")

    # Generate reports
    generate_json_report(results, json_path)
    generate_json_report(results, json_latest)

    md_content = format_markdown_report(results)
    with open(md_path, 'w') as f:
        f.write(md_content)
    with open(md_latest, 'w') as f:
        f.write(md_content)

    txt_content = format_summary_table(results)
    with open(txt_path, 'w') as f:
        f.write(txt_content)
    with open(txt_latest, 'w') as f:
        f.write(txt_content)

    print("")
    print("=" * 60)
    print("REPORTS GENERATED")
    print("=" * 60)
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Summary:  {txt_path}")
    print("")
    print("Latest versions (no timestamp):")
    print(f"  JSON:     {json_latest}")
    print(f"  Markdown: {md_latest}")
    print(f"  Summary:  {txt_latest}")
    print("=" * 60)

    return {
        "json": json_path,
        "markdown": md_path,
        "summary": txt_path,
        "json_latest": json_latest,
        "markdown_latest": md_latest,
        "summary_latest": txt_latest,
    }


if __name__ == "__main__":
    # Test with dummy results
    test_results = {
        "regime_a": {
            "eval_01": {"passed": True, "metrics": {"error": 0.001}},
            "eval_02": {"passed": True, "metrics": {}},
            "eval_03": {"passed": False, "metrics": {"coverage": 0.85}},
        },
        "regime_b": {
            "eval_01": {"passed": True, "metrics": {"rmse": 0.05}},
            "eval_02": {"skipped": True},
        },
        "regime_c": {
            "eval_01": {"passed": True, "metrics": {}},
            "eval_06": {"passed": True, "metrics": {"coverage": 0.95, "se_ratio": 1.02}},
        },
    }

    paths = generate_report(test_results, output_dir="evals/reports")
    print("\nTest complete. Check the generated files.")
