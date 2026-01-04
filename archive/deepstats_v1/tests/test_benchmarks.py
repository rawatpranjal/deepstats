"""Tests for benchmark dataset loaders."""

import numpy as np
import pandas as pd
import pytest

import deepstats as ds
from deepstats.datasets.benchmarks import (
    BenchmarkData,
    load_ihdp,
    load_jobs,
    load_twins,
    load_oj,
    load_acic,
    list_benchmarks,
    load_benchmark,
)


class TestBenchmarkData:
    """Test BenchmarkData dataclass."""

    def test_benchmark_data_structure(self):
        """Test BenchmarkData has expected attributes."""
        df = pd.DataFrame({
            "Y": [1.0, 2.0, 3.0],
            "T": [0, 1, 0],
            "X1": [0.1, 0.2, 0.3],
        })
        data = BenchmarkData(
            data=df,
            true_ate=1.0,
            true_ite=np.array([0.5, 1.5, 1.0]),
            description="Test dataset",
            source="Test",
            citation="Test citation",
        )

        assert data.n_obs == 3
        assert data.n_covariates == 1  # X1 only, Y and T excluded
        assert data.true_ate == 1.0
        assert len(data.true_ite) == 3

    def test_benchmark_data_none_values(self):
        """Test BenchmarkData with None for unknown ground truth."""
        df = pd.DataFrame({
            "Y": [1.0, 2.0],
            "T": [0, 1],
            "X1": [0.1, 0.2],
        })
        data = BenchmarkData(
            data=df,
            true_ate=None,
            true_ite=None,
            description="Real data",
            source="Real",
            citation="Citation",
        )

        assert data.true_ate is None
        assert data.true_ite is None


class TestListBenchmarks:
    """Test list_benchmarks function."""

    def test_list_benchmarks(self):
        """Test list_benchmarks returns expected datasets."""
        benchmarks = list_benchmarks()

        assert isinstance(benchmarks, list)
        assert "ihdp" in benchmarks
        assert "jobs" in benchmarks
        assert "twins" in benchmarks
        assert "oj" in benchmarks
        assert "acic" in benchmarks
        assert len(benchmarks) == 5


class TestLoadBenchmark:
    """Test generic load_benchmark function."""

    def test_load_benchmark_acic(self):
        """Test loading ACIC via generic loader."""
        data = load_benchmark("acic", year=2016, dgp=1)

        assert isinstance(data, BenchmarkData)
        assert "Y" in data.data.columns
        assert "T" in data.data.columns

    def test_load_benchmark_unknown(self):
        """Test loading unknown benchmark raises error."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_benchmark("unknown_dataset")


class TestLoadIHDP:
    """Test IHDP dataset loader."""

    def test_load_ihdp_structure(self):
        """Test IHDP returns correct structure."""
        data = load_ihdp(version=1)

        assert isinstance(data, BenchmarkData)
        assert isinstance(data.data, pd.DataFrame)
        assert "Y" in data.data.columns
        assert "T" in data.data.columns
        assert data.data["T"].isin([0, 1]).all()

    def test_load_ihdp_has_ground_truth(self):
        """Test IHDP has ground truth (semi-synthetic)."""
        data = load_ihdp(version=1)

        assert data.true_ate is not None
        assert data.true_ite is not None
        assert len(data.true_ite) == len(data.data)

    def test_load_ihdp_invalid_version(self):
        """Test invalid version raises error."""
        with pytest.raises(ValueError, match="version must be between"):
            load_ihdp(version=0)

        with pytest.raises(ValueError, match="version must be between"):
            load_ihdp(version=1001)


class TestLoadACIC:
    """Test ACIC dataset loader."""

    def test_load_acic_structure(self):
        """Test ACIC returns correct structure."""
        data = load_acic(year=2016, dgp=1)

        assert isinstance(data, BenchmarkData)
        assert "Y" in data.data.columns
        assert "T" in data.data.columns

    def test_load_acic_has_ground_truth(self):
        """Test ACIC has ground truth (synthetic)."""
        data = load_acic(year=2016, dgp=1)

        assert data.true_ate is not None
        assert data.true_ite is not None
        assert len(data.true_ite) == len(data.data)

    def test_load_acic_different_years(self):
        """Test loading different ACIC years."""
        for year in [2016, 2017, 2018]:
            data = load_acic(year=year, dgp=1)
            assert isinstance(data, BenchmarkData)
            assert len(data.data) > 0

    def test_load_acic_invalid_year(self):
        """Test invalid year raises error."""
        with pytest.raises(ValueError, match="year must be"):
            load_acic(year=2015)


class TestLoadJobs:
    """Test Jobs/LaLonde dataset loader."""

    def test_load_jobs_experimental(self):
        """Test loading Jobs with experimental controls."""
        data = load_jobs(control_group="experimental")

        assert isinstance(data, BenchmarkData)
        assert "Y" in data.data.columns
        assert "T" in data.data.columns

    def test_load_jobs_no_ground_truth(self):
        """Test Jobs has no ground truth (real data)."""
        data = load_jobs()

        assert data.true_ate is None
        assert data.true_ite is None

    def test_load_jobs_psid(self):
        """Test loading Jobs with PSID controls."""
        data = load_jobs(control_group="psid")

        assert isinstance(data, BenchmarkData)
        # PSID controls typically have more observations
        assert len(data.data) > 0


class TestLoadTwins:
    """Test Twins dataset loader."""

    def test_load_twins_structure(self):
        """Test Twins returns correct structure."""
        data = load_twins(seed=42)

        assert isinstance(data, BenchmarkData)
        assert "Y" in data.data.columns
        assert "T" in data.data.columns

    def test_load_twins_reproducibility(self):
        """Test Twins is reproducible with same seed."""
        data1 = load_twins(seed=42)
        data2 = load_twins(seed=42)

        # Should have same structure
        assert len(data1.data) == len(data2.data)


class TestLoadOJ:
    """Test OJ dataset loader."""

    def test_load_oj_structure(self):
        """Test OJ returns correct structure."""
        data = load_oj()

        assert isinstance(data, BenchmarkData)
        assert "Y" in data.data.columns
        assert "T" in data.data.columns

    def test_load_oj_no_ground_truth(self):
        """Test OJ has no ground truth (real data)."""
        data = load_oj()

        assert data.true_ate is None
        assert data.true_ite is None


class TestTopLevelImports:
    """Test that benchmarks are accessible from top level."""

    def test_import_from_deepstats(self):
        """Test benchmark functions accessible from ds."""
        assert hasattr(ds, "load_ihdp")
        assert hasattr(ds, "load_jobs")
        assert hasattr(ds, "load_twins")
        assert hasattr(ds, "load_oj")
        assert hasattr(ds, "load_acic")
        assert hasattr(ds, "list_benchmarks")
        assert hasattr(ds, "BenchmarkData")
        assert hasattr(ds, "clear_cache")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
