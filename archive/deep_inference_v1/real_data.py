"""Data loaders for real datasets.

Loads and preprocesses real-world datasets for inference comparison.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class RealDataResult:
    """Container for loaded real data."""
    X: np.ndarray           # (n, d) covariates
    T: np.ndarray           # (n,) treatment variable
    Y: np.ndarray           # (n,) outcome variable
    offset: Optional[np.ndarray] = None  # (n,) offset for Poisson
    feature_names: List[str] = None
    metadata: dict = None


def load_bank_marketing(path: str = "data/external/bank_marketing.csv") -> RealDataResult:
    """Load and preprocess Bank Marketing dataset for logit analysis.

    Treatment: contact method (cellular=1 vs telephone=0)
    Outcome: subscription (yes=1 vs no=0)

    Args:
        path: Path to CSV file

    Returns:
        RealDataResult with preprocessed X, T, Y
    """
    df = pd.read_csv(path, sep=';')

    # Treatment: contact method
    T = (df['contact'] == 'cellular').astype(float).values

    # Outcome: subscription
    Y = (df['y'] == 'yes').astype(float).values

    # Covariates
    # Continuous features
    continuous_cols = ['age', 'campaign', 'pdays', 'previous',
                       'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                       'euribor3m', 'nr.employed']

    # Categorical features (one-hot encode)
    categorical_cols = ['job', 'marital', 'education', 'default',
                        'housing', 'loan', 'month', 'day_of_week', 'poutcome']

    # Duration is excluded (it's only known after the call - data leakage)

    # Build feature matrix
    X_continuous = df[continuous_cols].values.astype(float)

    # One-hot encode categoricals
    X_categorical_list = []
    feature_names = list(continuous_cols)

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        X_categorical_list.append(dummies.values)
        feature_names.extend(dummies.columns.tolist())

    X_categorical = np.hstack(X_categorical_list) if X_categorical_list else np.empty((len(df), 0))

    # Combine and standardize
    X_raw = np.hstack([X_continuous, X_categorical])

    # Standardize (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    metadata = {
        'name': 'Bank Marketing',
        'n_obs': len(df),
        'treatment': 'contact (cellular vs telephone)',
        'outcome': 'y (subscribed yes/no)',
        'treatment_rate': float(T.mean()),
        'outcome_rate': float(Y.mean()),
        'n_features': X.shape[1],
    }

    return RealDataResult(
        X=X, T=T, Y=Y,
        feature_names=feature_names,
        metadata=metadata,
    )


def load_fremtpl2freq(
    path: str = "data/external/fremtpl2freq.csv",
    subsample: Optional[int] = None,
    seed: int = 42,
) -> RealDataResult:
    """Load and preprocess French Motor Insurance dataset for Poisson analysis.

    Treatment: fuel type (Diesel=1 vs Regular=0)
    Outcome: claim counts
    Offset: log(Exposure)

    Args:
        path: Path to CSV file
        subsample: If set, randomly sample this many observations
        seed: Random seed for subsampling

    Returns:
        RealDataResult with preprocessed X, T, Y, offset
    """
    df = pd.read_csv(path)

    # Subsample if requested (dataset is large: 678K)
    if subsample is not None and subsample < len(df):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df), subsample, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    # Treatment: fuel type (clean the quotes)
    df['VehGas'] = df['VehGas'].str.strip("'")
    T = (df['VehGas'] == 'Diesel').astype(float).values

    # Outcome: claim counts
    Y = df['ClaimNb'].values.astype(float)

    # Offset: log(Exposure) for Poisson
    offset = np.log(df['Exposure'].values + 1e-8)

    # Covariates
    # Continuous features
    continuous_cols = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']

    # Categorical features
    categorical_cols = ['Area', 'VehBrand', 'Region']

    # Build feature matrix
    X_continuous = df[continuous_cols].values.astype(float)

    # One-hot encode categoricals (clean quotes first)
    X_categorical_list = []
    feature_names = list(continuous_cols)

    for col in categorical_cols:
        df[col] = df[col].str.strip("'")
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        X_categorical_list.append(dummies.values)
        feature_names.extend(dummies.columns.tolist())

    X_categorical = np.hstack(X_categorical_list) if X_categorical_list else np.empty((len(df), 0))

    # Combine and standardize
    X_raw = np.hstack([X_continuous, X_categorical])

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    metadata = {
        'name': 'French Motor Insurance (freMTPL2freq)',
        'n_obs': len(df),
        'treatment': 'VehGas (Diesel vs Regular)',
        'outcome': 'ClaimNb (claim counts)',
        'treatment_rate': float(T.mean()),
        'outcome_mean': float(Y.mean()),
        'outcome_zeros': float((Y == 0).mean()),
        'n_features': X.shape[1],
        'exposure_mean': float(df['Exposure'].mean()),
    }

    return RealDataResult(
        X=X, T=T, Y=Y,
        offset=offset,
        feature_names=feature_names,
        metadata=metadata,
    )


def load_credit_default(path: str = "data/external/credit_default.csv") -> RealDataResult:
    """Load and preprocess Credit Card Default dataset for logit analysis.

    Treatment: education level (graduate school=1 vs others=0)
    Outcome: default payment next month (1=yes, 0=no)

    Args:
        path: Path to CSV file

    Returns:
        RealDataResult with preprocessed X, T, Y
    """
    df = pd.read_csv(path)

    # Find outcome column (varies by dataset version)
    outcome_col = [c for c in df.columns if 'default' in c.lower()][-1]

    # Treatment: Education (1=graduate school, 2=university, 3=high school, etc.)
    # Use graduate school (1) vs others as treatment
    T = (df['EDUCATION'] == 1).astype(float).values

    # Outcome: default
    Y = df[outcome_col].values.astype(float)

    # Covariates (exclude ID, EDUCATION which is treatment)
    continuous_cols = ['LIMIT_BAL', 'AGE',
                       'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    categorical_cols = ['SEX', 'MARRIAGE']

    # Build feature matrix
    X_continuous = df[continuous_cols].values.astype(float)

    # One-hot encode categoricals
    X_categorical_list = []
    feature_names = list(continuous_cols)

    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        X_categorical_list.append(dummies.values)
        feature_names.extend(dummies.columns.tolist())

    X_categorical = np.hstack(X_categorical_list) if X_categorical_list else np.empty((len(df), 0))

    # Combine and standardize
    X_raw = np.hstack([X_continuous, X_categorical])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    metadata = {
        'name': 'Credit Card Default',
        'n_obs': len(df),
        'treatment': 'EDUCATION (graduate school vs others)',
        'outcome': 'default payment next month',
        'treatment_rate': float(T.mean()),
        'outcome_rate': float(Y.mean()),
        'n_features': X.shape[1],
    }

    return RealDataResult(
        X=X, T=T, Y=Y,
        feature_names=feature_names,
        metadata=metadata,
    )


# Registry
DATASETS = {
    'bank_marketing': load_bank_marketing,
    'fremtpl2freq': load_fremtpl2freq,
    'credit_default': load_credit_default,
}


def load_dataset(name: str, **kwargs) -> RealDataResult:
    """Load dataset by name.

    Args:
        name: Dataset name ('bank_marketing', 'fremtpl2freq', 'credit_default')
        **kwargs: Additional arguments passed to loader

    Returns:
        RealDataResult
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](**kwargs)
