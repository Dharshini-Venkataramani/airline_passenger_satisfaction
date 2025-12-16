"""
Shared ML pipeline components for the airline classification project.

This module holds all custom transformers and helper functions that are used
both in training and in inference (FastAPI app), so that joblib pickles
refer to a stable module path: `airline_pipeline`.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =============================================================================
# Build preprocessing
# =============================================================================

# Numerical pipeline
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

# Categorical pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)


def build_preprocessing():
    """
    Return the ColumnTransformer preprocessing used in airline models.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
    )
    return preprocessing


# =============================================================================
# Estimator factory (used by baseline, PCA, Optuna, inference)
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Given a model name, return an unconfigured CLASSIFIER instance.
    """
    if name == "ridge":
        return RidgeClassifier(alpha=1.0)

    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        )

    elif name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
        )

    elif name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {name}")
