"""Model and preprocessing factory shared by all phases.

Hyperparameters for RF and XGBoost are kept IDENTICAL to v1, so any change
in results in Phase 0 comes from the data/feature fixes, not from tuning.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from deliveriq import config


def make_preprocessor(numeric: list[str], categorical: list[str], scale: bool = False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], verbose_feature_names_out=True)


def rf_params() -> dict:
    return dict(n_estimators=250, max_depth=18, min_samples_leaf=3,
                random_state=config.RANDOM_STATE, n_jobs=-1)


def xgb_params(**overrides) -> dict:
    p = dict(n_estimators=500, learning_rate=0.05, max_depth=7, min_child_weight=3,
             subsample=0.8, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
             objective="reg:squarederror", random_state=config.RANDOM_STATE,
             n_jobs=-1, tree_method="hist")
    p.update(overrides)
    return p


def make_model(name: str, numeric: list[str], categorical: list[str], **xgb_overrides) -> Pipeline:
    if name == "Linear Regression":
        est, scale = LinearRegression(), True
    elif name == "Random Forest":
        est, scale = RandomForestRegressor(**rf_params()), False
    elif name == "XGBoost":
        est, scale = XGBRegressor(**xgb_params(**xgb_overrides)), False
    else:
        raise ValueError(name)
    return Pipeline([("pre", make_preprocessor(numeric, categorical, scale)), ("model", est)])


BASE_MODELS = ["Linear Regression", "Random Forest", "XGBoost"]
