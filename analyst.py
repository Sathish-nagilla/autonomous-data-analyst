from __future__ import annotations
import json, re, os, sys, math, textwrap, datetime as dt
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import typer
from rich import print
from jinja2 import Environment, FileSystemLoader, select_autoescape

APP = typer.Typer(help="Autonomous Data Analyst (single-file MVP)")

# ---------- Step 1: Utilities ----------
def detect_task_type(y: pd.Series) -> str:
    """Heuristic: classification if few unique values or dtype looks categorical; else regression."""
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"
    if nunique <= max(10, int(0.02 * len(y))):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    return "classification"
    def metric_dict_classification(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }
    # Only try ROC-AUC for binary-like problems
    try:
        if y_proba is not None and (np.array(y_proba).ndim == 1 or np.array(y_proba).shape[1] == 2):
            proba = y_proba if np.ndim(y_proba)==1 else y_proba[:,1]
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        pass
    return out
    def metric_dict_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"rmse": float(rmse), "r2": float(r2_score(y_true, y_pred))}

def safe_figpath(name: str) -> str:
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", name)