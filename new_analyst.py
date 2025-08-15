#!/usr/bin/env python
# analyst.py
# --------------------------------------------------------
# This is the main script for the Autonomous Data Analyst AI.
# It loads dataset, automatically detects the task type,
# preprocesses the data, trains a model, evaluates performance,
# explains the results, generates a report, and answers questions.
# --------------------------------------------------------

from __future__ import annotations  # Allows forward references in type hints (Python 3.7+)

import json, re, os, sys, math, textwrap, datetime as dt  # Built-in libraries
from dataclasses import dataclass                       # To store analysis results neatly
from typing import Tuple, Dict, Any, Optional, List      # Type hints for cleaner code

# Scientific stack for data and ML
import numpy as np
import pandas as pd

# For model training/testing and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

# Plotting (Matplotlib in headless mode)
import matplotlib
matplotlib.use("Agg")  # Avoids GUI backend issues in servers
import matplotlib.pyplot as plt
import seaborn as sns

# CLI interface & pretty printing
import typer
from rich import print

# HTML template rendering
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Typer app (CLI entry point)
APP = typer.Typer(help="Autonomous Data Analyst (single-file MVP)")

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def detect_task_type(y: pd.Series) -> str:
    """
    Detect whether the target is for classification or regression.
    Logic:
      - Boolean or categorical dtype → classification
      - Few unique values (<= 10 or <= 2% of data) → classification
      - Otherwise numeric → regression
    """
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"
    if nunique <= max(10, int(0.02 * len(y))):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    return "classification"

def metric_dict_classification(y_true, y_pred, y_proba=None):
    """
    Compute classification metrics:
    - Accuracy
    - F1 Macro
    - ROC-AUC (if binary and probabilities available)
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }
    try:
        if y_proba is not None and (np.array(y_proba).ndim == 1 or np.array(y_proba).shape[1] == 2):
            proba = y_proba if np.ndim(y_proba) == 1 else y_proba[:, 1]
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        pass
    return out

def metric_dict_regression(y_true, y_pred):
    """Compute regression metrics: RMSE and R²."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"rmse": float(rmse), "r2": float(r2_score(y_true, y_pred))}

def safe_figpath(name: str) -> str:
    """Ensure outputs folder exists and return safe file path for saving figures."""
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", name)
# --------------------------------------------------------
# Data structure to store artifacts from a run
# --------------------------------------------------------
@dataclass
class RunArtifacts:
    task_type: str
    target: str
    model_name: str
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame]
    n_rows: int
    n_cols: int
    missing_summary: pd.DataFrame
    corr_top: Optional[pd.DataFrame]
    charts: List[str]
    train_cols: List[str]

# --------------------------------------------------------
# Main AI Analyst Class
# --------------------------------------------------------
class AnalystAgent:
    """
    Orchestrates:
    - load → profile → preprocess → train → evaluate → explain → report → Q&A
    """
    def __init__(self, csv_path: str, target: str, test_size: float=0.2, random_state: int=42):
        self.csv_path = csv_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.pipeline: Optional[Pipeline] = None
        self.task_type: Optional[str] = None
        self.model_name: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self.perm_importance_: Optional[pd.DataFrame] = None
        self.train_cols: List[str] = []
        self.artifacts: Optional[RunArtifacts] = None

    # Load CSV
    def load(self):
        self.df = pd.read_csv(self.csv_path)
        assert self.target in self.df.columns, f"Target '{self.target}' not in columns."
        print(f"[bold green]Loaded[/] {self.csv_path} with shape {self.df.shape}")
    # Profile dataset
    def profile(self) -> Dict[str, Any]:
        df = self.df
        missing = df.isna().sum().sort_values(ascending=False)
        missing_summary = missing[missing > 0].rename("missing").to_frame()
        # quick type inference
        category_cols=[]
        numerical_cols=[]
        binary_cols=[]
        for col in df.columns:
            if df[col].nunique()>2 and df[col].nunique()<15:
                category_cols.append(col)
            if df[col].nunique()==2:
                binary_cols.append(col)
            if df[col].nunique()>=15:
                numerical_cols.append(col)
        # quick correlations (numeric only, top absolute)
        corr_top = None
        number_cols = df.select_dtypes(include=np.number).columns.tolist()
        if self.target in num_cols:
            corr = df[number_cols].corr(numeric_only=True)[self.target].dropna().sort_values(key=np.abs, ascending=False)
            corr_top = corr.head(10).rename("corr_to_target").to_frame()
        # simple charts
        charts = []
        ## Histogram for each numerical column


        # for scatter relation plot
        sns.pairplot(df)
        return {
            "missing_summary": missing_summary,
            "numerical_cols": numerical_cols,
            "category_cols": category_cols,
            "binary_cols": binary_cols,
            "corr_top": corr_top,
            "charts": charts
        }
        
