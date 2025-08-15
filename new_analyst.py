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
    # plotting count bar plot for categorical columns
    def plot_bar_pie_with_nan(df:pd.DataFrame, categorical_columns:List, charts: List[str], plots_per_row:int =2, nan_label:str="nan", wrap_width=15):
        
        total_plots = len(categorical_columns) * 2
        rows = math.ceil(total_plots / plots_per_row)
        
        fig, axes = plt.subplots(rows, plots_per_row, figsize=(plots_per_row * 5.5, rows * 4))
        axes = axes.flatten()
        ax_idx = 0

        for col in categorical_columns:
            if col not in df.columns:
                print(f"Column '{col}' not found, skipping.")
                continue

            series = df[col].copy()
            
            # Add 'nan' label for missing values
            if pd.api.types.is_categorical_dtype(series):
                if nan_label not in series.cat.categories:
                    series = series.cat.add_categories([nan_label])
            series = series.fillna(nan_label)

            counts = series.value_counts(dropna=False)
            labels = [textwrap.fill(str(lab), wrap_width) for lab in counts.index]

            # Bar plot
            sns.barplot(x=labels, y=counts.values, ax=axes[ax_idx], palette="bright")
            axes[ax_idx].set_title(f'Bar - {col}')
            axes[ax_idx].tick_params(axis='x', rotation=30)
            axes[ax_idx].set_ylabel("Count")
            #y limits
            limmax=series.value_counts().max()
            y_max = math.ceil(limmax+(limmax*0.1))  # Next whole number
            axes[ax_idx].set(ylim=(0, y_max))
            # Add count labels on bars
            for container in axes[ax_idx].containers:
                axes[ax_idx].bar_label(container, fontsize=9, padding=2)
            ax_idx += 1

            # Pie chart
            axes[ax_idx].pie(counts.values, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("bright"))
            axes[ax_idx].set_title(f'Pie - {col}')
            ax_idx += 1

        # Remove any unused axes
        for i in range(ax_idx, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        figpath = safe_figpath(f"countplot_categorical.png")
        plt.savefig(figpath); plt.close()
        charts.append(figpath)

    # plotting count bar plot for binary columns
    def plot_bar_grid(df:pd.DataFrame, all_cols:List, charts: List[str], cols_per_row:int=3,nan_label:str="nan"):
        
        figpath = safe_figpath(f"countplot_binary.png")

        num_plots = len(all_cols)
        rows = math.ceil(num_plots / cols_per_row)
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(all_cols):
            # Replace NaN with a visible label
            temp_series = df[col].copy()
            temp_series = temp_series.fillna(nan_label)
            count=temp_series.count()

            # Plot
            sns.countplot(x=temp_series, ax=axes[i], palette="dark")
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Count")
            
            limmax=temp_series.value_counts().max()
            y_max = math.ceil(limmax+(limmax*0.1))  # Next whole number
            axes[i].set(ylim=(0, y_max))
        
            axes[i].tick_params(axis='x', rotation=30)
            # Add count labels on bars
            for container in axes[i].containers:
                axes[i].bar_label(container, fontsize=9, padding=2)
        # Hide unused axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(figpath); plt.close()
        charts.append(figpath)
    
    # plotting histogram for numerical columns
    def summary_plot_numerical_variables(df:pd.DataFrame, cols:List, charts: List[str]):
        
        l=len(cols)
        if l==0:
            l=1
        #fig, ax =plt.subplots(1,2*l, figsize=(12, 4),tight_layout=True)
        fig, ax =plt.subplots(l,2, figsize=(16, l * 5),tight_layout=True)

        
        for i,z in enumerate(cols):       
            #hist plot
            ax[i][0].hist(df[z].dropna(), bins=30, color='skyblue', edgecolor='black')
            ax[i][0].set_xlabel(z)
            ax[i][0].set_ylabel("count")
            ax[i][0].set_title(f'Histogram - {z}')
            
            #box plot
            ax[i][1].boxplot(df[z].dropna(),patch_artist=True,boxprops=dict(color='blue', facecolor='lightblue'))
            ax[i][1].set_title(f'Barplot - {z}')
            ax[i][1].yaxis.grid()
            
        plt.tight_layout()
        figpath = safe_figpath(f"histplots_numerical.png")
        plt.savefig(figpath); plt.close()
        charts.append(figpath)

    
