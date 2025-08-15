#!/usr/bin/env python
# analyst.py
# --------------------------------------------------------
# This is the main script for the Autonomous Data Analyst AI.
# It loads your dataset, automatically detects the task type,
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
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        # quick correlations (numeric only, top absolute)
        corr_top = None
        if self.target in num_cols:
            corr = df[num_cols].corr(numeric_only=True)[self.target].dropna().sort_values(key=np.abs, ascending=False)
            corr_top = corr.head(10).rename("corr_to_target").to_frame()
        # simple charts
        charts = []
        self._plot_hist(self.target, df[self.target], charts)
        for col in num_cols[:min(5, len(num_cols))]:
            if col == self.target: continue
            self._plot_scatter(col, self.target, charts)
        return {
            "missing_summary": missing_summary,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "corr_top": corr_top,
            "charts": charts
        }

    # Plot histogram
    def _plot_hist(self, name, s: pd.Series, charts: List[str]):
        figpath = safe_figpath(f"hist_{name}.png")
        plt.figure()
        plt.hist(s.dropna(), bins=30)
        plt.title(f"Histogram: {name}")
        plt.xlabel(name); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(figpath); plt.close()
        charts.append(figpath)

    # Plot scatter
    def _plot_scatter(self, x, y, charts: List[str]):
        figpath = safe_figpath(f"scatter_{x}_vs_{y}.png")
        plt.figure()
        plt.scatter(self.df[x], self.df[y])
        plt.title(f"{x} vs {y}")
        plt.xlabel(x); plt.ylabel(y)
        plt.tight_layout(); plt.savefig(figpath); plt.close()
        charts.append(figpath)

    # Preprocess and split data
    def preprocess_and_split(self, num_cols: List[str], cat_cols: List[str]):
        df = self.df.copy()
        y = df[self.target]
        X = df.drop(columns=[self.target])
        self.task_type = detect_task_type(y)
        numeric_t = Pipeline(steps=[("scale", StandardScaler(with_mean=False))])
        categorical_t = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_t, [c for c in num_cols if c in X.columns]),
                ("cat", categorical_t, [c for c in cat_cols if c in X.columns])
            ],
            remainder="drop"
        )
        if self.task_type == "classification":
            model = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=self.random_state)
        self.pipeline = Pipeline(steps=[("pre", pre), ("model", model)])
        strat = None
        if self.task_type == "classification" and y.nunique() > 1:
            strat = y if y.nunique() <= 50 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=strat
        )

    # Train and evaluate
    def train_evaluate(self):
        self.pipeline.fit(self.X_train, self.y_train)
        y_pred = self.pipeline.predict(self.X_test)
        if self.task_type == "classification":
            try:
                y_proba = self.pipeline.predict_proba(self.X_test)
            except Exception:
                y_proba = None
            self.metrics = metric_dict_classification(self.y_test, y_pred, y_proba)
            self.model_name = "RandomForestClassifier"
        else:
            self.metrics = metric_dict_regression(self.y_test, y_pred)
            self.model_name = "RandomForestRegressor"
        print(f"[bold cyan]{self.task_type.title()}[/] metrics:", self.metrics)

    # Explain feature importance
    def explain(self):
        try:
            r = permutation_importance(self.pipeline, self.X_test, self.y_test, n_repeats=5, random_state=self.random_state)
            feat_names = self._get_feature_names()
            pi = pd.DataFrame({"feature": feat_names, "importance_mean": r.importances_mean})
            pi = pi.sort_values("importance_mean", ascending=False)
            self.perm_importance_ = pi
            figpath = safe_figpath("feature_importance.png")
            plt.figure()
            top = pi.head(20)
            plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
            plt.title("Permutation Importance (top 20)")
            plt.tight_layout(); plt.savefig(figpath); plt.close()
        except Exception as e:
            print("[yellow]Explainability skipped:[/]", e)

    # Get feature names after preprocessing
    def _get_feature_names(self) -> List[str]:
        pre: ColumnTransformer = self.pipeline.named_steps["pre"]
        feat_names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                feat_names.extend(cols)
            elif name == "cat":
                try:
                    ohe: OneHotEncoder = trans.named_steps["onehot"]
                    cat_cols = cols
                    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
                    feat_names.extend(ohe_names)
                except Exception:
                    feat_names.extend(cols)
        return feat_names

    # Generate HTML report
    def report(self, profile_dict: Dict[str, Any]) -> str:
        env = Environment(
            loader=FileSystemLoader("templates"),
            autoescape=select_autoescape()
        )
        template = env.get_template("report.html.j2")
        missing_summary = profile_dict["missing_summary"]
        corr_top = profile_dict["corr_top"]
        charts = profile_dict["charts"]
        html = template.render(
            title="Autonomous Data Analyst Report",
            timestamp=dt.datetime.now().isoformat(timespec="seconds"),
            csv_path=self.csv_path,
            shape=str(self.df.shape),
            target=self.target,
            task_type=self.task_type,
            model_name=self.model_name,
            metrics=self.metrics,
            missing_table=missing_summary.reset_index().rename(columns={"index":"column"}).to_dict(orient="records"),
            corr_table=None if corr_top is None else corr_top.reset_index().rename(columns={"index":"column"}).to_dict(orient="records"),
            charts=charts + ([safe_figpath("feature_importance.png")] if os.path.exists(safe_figpath("feature_importance.png")) else []),
        )
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "report.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[bold green]Report written:[/] {out_path}")
        return out_path

    # Q&A capability
    def ask(self, question: str) -> str:
        q = question.lower()
        if "best model" in q or "which model" in q:
            return f"The best model used was {self.model_name} with metrics: {json.dumps(self.metrics)}."
        if "accuracy" in q or "f1" in q or "roc" in q or "rmse" in q or "r2" in q:
            return f"Key metrics: {json.dumps(self.metrics)}."
        if "feature" in q and ("important" in q or "importance" in q or "matters" in q):
            if self.perm_importance_ is None:
                return "Feature importance not available (try running analyze first)."
            top = self.perm_importance_.head(10)
            return "Top features:\n" + "\n".join(f"- {r.feature}: {r.importance_mean:.4f}" for r in top.itertuples())
        if "missing" in q or "null" in q:
            miss = self.df.isna().sum().sort_values(ascending=False)
            return "Missing values:\n" + "\n".join(f"- {c}: {int(v)}" for c, v in miss[miss>0].items())
        if "correlat" in q:
            num_cols = self.df.select_dtypes(include=np.number).columns
            if self.target in num_cols:
                corr = self.df[num_cols].corr(numeric_only=True)[self.target].dropna().sort_values(key=np.abs, ascending=False).head(10)
                return "Top correlations to target:\n" + "\n".join(f"- {c}: {v:.3f}" for c, v in corr.items())
        return "I can answer about metrics, model choice, feature importance, missing values, or target correlations."

# --------------------------------------------------------
# CLI Commands
# --------------------------------------------------------
@APP.command()
def analyze(csv: str = typer.Argument(..., help="Path to CSV"),
            target: str = typer.Argument(..., help="Target column"),
            test_size: float = 0.2):
    """
    Run full autonomous analysis.
    """
    agent = AnalystAgent(csv, target, test_size=test_size)
    agent.load()
    prof = agent.profile()
    agent.preprocess_and_split(prof["num_cols"], prof["cat_cols"])
    agent.train_evaluate()
    agent.explain()
    path = agent.report(prof)
    print(f"[bold]Done.[/] Open: {path}")

@APP.command()
def ask(csv: str = typer.Argument(..., help="Path to CSV used in analyze"),
        target: str = typer.Argument(..., help="Target used in analyze"),
        question: str = typer.Argument(..., help="Your question about the run")):
    """
    Ask questions about dataset or model.
    """
    agent = AnalystAgent(csv, target)
    agent.load()
    prof = agent.profile()
    agent.preprocess_and_split(prof["num_cols"], prof["cat_cols"])
    agent.train_evaluate()
    agent.explain()
    answer = agent.ask(question)
    print(answer)

if __name__ == "__main__":
    APP()

#python analyst.py analyze Multiclass_Diabetes_Dataset.csv Class

