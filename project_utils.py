"""
Reusable utilities for the BRFSS 2015 diabetes machine learning project.

The notebooks in this project are intentionally executable as independent files.
This module centralizes path management, plotting style, preprocessing, modeling,
evaluation, and export helpers so the workflow remains reproducible and concise.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    import shap
except Exception:  # pragma: no cover - notebook will report availability.
    shap = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TARGET = "Diabetes_012"
CLASS_NAMES = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
N_CLASSES = 3

DATA_PATH = ROOT / "diabetes_012_health_indicators_BRFSS2015.csv"
CODE_DIR = ROOT / "code"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = ROOT / "reports"
INTERIM_DIR = RESULTS_DIR / "interim"
for directory in [CODE_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR, INTERIM_DIR, ROOT / ".cache" / "matplotlib"]:
    directory.mkdir(parents=True, exist_ok=True)

MORANDI = [
    "#8FA1A6",
    "#C9A27E",
    "#A7B88F",
    "#B58E8E",
    "#8E8AAE",
    "#D0B49F",
    "#7895A3",
    "#B0A990",
    "#9BAEBC",
    "#C4A69D",
]


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds and plotting defaults."""
    random.seed(seed)
    np.random.seed(seed)
    sns.set_theme(style="whitegrid", context="notebook", palette=MORANDI)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_raw_data() -> pd.DataFrame:
    """Load the BRFSS 2015 diabetes indicators dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame, remove_duplicates: bool = True) -> pd.DataFrame:
    """Return a numeric, de-duplicated modeling table."""
    cleaned = df.copy()
    cleaned.columns = [c.strip() for c in cleaned.columns]
    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Tuple[Path, Path]:
    """Save a table as CSV and XLSX."""
    csv_path = TABLES_DIR / f"{name}.csv"
    xlsx_path = TABLES_DIR / f"{name}.xlsx"
    df.to_csv(csv_path, index=index)
    df.to_excel(xlsx_path, index=index)
    return csv_path, xlsx_path


def save_figure(fig: plt.Figure, name: str) -> Tuple[Path, Path]:
    """Save a figure as high-resolution PNG and PDF."""
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def target_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    features = [c for c in df.columns if c != TARGET]
    return df[features], df[TARGET].astype(int), features


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    X, y, features = target_features(df)
    return (*train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE), features)


def build_preprocessor(features: Iterable[str]) -> ColumnTransformer:
    """Scale all input columns; BRFSS fields are numeric/ordinal/binary."""
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator: Any, features: Iterable[str], use_smote: bool = True) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("preprocess", build_preprocessor(features))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))
    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def base_estimators() -> Dict[str, Any]:
    """Return the six required classifiers with reproducible settings."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, solver="lbfgs", multi_class="auto", n_jobs=-1, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=12, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=140,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            n_estimators=140,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(C=0.7, class_weight="balanced", dual="auto", random_state=RANDOM_STATE, max_iter=3000),
            cv=3,
        ),
    }


def sample_for_training(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int = 60000,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a stratified training subset for expensive algorithms."""
    if len(X) <= max_rows:
        return X.copy(), y.copy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=seed)
    idx, _ = next(splitter.split(X, y))
    return X.iloc[idx].copy(), y.iloc[idx].copy()


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC_AUC_ovr_macro": roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr"),
        "PR_AUC_macro": average_precision_score(y_bin, y_score, average="macro"),
    }
    report = pd.DataFrame(classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()), output_dict=True, zero_division=0)).T
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    return metrics, report, cm, y_score


def train_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    model_names: Iterable[str] | None = None,
    max_rows: int = 60000,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    models = base_estimators()
    if model_names is not None:
        models = {k: models[k] for k in model_names}
    X_fit, y_fit = sample_for_training(X_train, y_train, max_rows=max_rows)
    fitted: Dict[str, Any] = {}
    reports: Dict[str, pd.DataFrame] = {}
    cms: Dict[str, np.ndarray] = {}
    scores: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float]] = []
    for name, estimator in models.items():
        print(f"Training {name} on {len(X_fit):,} stratified training rows...")
        pipe = make_pipeline(clone(estimator), features, use_smote=True)
        pipe.fit(X_fit, y_fit)
        metrics, report, cm, y_score = evaluate_model(pipe, X_test, y_test, name)
        rows.append(metrics)
        reports[name] = report
        cms[name] = cm
        scores[name] = y_score
        fitted[name] = pipe
        joblib.dump(pipe, MODELS_DIR / f"{safe_name(name)}.joblib")
        save_table(report.reset_index().rename(columns={"index": "Class"}), f"classification_report_{safe_name(name)}")
    results = pd.DataFrame(rows).sort_values(["F1_macro", "ROC_AUC_ovr_macro"], ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    save_table(results, "model_performance")
    return results, fitted, reports, cms, scores


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def plot_class_distribution(y: pd.Series, title: str, name: str) -> None:
    counts = y.value_counts().sort_index().rename(index=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=MORANDI[:3])
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("Diabetes status")
    ax.set_ylabel("Number of participants")
    save_figure(fig, name)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(MORANDI[0], as_cmap=True), xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(), ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    save_figure(fig, name)


def plot_model_comparison(results: pd.DataFrame, name: str = "model_comparison_metrics") -> None:
    value_vars = ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "ROC_AUC_ovr_macro"]
    long = results.melt(id_vars="Model", value_vars=value_vars, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, x="Score", y="Model", hue="Metric", ax=ax, palette=MORANDI)
    ax.set_title("Model Performance Comparison")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    save_figure(fig, name)


def plot_roc_pr_curves(y_test: pd.Series, score_dict: Dict[str, np.ndarray], prefix: str) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i, (name, scores) in enumerate(score_dict.items()):
        try:
            RocCurveDisplay.from_predictions(y_bin.ravel(), scores.ravel(), name=name, ax=ax_roc, color=MORANDI[i % len(MORANDI)])
            PrecisionRecallDisplay.from_predictions(y_bin.ravel(), scores.ravel(), name=name, ax=ax_pr, color=MORANDI[i % len(MORANDI)])
        except Exception as exc:
            print(f"Curve skipped for {name}: {exc}")
    ax_roc.set_title("Micro-Averaged ROC Curves")
    ax_pr.set_title("Micro-Averaged Precision-Recall Curves")
    save_figure(fig_roc, f"{prefix}_roc_curves")
    save_figure(fig_pr, f"{prefix}_precision_recall_curves")


def descriptive_tables(df: pd.DataFrame) -> None:
    desc = df.describe().T.reset_index().rename(columns={"index": "Feature"})
    missing = pd.DataFrame({"Feature": df.columns, "Missing_Count": df.isna().sum().values, "Missing_Percent": df.isna().mean().values * 100})
    duplicates = pd.DataFrame({"Metric": ["Rows", "Columns", "Duplicate rows"], "Value": [df.shape[0], df.shape[1], int(df.duplicated().sum())]})
    class_dist = df[TARGET].astype(int).value_counts().sort_index().rename(index=CLASS_NAMES).reset_index()
    class_dist.columns = ["Class", "Count"]
    class_dist["Percent"] = class_dist["Count"] / class_dist["Count"].sum() * 100
    save_table(desc, "descriptive_statistics")
    save_table(missing, "missing_value_analysis")
    save_table(duplicates, "duplicate_analysis")
    save_table(class_dist, "class_distribution")


def plot_eda(df: pd.DataFrame) -> None:
    plot_class_distribution(df[TARGET].astype(int), "Diabetes Class Distribution in BRFSS 2015", "eda_class_distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["BMI"], bins=40, kde=True, ax=ax, color=MORANDI[0])
    ax.set_title("BMI Distribution")
    ax.set_xlabel("Body Mass Index")
    ax.set_ylabel("Number of participants")
    save_figure(fig, "eda_bmi_distribution")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=df, x=TARGET, y="BMI", ax=ax, palette=MORANDI[:3])
    ax.set_xticklabels([CLASS_NAMES[i] for i in [0, 1, 2]])
    ax.set_title("BMI by Diabetes Status")
    ax.set_xlabel("Diabetes status")
    ax.set_ylabel("Body Mass Index")
    save_figure(fig, "eda_bmi_by_diabetes_status")

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0, square=False, linewidths=0.25, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix of BRFSS Health Indicators")
    save_figure(fig, "eda_correlation_matrix")

    risk_features = ["HighBP", "HighChol", "BMI", "GenHlth", "DiffWalk", "Age", "Income", "PhysActivity"]
    means = df.groupby(TARGET)[risk_features].mean().T
    means.columns = [CLASS_NAMES[int(c)] for c in means.columns]
    means = means.reset_index().rename(columns={"index": "Feature"})
    save_table(means, "feature_means_by_class")
    long = means.melt(id_vars="Feature", var_name="Class", value_name="Mean")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=long, x="Mean", y="Feature", hue="Class", ax=ax, palette=MORANDI[:3])
    ax.set_title("Selected Health Indicator Means by Diabetes Status")
    ax.set_xlabel("Class-specific mean")
    ax.set_ylabel("")
    save_figure(fig, "eda_feature_means_by_class")


def create_preprocessed_artifacts(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test, features = split_data(df)
    preprocessor = build_preprocessor(features)
    X_train_scaled = pd.DataFrame(preprocessor.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=features, index=X_test.index)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
    before = y_train.value_counts().sort_index().rename(index=CLASS_NAMES).reset_index()
    before.columns = ["Class", "Count_Before_SMOTE"]
    after = pd.Series(y_smote).value_counts().sort_index().rename(index=CLASS_NAMES).reset_index()
    after.columns = ["Class", "Count_After_SMOTE"]
    smote_table = before.merge(after, on="Class")
    save_table(smote_table, "smote_class_distribution")
    plot_class_distribution(y_train, "Training Class Distribution Before SMOTE", "preprocessing_class_distribution_before_smote")
    plot_class_distribution(pd.Series(y_smote), "Training Class Distribution After SMOTE", "preprocessing_class_distribution_after_smote")

    cleaned_path = INTERIM_DIR / "cleaned_brfss2015.csv"
    df.to_csv(cleaned_path, index=False)
    split_payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
    }
    joblib.dump(split_payload, INTERIM_DIR / "train_test_split.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "standard_scaler_preprocessor.joblib")
    return X_train_scaled, X_test_scaled


def tune_tree_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, features: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    search_space = {
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
            {
                "model__n_estimators": [120, 180, 240],
                "model__max_depth": [8, 12, 16, None],
                "model__min_samples_leaf": [8, 15, 30],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "XGBoost": (
            XGBClassifier(objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss", tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE),
            {
                "model__n_estimators": [120, 180, 240],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.04, 0.08, 0.12],
                "model__subsample": [0.75, 0.9],
                "model__colsample_bytree": [0.75, 0.9],
            },
        ),
        "LightGBM": (
            LGBMClassifier(objective="multiclass", num_class=N_CLASSES, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            {
                "model__n_estimators": [120, 180, 240],
                "model__num_leaves": [15, 31, 45],
                "model__learning_rate": [0.04, 0.08, 0.12],
                "model__subsample": [0.75, 0.9],
                "model__colsample_bytree": [0.75, 0.9],
            },
        ),
    }
    X_fit, y_fit = sample_for_training(X_train, y_train, max_rows=22000)
    rows: List[Dict[str, Any]] = []
    best_models: Dict[str, Any] = {}
    for name, (estimator, params) in search_space.items():
        print(f"Tuning {name} with RandomizedSearchCV...")
        search = RandomizedSearchCV(
            make_pipeline(estimator, features, use_smote=True),
            param_distributions=params,
            n_iter=4,
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(X_fit, y_fit)
        model = search.best_estimator_
        metrics, report, cm, y_score = evaluate_model(model, X_test, y_test, f"Tuned {name}")
        metrics["Best_Params"] = json.dumps(search.best_params_)
        metrics["Best_CV_F1_macro"] = search.best_score_
        rows.append(metrics)
        best_models[f"Tuned {name}"] = model
        joblib.dump(model, MODELS_DIR / f"tuned_{safe_name(name)}.joblib")
        save_table(pd.DataFrame(search.cv_results_).sort_values("rank_test_score"), f"hyperparameter_results_{safe_name(name)}")
        save_table(report.reset_index().rename(columns={"index": "Class"}), f"classification_report_tuned_{safe_name(name)}")
        plot_confusion_matrix(cm, f"Tuned {name}", f"confusion_matrix_tuned_{safe_name(name)}")
    results = pd.DataFrame(rows).sort_values(["F1_macro", "ROC_AUC_ovr_macro"], ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    save_table(results, "tuned_model_performance")
    return results, best_models


def train_stacking_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
) -> Tuple[pd.DataFrame, Any, Dict[str, np.ndarray]]:
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=15, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss", tree_method="hist", n_estimators=120, max_depth=4, learning_rate=0.08, n_jobs=-1, random_state=RANDOM_STATE)),
        ("lgbm", LGBMClassifier(objective="multiclass", num_class=N_CLASSES, n_estimators=120, learning_rate=0.08, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )
    X_fit, y_fit = sample_for_training(X_train, y_train, max_rows=22000)
    model = make_pipeline(stack, features, use_smote=True)
    print("Training StackingClassifier ensemble...")
    model.fit(X_fit, y_fit)
    metrics, report, cm, y_score = evaluate_model(model, X_test, y_test, "Stacking Ensemble")
    result = pd.DataFrame([metrics])
    save_table(result, "stacking_model_performance")
    save_table(report.reset_index().rename(columns={"index": "Class"}), "classification_report_stacking_ensemble")
    plot_confusion_matrix(cm, "Stacking Ensemble", "confusion_matrix_stacking_ensemble")
    joblib.dump(model, MODELS_DIR / "stacking_ensemble.joblib")
    return result, model, {"Stacking Ensemble": y_score}


def permutation_importance_table(model: Any, X_test: pd.DataFrame, y_test: pd.Series, features: List[str], name: str) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    X_eval, y_eval = sample_for_training(X_test, y_test, max_rows=12000)
    result = permutation_importance(model, X_eval, y_eval, scoring="f1_macro", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    table = pd.DataFrame({"Feature": features, "Importance_Mean": result.importances_mean, "Importance_SD": result.importances_std}).sort_values("Importance_Mean", ascending=False)
    save_table(table, name)
    fig, ax = plt.subplots(figsize=(9, 7))
    top = table.head(15).iloc[::-1]
    ax.barh(top["Feature"], top["Importance_Mean"], xerr=top["Importance_SD"], color=MORANDI[0])
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Decrease in macro F1 after permutation")
    ax.set_ylabel("")
    save_figure(fig, f"{name}_plot")
    return table


def shap_analysis(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    if shap is None:
        raise ImportError("SHAP is not installed.")
    print("Computing SHAP values on a reproducible sample...")
    X_bg, _ = sample_for_training(X_train, pd.Series(np.zeros(len(X_train))), max_rows=250)
    X_exp, _ = sample_for_training(X_test, pd.Series(np.zeros(len(X_test))), max_rows=600)
    transformed_bg = model.named_steps["preprocess"].transform(X_bg)
    transformed_exp = model.named_steps["preprocess"].transform(X_exp)
    estimator = model.named_steps["model"]
    explainer = shap.Explainer(estimator, transformed_bg, feature_names=features)
    values = explainer(transformed_exp)
    raw = values.values
    if raw.ndim == 3:
        importance = np.abs(raw).mean(axis=(0, 2))
        plot_values = raw[:, :, 2] if raw.shape[2] > 2 else raw.mean(axis=2)
    else:
        importance = np.abs(raw).mean(axis=0)
        plot_values = raw
    table = pd.DataFrame({"Feature": features, "Mean_ABS_SHAP": importance}).sort_values("Mean_ABS_SHAP", ascending=False)
    save_table(table, "shap_feature_importance")

    plt.figure(figsize=(9, 7))
    shap.summary_plot(plot_values, transformed_exp, feature_names=features, show=False, color=MORANDI[0], max_display=15)
    fig = plt.gcf()
    fig.suptitle("SHAP Summary Plot for Diabetes Prediction", y=1.02, fontsize=15, fontweight="bold")
    save_figure(fig, "shap_summary_plot")

    fig, ax = plt.subplots(figsize=(9, 7))
    top = table.head(15).iloc[::-1]
    ax.barh(top["Feature"], top["Mean_ABS_SHAP"], color=MORANDI[2])
    ax.set_title("SHAP Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    save_figure(fig, "shap_feature_importance_plot")
    return table


def combine_final_results() -> pd.DataFrame:
    frames = []
    for path in [TABLES_DIR / "model_performance.csv", TABLES_DIR / "tuned_model_performance.csv", TABLES_DIR / "stacking_model_performance.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No model performance tables found.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop(columns=[c for c in ["Rank"] if c in combined.columns], errors="ignore")
    combined = combined.sort_values(["F1_macro", "ROC_AUC_ovr_macro"], ascending=False).reset_index(drop=True)
    combined.insert(0, "Final_Rank", np.arange(1, len(combined) + 1))
    save_table(combined, "final_model_comparison")
    plot_model_comparison(combined, "final_model_comparison_metrics")
    return combined
