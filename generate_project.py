"""Generate the BRFSS diabetes project notebooks and report scaffolding."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

from project_utils import CODE_DIR, ROOT


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


COMMON_IMPORTS = """
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "code":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from project_utils import *
set_reproducibility()

print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset path: {DATA_PATH}")
"""


def write_notebook(filename: str, title: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(f"# {title}\n\nThis notebook is part of a reproducible machine learning project using the BRFSS 2015 diabetes health indicators dataset. All outputs are saved automatically to the `results` directory."),
        code(COMMON_IMPORTS),
    ] + cells
    nbf.write(nb, CODE_DIR / filename)


def generate_notebooks():
    write_notebook(
        "01_Data_Inspection_and_EDA.ipynb",
        "01 Data Inspection and Exploratory Data Analysis",
        [
            md(
                """
                ## Purpose

                This notebook inspects the raw BRFSS 2015 dataset before modeling. We examine data dimensions, variable types, missing values, duplicate rows, class imbalance, and important descriptive relationships. In academic machine learning, this step is essential because model performance depends strongly on data quality and the distribution of the target variable.
                """
            ),
            code(
                """
                print("Loading raw data...")
                raw_df = load_raw_data()
                print(f"Raw dataset shape: {raw_df.shape}")
                display(raw_df.head())
                """
            ),
            md("## Data Quality Summary\n\nThe following tables document missing values, duplicate records, descriptive statistics, and the target distribution. These artifacts are saved as both CSV and XLSX files for reporting."),
            code(
                """
                descriptive_tables(raw_df)
                summary = pd.DataFrame({
                    "Metric": ["Rows", "Columns", "Missing cells", "Duplicate rows"],
                    "Value": [raw_df.shape[0], raw_df.shape[1], int(raw_df.isna().sum().sum()), int(raw_df.duplicated().sum())]
                })
                save_table(summary, "eda_dataset_summary")
                display(summary)
                display(pd.read_csv(TABLES_DIR / "class_distribution.csv"))
                """
            ),
            md("## Exploratory Visualizations\n\nThe plots below use a restrained Morandi color palette, high-resolution export settings, and academic labels. The target distribution plot highlights the major class imbalance: prediabetes is much less frequent than the other two classes."),
            code(
                """
                print("Generating EDA figures...")
                plot_eda(raw_df)
                print(f"Figures saved to: {FIGURES_DIR}")
                """
            ),
            md("## Cleaned Dataset\n\nDuplicate rows are removed for the modeling workflow. Removing exact duplicate rows reduces repeated observations that can bias training and evaluation. No missing-value imputation is required because the dataset contains no missing cells."),
            code(
                """
                clean_df = clean_data(raw_df, remove_duplicates=True)
                save_table(pd.DataFrame({
                    "Metric": ["Rows before cleaning", "Rows after duplicate removal", "Removed duplicate rows"],
                    "Value": [len(raw_df), len(clean_df), len(raw_df) - len(clean_df)]
                }), "cleaning_summary")
                clean_df.to_csv(INTERIM_DIR / "cleaned_brfss2015.csv", index=False)
                print(f"Cleaned dataset shape: {clean_df.shape}")
                display(clean_df.head())
                """
            ),
            md(
                """
                ## Academic Interpretation

                The BRFSS target variable is highly imbalanced, with most participants belonging to the no-diabetes group and a much smaller prediabetes group. This imbalance means that accuracy alone can be misleading: a model may appear strong by favoring the majority class. Therefore, later notebooks emphasize macro-averaged precision, recall, F1-score, ROC-AUC, and precision-recall curves.
                """
            ),
        ],
    )

    write_notebook(
        "02_Data_Preprocessing.ipynb",
        "02 Data Preprocessing",
        [
            md(
                """
                ## Purpose

                This notebook prepares the data for machine learning. It applies duplicate removal, stratified train/test splitting, feature scaling with `StandardScaler`, and class balancing with SMOTE. Stratification preserves the original target proportions in both training and test sets. SMOTE is applied only to the training data to avoid test-set leakage.
                """
            ),
            code(
                """
                raw_df = load_raw_data()
                clean_df = clean_data(raw_df, remove_duplicates=True)
                X_train, X_test, y_train, y_test, features = split_data(clean_df)

                split_summary = pd.DataFrame({
                    "Split": ["Training", "Test"],
                    "Rows": [len(X_train), len(X_test)],
                    "Feature columns": [len(features), len(features)]
                })
                save_table(split_summary, "train_test_split_summary")
                display(split_summary)
                """
            ),
            md("## Standardization and SMOTE\n\n`StandardScaler` gives continuous and ordinal variables comparable numeric scale. SMOTE synthesizes minority-class training examples, improving the model's opportunity to learn prediabetes and diabetes patterns instead of mainly optimizing for the majority class."),
            code(
                """
                X_train_scaled, X_test_scaled = create_preprocessed_artifacts(clean_df)
                display(pd.read_csv(TABLES_DIR / "smote_class_distribution.csv"))
                print("Preprocessing artifacts saved.")
                """
            ),
            md("## Feature Engineering Notes\n\nThe BRFSS dataset already provides clinically meaningful engineered indicators, including blood pressure, cholesterol, BMI, physical activity, general health, age group, education, and income. We preserve the original interpretable feature names so that SHAP and feature-importance results remain useful for public health interpretation."),
            code(
                """
                feature_info = pd.DataFrame({
                    "Feature": features,
                    "Role": ["Predictor"] * len(features),
                    "Preprocessing": ["StandardScaler"] * len(features)
                })
                save_table(feature_info, "feature_preprocessing_plan")
                display(feature_info)
                """
            ),
        ],
    )

    write_notebook(
        "03_Baseline_Models.ipynb",
        "03 Baseline Models",
        [
            md(
                """
                ## Purpose

                This notebook trains six required machine learning models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and Support Vector Machine. Each model is evaluated on the same held-out stratified test set using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
                """
            ),
            code(
                """
                clean_df = clean_data(load_raw_data(), remove_duplicates=True)
                X_train, X_test, y_train, y_test, features = split_data(clean_df)
                results, fitted_models, reports, cms, score_dict = train_evaluate_models(
                    X_train, y_train, X_test, y_test, features, max_rows=55000
                )
                display(results)
                """
            ),
            md("## Confusion Matrices and Curves\n\nConfusion matrices show which classes are confused with each other. ROC and precision-recall curves summarize discrimination across classification thresholds."),
            code(
                """
                for model_name, cm in cms.items():
                    plot_confusion_matrix(cm, model_name, f"confusion_matrix_{safe_name(model_name)}")
                plot_model_comparison(results, "baseline_model_comparison_metrics")
                plot_roc_pr_curves(y_test, score_dict, "baseline_models")
                print("Baseline model figures and tables saved.")
                """
            ),
            md("## Interpretation\n\nMacro F1-score is especially important here because it treats all three classes equally. A high weighted F1-score can still hide weak prediabetes recognition, while macro F1 penalizes poor minority-class performance."),
        ],
    )

    write_notebook(
        "04_Advanced_Models_and_Tuning.ipynb",
        "04 Advanced Models and Hyperparameter Tuning",
        [
            md(
                """
                ## Purpose

                This notebook tunes Random Forest, XGBoost, and LightGBM with `RandomizedSearchCV`. Hyperparameter tuning searches for model settings that improve macro F1-score under cross-validation. The search is intentionally reproducible through fixed random seeds.
                """
            ),
            code(
                """
                clean_df = clean_data(load_raw_data(), remove_duplicates=True)
                X_train, X_test, y_train, y_test, features = split_data(clean_df)
                tuned_results, tuned_models = tune_tree_models(X_train, y_train, X_test, y_test, features)
                display(tuned_results)
                """
            ),
            md("## Tuning Interpretation\n\nTree-based models can capture nonlinear interactions among risk indicators such as BMI, blood pressure, cholesterol, age, and general health. Tuning controls complexity so the models remain expressive without overfitting."),
            code(
                """
                plot_model_comparison(tuned_results, "tuned_model_comparison_metrics")
                print("Tuned model tables, confusion matrices, and saved models are available in the results directory.")
                """
            ),
        ],
    )

    write_notebook(
        "05_Ensemble_Learning.ipynb",
        "05 Ensemble Learning",
        [
            md(
                """
                ## Purpose

                This notebook trains a `StackingClassifier` ensemble. Stacking combines several strong base learners and then trains a final meta-model on their predicted probabilities. The goal is to use complementary strengths: Random Forest handles robust bagged trees, XGBoost captures boosted nonlinear patterns, and LightGBM provides efficient gradient boosting.
                """
            ),
            code(
                """
                clean_df = clean_data(load_raw_data(), remove_duplicates=True)
                X_train, X_test, y_train, y_test, features = split_data(clean_df)
                stacking_results, stacking_model, stacking_scores = train_stacking_model(X_train, y_train, X_test, y_test, features)
                display(stacking_results)
                """
            ),
            md("## Ensemble Evaluation\n\nThe same held-out test set is used for the ensemble so it can be compared fairly with single models."),
            code(
                """
                plot_roc_pr_curves(y_test, stacking_scores, "stacking_ensemble")
                print("Stacking ensemble saved.")
                """
            ),
        ],
    )

    write_notebook(
        "06_Model_Explainability_SHAP.ipynb",
        "06 Model Explainability with SHAP",
        [
            md(
                """
                ## Purpose

                This notebook applies SHAP explainability to a tree-based diabetes classifier. SHAP estimates how much each feature contributes to predictions. This is important for academic reporting because predictive accuracy alone does not explain which public health factors are associated with model decisions.
                """
            ),
            code(
                """
                clean_df = clean_data(load_raw_data(), remove_duplicates=True)
                X_train, X_test, y_train, y_test, features = split_data(clean_df)

                model_path = MODELS_DIR / "tuned_lightgbm.joblib"
                if model_path.exists():
                    explain_model = joblib.load(model_path)
                    print("Loaded tuned LightGBM for SHAP.")
                else:
                    print("Tuned LightGBM not found; training baseline LightGBM for SHAP.")
                    explain_model = make_pipeline(base_estimators()["LightGBM"], features, use_smote=True)
                    X_fit, y_fit = sample_for_training(X_train, y_train, max_rows=35000)
                    explain_model.fit(X_fit, y_fit)
                    joblib.dump(explain_model, MODELS_DIR / "lightgbm_for_shap.joblib")

                shap_table = shap_analysis(explain_model, X_train, X_test, features)
                display(shap_table.head(15))
                """
            ),
            md("## Public Health Interpretation\n\nThe highest-ranked SHAP features identify variables the model uses most strongly when separating diabetes classes. In a public health context, features such as BMI, general health, high blood pressure, age, difficulty walking, cholesterol, and income may reflect metabolic risk, chronic disease burden, access to resources, and social determinants of health. SHAP does not prove causation; it explains the trained model's predictive behavior."),
        ],
    )

    write_notebook(
        "07_Final_Evaluation_and_Comparison.ipynb",
        "07 Final Evaluation and Comparison",
        [
            md(
                """
                ## Purpose

                This notebook combines baseline, tuned, and ensemble results into one final ranking. It also generates final publication-quality comparison figures and confirms that key artifacts exist.
                """
            ),
            code(
                """
                final_results = combine_final_results()
                display(final_results)
                """
            ),
            md("## Final Artifact Verification\n\nThe project should contain notebooks, model files, figures, tables, and reports. This verification table supports reproducibility checks for grading."),
            code(
                """
                required_paths = [
                    *[CODE_DIR / f for f in [
                        "01_Data_Inspection_and_EDA.ipynb",
                        "02_Data_Preprocessing.ipynb",
                        "03_Baseline_Models.ipynb",
                        "04_Advanced_Models_and_Tuning.ipynb",
                        "05_Ensemble_Learning.ipynb",
                        "06_Model_Explainability_SHAP.ipynb",
                        "07_Final_Evaluation_and_Comparison.ipynb",
                    ]],
                    TABLES_DIR / "final_model_comparison.csv",
                    TABLES_DIR / "shap_feature_importance.csv",
                    FIGURES_DIR / "final_model_comparison_metrics.png",
                    MODELS_DIR / "stacking_ensemble.joblib",
                ]
                verification = pd.DataFrame({
                    "Artifact": [str(p.relative_to(PROJECT_ROOT)) for p in required_paths],
                    "Exists": [p.exists() for p in required_paths]
                })
                save_table(verification, "artifact_verification")
                display(verification)
                if not verification["Exists"].all():
                    missing = verification.loc[~verification["Exists"], "Artifact"].tolist()
                    raise FileNotFoundError(f"Missing required artifacts: {missing}")
                """
            ),
            md("## Final Interpretation\n\nThe final ranking should be interpreted using macro F1 and ROC-AUC together. Macro F1 reflects balanced class recognition, while ROC-AUC reflects the ability to rank classes probabilistically. Because diabetes screening has public health consequences, minority-class recall should be discussed alongside overall accuracy."),
        ],
    )


def write_requirements():
    text = """numpy>=2.0
pandas>=2.0
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
imbalanced-learn>=0.12
xgboost>=2.0
lightgbm>=4.0
shap>=0.45
joblib>=1.3
nbformat>=5.9
nbclient>=0.10
openpyxl>=3.1
python-docx>=1.1
jupyter>=1.0
"""
    (ROOT / "requirements.txt").write_text(text, encoding="utf-8")


def main():
    generate_notebooks()
    write_requirements()
    print("Generated notebooks and requirements.txt")


if __name__ == "__main__":
    main()
