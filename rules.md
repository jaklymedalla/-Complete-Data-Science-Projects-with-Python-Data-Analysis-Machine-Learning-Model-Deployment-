# Data Science Workflow Rules

Follow this workflow in order throughout the project:

1. Problem definition
2. Data understanding
3. Data cleaning
4. EDA
5. Train/test split
6. Preprocessing
7. Feature engineering
8. Modeling
9. Evaluation
10. Reporting

## Data Leakage Prevention

- Always split the data into training and test sets before scaling, encoding, imputation, or feature selection.
- Fit all transformers only on the training set.
- Apply fitted transformers to the test set only for transformation.
- Never use test data to fit preprocessing or feature-selection operations.

## Reproducibility

- Use a fixed `random_state` in reproducible operations.
- For classification problems, stratify the train/test split when appropriate.
- Record the `random_state`, split method, and resulting sample counts when the split is created.

## Data Quality

- Handle missing values deliberately.
- Handle duplicate records deliberately.
- Handle outliers deliberately.
- Document every data decision with:
  - the issue observed;
  - the action taken;
  - the reason for the action;
  - the affected columns or rows;
  - the impact of the action.

## Preprocessing

- Prefer sklearn `Pipeline` objects.
- Prefer sklearn `ColumnTransformer` for separate numeric and categorical preprocessing.
- Keep preprocessing steps consistent and reproducible.

## Modeling

- Always build a simple baseline model before trying complex models.
- Use cross-validation on the training set for model selection and hyperparameter tuning.
- Use the test set only once at the end for final evaluation.
- Do not tune models or make model-selection decisions using test-set performance.

## Evaluation

- Choose evaluation metrics that fit the problem type and data distribution.
- Do not rely only on accuracy for imbalanced classification problems.
- Use appropriate metrics such as precision, recall, F1, ROC-AUC, PR-AUC, or per-class metrics when relevant.

## Evidence and Reporting

- Never invent or assume numbers.
- Every value written to the reports must come from actually running code.
- Clearly distinguish executed findings from interpretation or hypotheses.
- After every completed step, update `data_characteristics.md` and `Findings_Report.md`.
