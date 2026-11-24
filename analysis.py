from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "demand_forecasting.csv"
CHARACTERISTICS_PATH = BASE_DIR / "data_characteristics.md"
FINDINGS_PATH = BASE_DIR / "Findings_Report.md"
TARGET = "Demand"


def markdown_table(df, float_columns=None):
    if df.empty:
        return "_No rows._"
    output = df.copy()
    if float_columns is None:
        float_columns = []
    for column in float_columns:
        if column in output.columns:
            output[column] = output[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
    return output.to_markdown(index=False)


def cramers_v(left, right):
    contingency = pd.crosstab(left, right)
    if contingency.size == 0:
        return np.nan
    observed = contingency.to_numpy(dtype=float)
    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.sum((observed - expected) ** 2 / expected)
    n = observed.sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min((k_corr - 1), (r_corr - 1))
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denominator))


df = pd.read_csv(DATA_PATH)

print("=== BASIC STRUCTURE ===")
print("\nHEAD")
print(df.head())
print("\nTAIL")
print(df.tail())
print("\nSAMPLE(5)")
print(df.sample(5, random_state=42))
print("\nSHAPE")
print(df.shape)
print("\nINFO")
df.info()
print("\nDTYPES")
print(df.dtypes)

numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
datetime_columns = []
parsed_dates = {}
for column in df.columns:
    if "date" in column.lower() or "time" in column.lower():
        try:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                datetime_columns.append(column)
                parsed_dates[column] = parsed
        except (TypeError, ValueError):
            pass

id_columns = [
    column for column in df.columns
    if (column.lower().strip().endswith("id") and column not in datetime_columns)
    or df[column].nunique(dropna=False) == len(df)
]

print("\nCOLUMN TYPES")
print("Numerical:", numerical_columns)
print("Categorical:", categorical_columns)
print("Datetime:", datetime_columns)
print("Possible ID:", id_columns)

wrong_type_flags = {}
for column in categorical_columns:
    non_null = df[column].dropna()
    empty_count = int(non_null.astype(str).str.strip().eq("").sum())
    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
    if len(non_null) > 0 and numeric_count / len(non_null) > 0.95:
        wrong_type_flags[column] = f"{numeric_count / len(non_null):.2%} parse as numeric"
    if empty_count:
        wrong_type_flags[column] = (
            wrong_type_flags.get(column, "") + f"; {empty_count} empty strings".lstrip("; ")
        )
for column in datetime_columns:
    wrong_type_flags[column] = "stored as object/string but parseable as datetime"

print("\nSUSPICIOUS TYPE FLAGS")
print(wrong_type_flags if wrong_type_flags else "None")

print("\n=== STATISTICAL SUMMARY ===")
print("\nNUMERICAL DESCRIBE")
print(df[numerical_columns].describe())
print("\nCATEGORICAL DESCRIBE")
if categorical_columns:
    print(df[categorical_columns].describe(include="object"))

skew_kurtosis = pd.DataFrame({
    "Column": numerical_columns,
    "Skewness": [df[column].skew() for column in numerical_columns],
    "Kurtosis": [df[column].kurtosis() for column in numerical_columns],
})
skew_kurtosis["Highly Skewed (|skew| > 1)"] = skew_kurtosis["Skewness"].abs() > 1
print("\nSKEWNESS AND KURTOSIS")
print(skew_kurtosis.to_string(index=False))

unique_counts = df.nunique(dropna=True)
unique_table = pd.DataFrame({
    "Column": df.columns,
    "Unique Count": [unique_counts[column] for column in df.columns],
    "Unique Percentage": [unique_counts[column] / len(df) * 100 for column in df.columns],
})
unique_table["Constant"] = unique_table["Unique Count"] <= 1
unique_table["Near-Constant"] = (
    df.apply(lambda column: column.value_counts(dropna=False, normalize=True).iloc[0])
    > 0.95
).tolist()
unique_table["High-Cardinality Categorical"] = [
    column in (categorical_columns + id_columns)
    and unique_counts[column] / len(df) > 0.5
    for column in df.columns
]
print("\nUNIQUE VALUES")
print(unique_table.to_string(index=False))

print("\n=== DATA QUALITY ===")
missing_table = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isna().sum().to_numpy(),
    "Missing Percentage": df.isna().mean().mul(100).to_numpy(),
})
missing_table = missing_table.sort_values(
    ["Missing Count", "Column"], ascending=[False, True]
).reset_index(drop=True)
print("\nMISSING VALUES")
print(missing_table.to_string(index=False))

missing_correlation = df.isna().corr()
missing_pairs = []
for i, left in enumerate(df.columns):
    for right in df.columns[i + 1:]:
        value = missing_correlation.loc[left, right]
        if pd.notna(value) and abs(value) >= 0.5:
            missing_pairs.append((left, right, value))
print("\nMISSINGNESS PAIRS (|correlation| >= 0.5)")
print(missing_pairs if missing_pairs else "None")

duplicate_count = int(df.duplicated().sum())
print("\nFULLY DUPLICATED ROWS")
print(duplicate_count)

placeholder_values = {}
for column in df.columns:
    counts = {}
    for placeholder in [0, -1, 999, "unknown", "?", ""]:
        if column in df.columns:
            if isinstance(df[column].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[column].dtype):
                count = int(df[column].astype(str).str.strip().str.lower().eq(str(placeholder).lower()).sum())
            else:
                try:
                    count = int(df[column].eq(placeholder).sum())
                except (TypeError, ValueError):
                    count = 0
            if count:
                counts[str(placeholder)] = count
    if counts:
        placeholder_values[column] = counts

negative_values = {}
for column in numerical_columns:
    if column not in (datetime_columns):
        count = int((df[column] < 0).sum())
        if count:
            negative_values[column] = count

impossible_ranges = {}
for column in ["Price", "Competitor Pricing"]:
    if column in df.columns:
        count = int((df[column] <= 0).sum())
        if count:
            impossible_ranges[column] = f"{count} values <= 0"
if "Discount" in df.columns:
    count = int(((df["Discount"] < 0) | (df["Discount"] > 100)).sum())
    if count:
        impossible_ranges["Discount"] = f"{count} values outside 0-100"
for column in ["Promotion", "Epidemic"]:
    if column in df.columns:
        count = int((~df[column].dropna().isin([0, 1])).sum())
        if count:
            impossible_ranges[column] = f"{count} values outside 0-1"

print("\nNEGATIVE NUMERIC VALUES")
print(negative_values if negative_values else "None")
print("\nIMPOSSIBLE RANGES")
print(impossible_ranges if impossible_ranges else "None")
print("\nPLACEHOLDER VALUES")
print(placeholder_values if placeholder_values else "None")

print("\n=== OUTLIER ANALYSIS ===")
outlier_rows = []
for column in numerical_columns:
    values = df[column].dropna()
    if len(values) == 0:
        continue
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        iqr_mask = pd.Series(False, index=df.index)
    else:
        iqr_mask = (df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)
    iqr_count = int(iqr_mask.fillna(False).sum())
    iqr_percentage = iqr_count / len(df) * 100
    z_mask = pd.Series(False, index=df.index)
    if values.std(ddof=0) != 0:
        z_scores = zscore(values)
        z_mask.loc[values.index] = np.abs(z_scores) > 3
    z_count = int(z_mask.fillna(False).sum())
    z_percentage = z_count / len(df) * 100
    serious = iqr_count / len(df) > 0.05 or z_count / len(df) > 0.01
    outlier_rows.append({
        "Column": column,
        "IQR Outlier Count": iqr_count,
        "IQR Outlier Percentage": iqr_percentage,
        "Z-score Outlier Count": z_count,
        "Z-score Outlier Percentage": z_percentage,
        "Serious Outlier Problem": serious,
    })
outlier_table = pd.DataFrame(outlier_rows)
print(outlier_table.to_string(index=False))

print("\n=== TARGET VARIABLE ANALYSIS ===")
target_dtype = df[TARGET].dtype
target_is_numeric = pd.api.types.is_numeric_dtype(target_dtype)
target_unique = df[TARGET].nunique(dropna=True)
target_is_classification = not target_is_numeric or target_unique <= 20

target_distribution = None
if target_is_classification:
    target_distribution = df[TARGET].value_counts(dropna=False)
    target_percentage = df[TARGET].value_counts(normalize=True, dropna=False).mul(100)
    target_table = pd.DataFrame({
        "Class": [str(index) for index in target_distribution.index],
        "Count": target_distribution.to_numpy(),
        "Percentage": target_percentage.to_numpy(),
    })
    max_pct = target_percentage.max()
    min_pct = target_percentage.min()
    imbalance = max_pct / min_pct >= 2 if min_pct > 0 else True
    print("\nCLASSIFICATION TARGET")
    print(target_table.to_string(index=False))
    print("Imbalanced:", imbalance)
else:
    target_skew = df[TARGET].skew()
    print("\nREGRESSION TARGET")
    print(df[TARGET].describe())
    print("Skewness:", target_skew)
    print("Log transform may be needed later:", abs(target_skew) > 1)
    imbalance = False

feature_target_relationship = []
for column in df.columns:
    if column == TARGET:
        continue
    if pd.api.types.is_numeric_dtype(df[column].dtype):
        pairs = df[[column, TARGET]].dropna()
        pearson = pairs[column].corr(pairs[TARGET], method="pearson") if len(pairs) > 1 else np.nan
        spearman = pairs[column].corr(pairs[TARGET], method="spearman") if len(pairs) > 1 else np.nan
        feature_target_relationship.append({
            "Feature": column,
            "Type": "Numerical",
            "Target Relationship": f"Pearson={pearson:.6f}; Spearman={spearman:.6f}",
        })
    elif column not in id_columns:
        groups = df[[column, TARGET]].dropna().groupby(column, observed=True)[TARGET]
        means = groups.mean().sort_values(ascending=False)
        summary = "; ".join(
            f"{str(index)}={value:.6f}" for index, value in means.head(10).items()
        )
        feature_target_relationship.append({
            "Feature": column,
            "Type": "Categorical",
            "Target Relationship": f"Top group means: {summary}",
        })
print("\nFEATURE-TARGET RELATIONSHIPS")
print(pd.DataFrame(feature_target_relationship).to_string(index=False))

print("\n=== RELATIONSHIP ANALYSIS ===")
pearson_corr = df[numerical_columns].corr(method="pearson")
spearman_corr = df[numerical_columns].corr(method="spearman")
print("\nPEARSON CORRELATION MATRIX")
print(pearson_corr.to_string())
print("\nSPEARMAN CORRELATION MATRIX")
print(spearman_corr.to_string())

big_differences = []
for i, left in enumerate(numerical_columns):
    for right in numerical_columns[i + 1:]:
        pearson_value = pearson_corr.loc[left, right]
        spearman_value = spearman_corr.loc[left, right]
        difference = abs(pearson_value - spearman_value)
        if pd.notna(difference) and difference > 0.2:
            big_differences.append({
                "Feature 1": left,
                "Feature 2": right,
                "Pearson": pearson_value,
                "Spearman": spearman_value,
                "Absolute Difference": difference,
            })
print("\nBIG PEARSON-SPEARMAN DIFFERENCES (>0.2)")
print(pd.DataFrame(big_differences).to_string(index=False) if big_differences else "None")

correlation_candidates = []
for i, left in enumerate(numerical_columns):
    for right in numerical_columns[i + 1:]:
        if left == TARGET or right == TARGET:
            continue
        value = pearson_corr.loc[left, right]
        if pd.notna(value) and abs(value) > 0.7:
            correlation_candidates.append({
                "Feature 1": left,
                "Feature 2": right,
                "Pearson Correlation": value,
            })
print("\nMULTICOLLINEARITY CANDIDATES (|Pearson| > 0.7)")
print(pd.DataFrame(correlation_candidates).to_string(index=False) if correlation_candidates else "None")

vif_features = [column for column in numerical_columns if column != TARGET]
vif_complete = df[vif_features].dropna()
vif_table = pd.DataFrame()
if vif_complete.shape[1] > 0 and vif_complete.shape[0] > vif_complete.shape[1]:
    vif_numeric = vif_complete.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    vif_numeric = vif_numeric.loc[:, vif_numeric.nunique(dropna=True) > 1]
    if vif_numeric.shape[1] > 0:
        try:
            vif_values = []
            scaled = StandardScaler().fit_transform(vif_numeric)
            scaled = pd.DataFrame(scaled, columns=vif_numeric.columns)
            for i, column in enumerate(scaled.columns):
                vif_values.append({
                    "Feature": column,
                    "VIF": variance_inflation_factor(scaled.to_numpy(), i),
                })
            vif_table = pd.DataFrame(vif_values)
        except Exception as error:
            vif_table = pd.DataFrame([{"Feature": "VIF not calculated", "VIF": str(error)}])
print("\nVIF")
print(vif_table.to_string(index=False) if not vif_table.empty else "Not calculable")

cramers_columns = [
    column for column in categorical_columns
    if column not in id_columns
]
cramers_rows = []
for i, left in enumerate(cramers_columns):
    for right in cramers_columns[i + 1:]:
        value = cramers_v(df[left].fillna("__MISSING__"), df[right].fillna("__MISSING__"))
        if pd.notna(value) and value > 0.3:
            cramers_rows.append({
                "Feature 1": left,
                "Feature 2": right,
                "Cramer's V": value,
            })
print("\nCRAMER'S V RELATIONS (>0.3)")
print(pd.DataFrame(cramers_rows).to_string(index=False) if cramers_rows else "None")

mi_features = [column for column in df.columns if column != TARGET]
mi_encoded = df[mi_features].copy()
for column in mi_encoded.columns:
    if pd.api.types.is_numeric_dtype(mi_encoded[column].dtype):
        mi_encoded[column] = mi_encoded[column].fillna(mi_encoded[column].median())
    else:
        encoder = LabelEncoder()
        mi_encoded[column] = encoder.fit_transform(mi_encoded[column].astype(str).fillna("__MISSING__"))
valid_mi_rows = mi_encoded.notna().all(axis=1) & df[TARGET].notna()
mi_scores = mutual_info_regression(
    mi_encoded.loc[valid_mi_rows],
    df.loc[valid_mi_rows, TARGET],
    random_state=42,
)
mi_table = pd.DataFrame({
    "Feature": mi_features,
    "Mutual Information": mi_scores,
}).sort_values("Mutual Information", ascending=False)
print("\nMUTUAL INFORMATION WITH TARGET")
print(mi_table.to_string(index=False))

top_skewed = skew_kurtosis[skew_kurtosis["Highly Skewed (|skew| > 1)"]]["Column"].tolist()
top_missing = missing_table[missing_table["Missing Percentage"] > 0].head(10)
top_outliers = outlier_table[outlier_table["Serious Outlier Problem"]]["Column"].tolist()
high_cardinality = unique_table[unique_table["High-Cardinality Categorical"]]["Column"].tolist()
near_constant = unique_table[unique_table["Near-Constant"]]["Column"].tolist()

characteristics = f"""# Data Characteristics

## Dataset Overview

- Source: `demand_forecasting.csv`
- Row count: {len(df)}
- Column count: {len(df.columns)}
- Target variable: `{TARGET}`
- Target type: {"classification" if target_is_classification else "regression"}
- Observation period: {datetime_columns[0] + " available" if datetime_columns else "Not identified"}

## Column Names

{chr(10).join(f"- `{column}`" for column in df.columns)}

## Basic Structure

### Shape

- Rows: {len(df)}
- Columns: {len(df.columns)}

### Data Types

{markdown_table(pd.DataFrame(list(zip(df.columns, [str(dtype) for dtype in df.dtypes.to_numpy()])), columns=["Column", "Data Type"]))}

### Column Separation

- Numerical: {", ".join(f"`{column}`" for column in numerical_columns)}
- Categorical: {", ".join(f"`{column}`" for column in categorical_columns)}
- Datetime: {", ".join(f"`{column}`" for column in datetime_columns) if datetime_columns else "None"}
- Possible ID: {", ".join(f"`{column}`" for column in id_columns) if id_columns else "None"}

### Suspicious or Wrong Types

{chr(10).join(f"- `{column}`: {reason}" for column, reason in wrong_type_flags.items()) if wrong_type_flags else "- None detected"}

## Statistical Summary

### Numerical Describe

{markdown_table(df[numerical_columns].describe().T.reset_index().rename(columns={"index": "Column"}), ["mean", "std", "min", "25%", "50%", "75%", "max"])}

### Categorical Describe

{markdown_table(df[categorical_columns].describe(include="object").T.reset_index().rename(columns={"index": "Column"})) if categorical_columns else "_No categorical object columns._"}

### Skewness and Kurtosis

{markdown_table(skew_kurtosis, ["Skewness", "Kurtosis"])}

Highly skewed columns (|skew| > 1): {", ".join(f"`{column}`" for column in top_skewed) if top_skewed else "None"}

### Unique Values

{markdown_table(unique_table, ["Unique Percentage"])}

## Data Quality

### Missing Values

{markdown_table(missing_table, ["Missing Percentage"])}

### Missingness Together

- Missingness pairs with absolute correlation >= 0.5: {", ".join(f"`{left}` / `{right}` ({value:.6f})" for left, right, value in missing_pairs) if missing_pairs else "None"}

### Duplicate Rows

- Fully duplicated row count: {duplicate_count}

### Negative Numeric Values

{chr(10).join(f"- `{column}`: {count}" for column, count in negative_values.items()) if negative_values else "- None"}

### Impossible Ranges

{chr(10).join(f"- `{column}`: {reason}" for column, reason in impossible_ranges.items()) if impossible_ranges else "- None"}

### Placeholder Values

{chr(10).join(f"- `{column}`: " + ", ".join(f"{placeholder} ({count})" for placeholder, count in counts.items()) for column, counts in placeholder_values.items()) if placeholder_values else "- None"}

## Outlier Analysis

{markdown_table(outlier_table, ["IQR Outlier Percentage", "Z-score Outlier Percentage"])}

Serious outlier columns: {", ".join(f"`{column}`" for column in top_outliers) if top_outliers else "None"}

## Target Variable Analysis

{"### Classification Distribution" if target_is_classification else "### Regression Distribution"}

{markdown_table(target_table, ["Percentage"]) if target_is_classification else markdown_table(df[TARGET].describe().to_frame().T.reset_index(drop=True))}

{"- Imbalanced: " + str(imbalance) if target_is_classification else f"- Skewness: {target_skew:.6f}\n- Log transform may be needed: {abs(target_skew) > 1}"}

### Relationship with Features

{markdown_table(pd.DataFrame(feature_target_relationship))}

## Relationship Analysis

### Pearson Correlation Matrix

{markdown_table(pearson_corr.reset_index().rename(columns={"index": "Column"}), numerical_columns)}

### Spearman Correlation Matrix

{markdown_table(spearman_corr.reset_index().rename(columns={"index": "Column"}), numerical_columns)}

### Pearson-Spearman Large Differences (>0.2)

{markdown_table(pd.DataFrame(big_differences), ["Pearson", "Spearman", "Absolute Difference"]) if big_differences else "None"}

### Multicollinearity Candidates (|Pearson| > 0.7)

{markdown_table(pd.DataFrame(correlation_candidates), ["Pearson Correlation"]) if correlation_candidates else "None"}

### VIF

{markdown_table(vif_table, ["VIF"]) if not vif_table.empty else "Not calculable"}

### Cramer's V (Categorical Pairs > 0.3)

{markdown_table(pd.DataFrame(cramers_rows), ["Cramer's V"]) if cramers_rows else "None"}

### Mutual Information with Target

{markdown_table(mi_table, ["Mutual Information"])}

## Key Observations & Risks

{chr(10).join(f"- {row['Column']}: {row['Missing Percentage']:.6f}% missing" for _, row in top_missing.iterrows()) if not top_missing.empty else "- No missing values detected"}
- Highly skewed numerical columns: {", ".join(f"`{column}`" for column in top_skewed) if top_skewed else "None"}
- Serious IQR/Z-score outlier columns: {", ".join(f"`{column}`" for column in top_outliers) if top_outliers else "None"}
- High-cardinality categorical columns: {", ".join(f"`{column}`" for column in high_cardinality) if high_cardinality else "None"}
- Near-constant columns: {", ".join(f"`{column}`" for column in near_constant) if near_constant else "None"}
- Multicollinearity candidates: {", ".join(f"`{left}` / `{right}`" for left, right in [(row["Feature 1"], row["Feature 2"]) for _, row in pd.DataFrame(correlation_candidates).iterrows()]) if correlation_candidates else "None"}
- Target imbalance risk: {"Present" if target_is_classification and imbalance else "Not detected"}
- Leakage suspects: Candidate IDs and columns unavailable at prediction time require business-time validation; no conclusion is made at observation stage.
- Data cleaning items: negative values, impossible ranges, placeholders, duplicate rows, missing values, and outliers should be handled only in the later cleaning stage.
"""

CHARACTERISTICS_PATH.write_text(characteristics, encoding="utf-8")

if FINDINGS_PATH.exists():
    findings_text = FINDINGS_PATH.read_text(encoding="utf-8")
else:
    findings_text = "# Findings Report\n"
findings_text = findings_text.replace(
    "## Key EDA Insights\n\nRecord important findings from exploratory data analysis here.",
    "## Key EDA Insights\n\n" + "\n".join(
        f"- {row['Feature']}: {row['Target Relationship']}"
        for row in feature_target_relationship
    )
)
observation_heading = "## Data Understanding (Observation Only)"
observation_section = f"""{observation_heading}

- Dataset shape: {len(df)} rows and {len(df.columns)} columns.
- Target variable: `{TARGET}`.
- Target type: {"classification" if target_is_classification else "regression"}.
- Fully duplicated rows: {duplicate_count}.
- Highly skewed numerical columns: {", ".join(f"`{column}`" for column in top_skewed) if top_skewed else "None"}.
- Serious outlier columns: {", ".join(f"`{column}`" for column in top_outliers) if top_outliers else "None"}.
- Multicollinearity candidates: {len(correlation_candidates)} Pearson pairs with absolute correlation greater than 0.7.
- Data characteristics and all executed values were recorded in `data_characteristics.md`.
"""
if observation_heading in findings_text:
    prefix = findings_text.split(observation_heading)[0]
    findings_text = prefix + observation_section
else:
    findings_text = findings_text.rstrip() + "\n\n" + observation_section
FINDINGS_PATH.write_text(findings_text, encoding="utf-8")

print("\n=== FIVE IMPORTANT FINDINGS ===")
important_findings = [
    f"Dataset shape is {len(df)} rows and {len(df.columns)} columns.",
    f"Fully duplicated rows: {duplicate_count}.",
    f"Highly skewed numerical columns: {', '.join(top_skewed) if top_skewed else 'none detected'}.",
    f"Serious outlier columns: {', '.join(top_outliers) if top_outliers else 'none detected'}.",
    f"Multicollinearity candidate pairs: {len(correlation_candidates)}.",
]
for index, finding in enumerate(important_findings, start=1):
    print(f"{index}. {finding}")
print("\nREPORTS UPDATED")
print(CHARACTERISTICS_PATH)
print(FINDINGS_PATH)
