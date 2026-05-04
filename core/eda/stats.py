import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, shapiro

# =========================
# OUTLIERS (IQR METHOD)
# =========================
def detect_outliers_iqr(series: pd.Series) -> list:

    series = series.dropna()

    if series.empty:
        return []

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return series[(series < lower) | (series > upper)].tolist()


# =========================
# NORMALITY TEST
# =========================
def normality_test(series: pd.Series) -> dict:

    s = series.dropna()

    if len(s) < 8:
        return {"stat": None, "p": None}

    if len(s) > 5000:
        s = s.sample(5000, random_state=42)

    if s.std() == 0:
        return {"stat": None, "p": None}

    stat, p = shapiro(s)

    return {"stat": float(stat), "p": float(p)}


# =========================
# CRAMER'S V
# =========================
def cramers_v(x: pd.Series, y: pd.Series) -> float:

    confusion_matrix = pd.crosstab(x, y)

    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return np.nan

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    k = min(confusion_matrix.shape) - 1

    return np.sqrt(chi2 / (n * k)) if k > 0 else np.nan


# =========================
# NUMERIC STATISTICS
# =========================
def numeric_stats(df: pd.DataFrame, roles: dict) -> dict:

    numeric = df.select_dtypes(include="number")

    # escludi ID
    exclude = set(roles.get("id_columns", []))
    numeric = numeric.drop(columns=exclude, errors="ignore")

    stats = {}

    for col in numeric.columns:

        s = numeric[col].dropna()

        if s.empty:
            continue

        mean = float(s.mean())
        std = float(s.std()) if s.std() != 0 else np.nan

        outliers = detect_outliers_iqr(s)

        stats[col] = {
            "mean": mean,
            "median": float(s.median()),
            "std": std,
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "outliers_iqr": outliers,
            "outlier_count": len(outliers),
            "outlier_pct": len(outliers) / len(s) * 100,
            "normality": normality_test(s),
            "coef_variation": float(std / mean) if mean != 0 and not np.isnan(std) else np.nan,
        }

    # -------------------------
    # PEARSON CORRELATION MATRIX
    # -------------------------
    correlation_matrix = {}
    if len(numeric.columns) >= 2:
        corr_df = numeric.corr(method="pearson")
        for i, col1 in enumerate(numeric.columns):
            correlation_matrix[col1] = {}
            for j, col2 in enumerate(numeric.columns):
                if i < j:
                    val = corr_df.loc[col1, col2]
                    correlation_matrix[col1][col2] = float(val) if not np.isnan(val) else None

    return {
        "columns": stats,
        "pearson_correlation": correlation_matrix
    }


# =========================
# CATEGORICAL STATISTICS
# =========================
def categorical_stats(df: pd.DataFrame, roles: dict) -> dict:

    cat_cols = roles.get("categorical_columns", [])
    lookup_cols = roles.get("lookup_columns", [])
    all_cat = cat_cols + lookup_cols

    stats = {}

    # -------------------------
    # BASIC STATS (Per tutte le testuali/categoriali)
    # -------------------------
    for col in all_cat:
        s = df[col]
        stats[col] = {
            "n_unique": int(s.nunique(dropna=True)),
            "mode": str(s.mode().iloc[0]) if not s.mode().empty else None,
            "value_counts": s.value_counts().head(10).to_dict()
        }

    # -------------------------
    # CHI-SQUARE (Solo per vere variabili categoriali, no nomi/lookup)
    # -------------------------
    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[:2]
        table = pd.crosstab(df[c1], df[c2])

        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(table)

            stats["chi2"] = {
                "columns": [c1, c2],
                "stat": float(chi2),
                "p": float(p)
            }

    # -------------------------
    # CRAMER'S V MATRIX (Solo per vere variabili categoriali)
    # -------------------------
    cramer_matrix = {}

    for i, col1 in enumerate(cat_cols):
        cramer_matrix[col1] = {}

        for j, col2 in enumerate(cat_cols):
            if i < j:
                cramer_matrix[col1][col2] = float(
                    cramers_v(df[col1], df[col2])
                )

    stats["cramers_v"] = cramer_matrix

    return stats


# =========================
# MAIN ENTRY POINT
# =========================
def basic_stats(df: pd.DataFrame, roles: dict) -> dict:

    return {
        "numerical": numeric_stats(df, roles),
        "categorical": categorical_stats(df, roles)
    }