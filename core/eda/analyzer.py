import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_object_dtype
)

from core.io import load_file
from core.eda.stats import basic_stats as compute_statistics
from core.eda.visualizer import generate_plots, generate_mapping_plots


# =========================
# TYPE DETECTION
# =========================
def detect_column_type(series: pd.Series) -> str:

    if len(series) == 0:
        return "unknown"

    s = series.dropna()

    if len(s) == 0:
        return "unknown"

    if is_numeric_dtype(series):
        return "numerical"

    if is_datetime64_any_dtype(series):
        return "datetime"

    if is_bool_dtype(series):
        return "boolean"

    if is_object_dtype(series) or series.dtype == "string":

        s = s.astype(str)

        nunique = s.nunique()
        nunique_ratio = nunique / len(s)
        avg_len = s.str.len().mean()

        if avg_len > 40 and nunique_ratio < 0.3:
            return "text"

        return "categorical"

    return "unknown"


# =========================
# TYPES
# =========================
def infer_column_types(df):
    types = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
        "unknown": []
    }

    for col in df.columns:
        t = detect_column_type(df[col])
        types[t].append(col)

    return types


def build_type_map(types):
    return {
        col: t
        for t, cols in types.items()
        for col in cols
    }


# =========================
# FEATURE ROLES
# =========================
def detect_feature_roles(df, type_map):

    roles = {
        "id_columns": [],
        "lookup_columns": [],
        "categorical_columns": [],
        "numerical_columns": [],
        "datetime_columns": [],
        "boolean_columns": [],
        "text_columns": [],
        "unknown": []
    }

    for col in df.columns:

        s = df[col].dropna()

        if len(s) == 0:
            roles["unknown"].append(col)
            continue

        col_lower = col.lower()
        nunique_ratio = s.nunique() / len(s)
        t = type_map.get(col, "categorical")

        if col_lower.endswith("id"):
            roles["id_columns"].append(col)
            continue

        if t == "numerical":
            roles["numerical_columns"].append(col)
            continue

        if t == "datetime":
            roles["datetime_columns"].append(col)
            continue

        if t == "boolean":
            # Trattiamo i booleani come categoriali per contarne le frequenze (True/False)
            roles["categorical_columns"].append(col)
            continue

        if t == "text":
            roles["text_columns"].append(col)
            continue

        if t == "categorical":
            if nunique_ratio > 0.3:
                roles["lookup_columns"].append(col)
            else:
                roles["categorical_columns"].append(col)
            continue

        roles["unknown"].append(col)

    return roles


# =========================
# MAIN PIPELINE
# =========================
def analyze_file(path: str):

    df = load_file(path)

    if df.empty:
        return {"error": "empty dataset"}

    # TYPES
    types = infer_column_types(df)
    type_map = build_type_map(types)

    # ROLES
    roles = detect_feature_roles(df, type_map)

    # DATASET TYPE (FIX POSIZIONE)
    dataset_type = "mapping_table" if len(df.columns) == 2 else "general"

    # STATS
    stats = compute_statistics(df, roles)

    # PLOTS (ROLE-AWARE)
    if dataset_type == "mapping_table":
        plots = generate_mapping_plots(df, roles, output_dir="outputs/")
    else:
        plots = generate_plots(df, roles, output_dir="outputs/")

    # MAPPING QUALITY
    mapping_quality = {}
    if dataset_type == "mapping_table":
        id_col = roles.get("id_columns", [None])[0]
        name_col = roles.get("lookup_columns", [None])[0]

        if id_col and name_col:
            mapping_quality = {
                "duplicate_ids": int(df[id_col].duplicated().sum()),
                "duplicate_names": int(df[name_col].duplicated().sum()),
                "one_to_one": df[id_col].nunique() == df[name_col].nunique()
            }

    report = {
        "overview": {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "columns": list(df.columns)
        },
        "types": types,
        "roles": roles,
        "missing": df.isna().sum().to_dict(),
        "cardinality": df.nunique().to_dict(),
        "dataset_type": dataset_type,
        "mapping_quality": mapping_quality,
        "statistics": stats,
        "visualizations": plots
    }

    return {
        "report": report,
        "stats": stats,
        "plots": plots
    }