import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# =========================
# UTILITY
# =========================
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================
# ENTRY POINT
# =========================
def generate_plots(df, roles, output_dir="outputs/"):

    _ensure_dir(output_dir)

    plots_info = []

    numeric_cols = roles.get("numerical_columns", [])
    categorical_cols = roles.get("categorical_columns", []) + roles.get("lookup_columns", [])

    # LIMIT SAFETY (evita esplosione grafici)
    numeric_cols = numeric_cols[:10]
    categorical_cols = categorical_cols[:10]

    # =========================
    # NUMERIC PLOTS
    # =========================
    for col in numeric_cols:

        if col in roles.get("id_columns", []):
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        # -------------------------
        # HISTOGRAM (PLOTLY)
        # -------------------------
        fig = px.histogram(df, x=col, title=f"Distribuzione: {col}")
        file_html = os.path.join(output_dir, f"{col}_hist.html")
        fig.write_html(file_html)

        plots_info.append({
            "column": col,
            "type": "histogram",
            "file": file_html
        })

        # -------------------------
        # BOXPLOT
        # -------------------------
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=s)
        sns.swarmplot(x=s, color=".25", alpha=0.5)
        plt.title(f"Boxplot: {col}")

        file_png = os.path.join(output_dir, f"{col}_box.png")
        plt.savefig(file_png)
        plt.close()

        plots_info.append({
            "column": col,
            "type": "boxplot",
            "file": file_png
        })

        # -------------------------
        # ECDF
        # -------------------------
        sorted_vals = np.sort(s)
        ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        plt.figure(figsize=(6, 4))
        plt.step(sorted_vals, ecdf)
        plt.title(f"ECDF: {col}")
        plt.xlabel(col)
        plt.ylabel("ECDF")

        file_png = os.path.join(output_dir, f"{col}_ecdf.png")
        plt.savefig(file_png)
        plt.close()

        plots_info.append({
            "column": col,
            "type": "ecdf",
            "file": file_png
        })

    # =========================
    # CATEGORICAL PLOTS
    # =========================
    for col in categorical_cols:

        if col in roles.get("id_columns", []):
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        top_k = s.value_counts().head(20)

        plt.figure(figsize=(6, 4))
        top_k.plot(kind="barh")
        plt.title(f"Frequenze: {col}")

        file_png = os.path.join(output_dir, f"{col}_count.png")
        plt.savefig(file_png)
        plt.close()

        plots_info.append({
            "column": col,
            "type": "countplot",
            "file": file_png
        })

    # =========================
    # NUMERIC vs CATEGORICAL
    # =========================
    for num_col in numeric_cols:
        for cat_col in categorical_cols:

            if df[cat_col].nunique() > 20:
                continue

            if num_col in roles.get("id_columns", []):
                continue

            plt.figure(figsize=(6, 4))

            sns.boxplot(x=df[cat_col], y=df[num_col])
            sns.swarmplot(x=df[cat_col], y=df[num_col], color=".25", alpha=0.4)

            plt.title(f"{num_col} vs {cat_col}")

            file_png = os.path.join(
                output_dir,
                f"{num_col}_by_{cat_col}.png"
            )

            plt.savefig(file_png)
            plt.close()

            plots_info.append({
                "column": f"{num_col} vs {cat_col}",
                "type": "box_swarm",
                "file": file_png
            })

    # =========================
    # MULTIVARIATE NUMERIC
    # =========================
    if len(numeric_cols) >= 2:

        # -------------------------
        # PAIRPLOT
        # -------------------------
        pairplot_file = os.path.join(output_dir, "pairplot.png")

        g = sns.pairplot(df[numeric_cols].dropna())
        g.savefig(pairplot_file)
        plt.close()

        plots_info.append({
            "column": "numeric_pairplot",
            "type": "pairplot",
            "file": pairplot_file
        })

        # -------------------------
        # HEATMAP CORRELATION
        # -------------------------
        plt.figure(figsize=(8, 6))

        corr = df[numeric_cols].corr(method="pearson")

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")

        plt.title("Correlazione Pearson")

        file_png = os.path.join(output_dir, "heatmap.png")
        plt.savefig(file_png)
        plt.close()

        plots_info.append({
            "column": "numeric_correlation",
            "type": "heatmap",
            "file": file_png
        })

    # =========================
    # MULTIVARIATE CATEGORICAL
    # =========================
    if len(categorical_cols) >= 2:

        cols = categorical_cols[:5]

        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:

                ct = pd.crosstab(df[c1], df[c2])

                ct.plot(kind="bar", stacked=True, figsize=(6, 4))
                plt.title(f"{c1} vs {c2}")

                file_png = os.path.join(
                    output_dir,
                    f"{c1}_vs_{c2}_stacked.png"
                )

                plt.savefig(file_png)
                plt.close()

                plots_info.append({
                    "column": f"{c1} vs {c2}",
                    "type": "stacked_bar",
                    "file": file_png
                })

    return plots_info


# =========================
# SINGLE COLUMN PLOT (LLM TOOL)
# =========================
def plot_column(df: pd.DataFrame, column: str) -> dict:

    if column not in df.columns:
        return {"error": f"Colonna '{column}' non trovata"}

    s = df[column].dropna()

    plt.figure(figsize=(6, 4))

    if pd.api.types.is_numeric_dtype(s):
        sns.histplot(s, kde=True)
        file_name = f"{column}_hist.png"
    else:
        sns.countplot(y=s, order=s.value_counts().index)
        file_name = f"{column}_count.png"

    plt.title(column)
    plt.savefig(file_name)
    plt.close()

    return {
        "column": column,
        "file": file_name
    }


# =========================
# MAPPING VISUALIZATION
# =========================
def generate_mapping_plots(df, roles, output_dir="outputs/"):

    _ensure_dir(output_dir)

    plots = []

    label_col = roles.get("lookup_columns", [None])[0]

    if label_col is None:
        return plots

    s = df[label_col]

    plt.figure(figsize=(6, 4))
    s.value_counts().head(20).plot(kind="barh")
    plt.title(f"Mapping: {label_col}")

    file_path = os.path.join(output_dir, f"{label_col}_mapping.png")
    plt.savefig(file_path)
    plt.close()

    plots.append({
        "column": label_col,
        "type": "mapping_countplot",
        "file": file_path
    })

    return plots