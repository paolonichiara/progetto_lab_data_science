# =========================
# IO MODULE - DATA LOADING
# =========================

import pandas as pd
import re
import logging
import os

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =========================
# LOAD FILE (CSV + EXCEL)
# =========================
def load_file(path: str) -> pd.DataFrame:
    try:
        print(f"INFO: Caricamento file: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File non trovato: {path}")

        if path.endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.endswith(".csv"):
            # L'engine 'python' con sep=None permette a Pandas di auto-rilevare
            # se il file usa virgole (,) o punti e virgola (;) come separatore.
            df = pd.read_csv(path, sep=None, engine="python")
        else:
            raise ValueError(f"Formato non supportato: {path}")

        if df is None or df.empty:
            raise ValueError("Dataset vuoto")

        return df

    except Exception as e:
        print(f"ERROR: Errore caricamento dataset: {e}")
        raise