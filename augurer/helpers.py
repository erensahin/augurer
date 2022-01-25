"""
Helper functions
"""

import pandas as pd
import numpy as np


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forecast error metrics
    """
    df = df[~df["actual"].isnull()]
    df["error"] = df["actual"] - df["prediction"]
    df["abs_err"] = np.abs(df["error"])
    df["rel_err"] = df["abs_err"] / df["actual"]
    df["squared_error"] = df["error"] ** 2

    mae = df["abs_err"].mean()
    mdae = df["abs_err"].median()
    rmse = np.sqrt(df["squared_error"].mean())

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    mdape = df["rel_err"].median() * 100
    mape = df["rel_err"].mean() * 100

    return pd.DataFrame(
        [mae, mdae, mape, mdape, rmse],
        index=["MAE", "MdAE", "MAPE", "MdAPE", "RMSE"],
        columns=["error"]
    ).round(1)
