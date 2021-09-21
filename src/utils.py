"""
Helper functions
"""
import os

import pandas as pd

DS_FOLDER = "./datasets"


def list_datasets():
    return [f for f in os.listdir(DS_FOLDER) if f.endswith(".csv")]


def read_data(dataset):
    path = os.path.join(DS_FOLDER, dataset)
    return pd.read_csv(path)


def decompose_data(df, period="D"):
    train_data = df[["yearweek", "ds", "y"]]
    train_data = train_data[train_data["y"].notna()]

    holidays = df[["yearweek", "ds", "holiday"]]
    holidays["holiday"] = holidays["holiday"].str.split("|")
    holidays = holidays.explode("holiday")
    holidays = holidays[holidays["holiday"].notna()]

    if period == "W":
        train_data = train_data.groupby("yearweek").agg({
            "ds": "first",
            "y": "sum"
        })
        holidays = holidays.groupby("yearweek").agg({
            "ds": "first",
            "holiday": list
        })
        holidays = holidays.explode("holiday")

    return train_data[["ds", "y"]], holidays[["ds", "holiday"]]
