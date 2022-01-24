"""
Helper functions
"""
import os
from typing import List, Literal, Tuple

import pandas as pd
import streamlit as st

DS_FOLDER = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "datasets")


def list_datasets() -> List[str]:
    """
    List CSV datasets in the dataset folder.

    :return: list of CSV file names
    :rtype: List[str]
    """
    return [f for f in os.listdir(DS_FOLDER) if f.endswith(".csv")]


@st.cache(allow_output_mutation=True)
def read_data(dataset: str) -> pd.DataFrame:
    """
    Reads any desired dataset

    :param dataset: name of the file to read
    :type dataset: str
    :return: dataframe which have been read
    :rtype: pd.DataFrame
    """
    path = os.path.join(DS_FOLDER, dataset)
    return pd.read_csv(path)


@st.cache(allow_output_mutation=True)
def decompose_data(
    df: pd.DataFrame,
    period: Literal["D", "W"] = "D"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper method to decompose input data to time-series targets to train
    and holidays

    :param df: dataframe to decompose
    :type df: pd.DataFrame
    :param period: periodicity of the data. "D" for day, "W" for week
    :type period: Literal["D", "W"]
    :return: tuple of train data and holiday data
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
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
