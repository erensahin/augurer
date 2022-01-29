"""
Helper functions
"""
import os
import re
from typing import List, Optional, Tuple

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


def get_uploaded_datasets() -> List[str]:
    """
    Returns a list of uploaded datasets from session state

    :return: list of uploaded file names
    :rtype: List[str]
    """
    if "uploaded_data" in st.session_state:
        return list(st.session_state["uploaded_data"])
    return []


def get_uploaded_data(dataset: str) -> Optional[pd.DataFrame]:
    """
    Returns content of the dataset if it exist in uploaded data.
    Otherwise, None

    :param dataset: name of the uploaded file
    :type dataset: str
    :return: dataframe which have been read
    :rtype: Optional[pd.DataFrame]
    """
    if dataset in get_uploaded_datasets():
        return st.session_state["uploaded_data"][dataset]
    return None


def put_uploaded_data_in_state(name: str, df: pd.DataFrame) -> None:
    """
    Puts uploaded data in session state.

    :param name: name of the dataset
    :type name: str
    :param df: uploaded and validated data
    :type df: pd.DataFrame
    :return: None
    :rtype: None
    """
    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = {}
    st.session_state["uploaded_data"][name] = df


def validate_uploaded_data(df: pd.DataFrame) -> bool:
    """
    Validates uploaded data to check whether it has necessary
    columns or not. Necessary columns are:

        * ds
        * yearweek
        * y
        * holiday

    :return: True if input dataframe is valid. Otherwise, AssertionError
        is raises.
    :rtype: bool
    :raies: AssertionError
    """
    expected_cols = {"ds", "yearweek", "y", "holiday"}
    assert expected_cols.issubset(df.columns.tolist()), (
        "Following columns are expected in the uploaded data: {}"
        "Uploaded data columns: {}"
    ).format(list(expected_cols), df.columns.tolist())

    return True


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
def get_data(dataset: str) -> pd.DataFrame:
    """
    Returns corresponding dataframe for the desired dataset.
    If the dataset is in session_state, it is returned.
    Otherwise, it is read.

    :param dataset: name of the dataset
    :type dataset: str
    :return: dataframe which have been read
    :rtype: pd.DataFrame
    """
    if dataset in get_uploaded_datasets():
        return get_uploaded_data(dataset)
    return read_data(dataset)


@st.cache(allow_output_mutation=True)
def decompose_data(
    df: pd.DataFrame,
    period: str = "D"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper method to decompose input data to time-series targets to train
    and holidays

    :param df: dataframe to decompose
    :type df: pd.DataFrame
    :param period: periodicity of the data. "D" for day, "W" for week
    :type period: str
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
