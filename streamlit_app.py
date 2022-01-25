from typing import Callable, Dict, Literal, Tuple, Type

import pandas as pd
import streamlit as st
from augurer.helpers import calculate_metrics

from augurer.model import MODELS
from augurer.model.base import BaseForecaster
from augurer.streamlit_utils.utils import decompose_data, list_datasets, read_data
from augurer.streamlit_utils.widget import InputWidgetOption

# Initial page config

st.set_page_config(
    page_title="augurer - Time Series Forecasting Experimentation Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> Tuple[Type[BaseForecaster], Dict]:
    """
    Renders the sidebar - Model and related model parameter widgets

    :return: model class and arguments which will be passed to model
    :rtype: Tuple[Type[BaseForecaster], Dict]
    """
    st.sidebar.markdown("**Models**")
    model_name = st.sidebar.selectbox(
        "Model", list(MODELS.keys()), key="model_name")
    model_klass = MODELS[model_name]

    widget_options = model_klass.get_options()
    model_args = {}
    no_return_keys = []
    for key, value in widget_options.items():
        widget_option = InputWidgetOption(**value)
        widget_value = widget_option.render(st.sidebar, **model_args)
        model_args[key] = widget_value
        if widget_option.no_return:
            no_return_keys.append(key)

    model_args = {
        k: v for k, v in model_args.items() if k not in no_return_keys}

    return model_klass, model_args


def render_header() -> Tuple[Dict, Callable]:
    """
    Renders the header:

        * dataset selectbox
        * period selectbox
        * horizon selectbox
        * run button
        * download button

    :return: header options and callback for rendering download button
    :rtype: Tuple[Dict, Callable]
    """
    with st.container():
        columns: list[st.container] = st.columns(5)

        # datasets
        datasets = list_datasets()
        with columns[0]:
            dataset = st.selectbox("Dataset", datasets, key="dataset")

        # period
        with columns[1]:
            period = st.selectbox("Period", ["W", "D"])
            default_horizon = 52 if period == "W" else 365
        with columns[2]:
            horizon = st.number_input(
                "horizon", value=default_horizon, min_value=1)

        # run button
        with columns[3]:
            st.markdown("Run now!")
            run_btn = st.button("Run!")

        def render_download_btn(fc):
            with columns[4]:
                download_data = fc.to_csv().encode("utf-8")
                model_name = st.session_state["model_name"]
                file_name = "_".join([
                    dataset.replace(".csv", ""), model_name, ".csv"])
                st.download_button(
                    label="Download",
                    data=download_data,
                    file_name=file_name,
                    mime='text/csv',
                )

    header_conf = {
        "dataset": dataset,
        "period": period,
        "horizon": horizon,
        "run_btn": run_btn
    }

    return header_conf, render_download_btn


def run_model(
    model: BaseForecaster,
    dataset: str,
    period: Literal["D", "W"],
    horizon: int
) -> pd.DataFrame:
    """
    Helper function to run a model on fit_predict mode for given dataset

    :param model: model instance
    :type model: BaseForecaster
    :param dataset: name of the selected dataset
    :type dataset: str
    :param period: periodicity of data. "D" for day, "W" for week
    :type period: Literal["D", "W"]
    :param horizon: forecast horizon length
    :type horizon: int
    :return: predictions
    :rtype: pd.DataFrame
    """
    data = read_data(dataset)
    train_df, holidays_df = decompose_data(data, period)
    test_df = model.make_test_dataframe(train_df, period, horizon)
    fc = model.fit_predict(train_df, test_df, period=period, horizon=horizon)
    return fc


def plot_forecasts(fc: pd.DataFrame) -> None:
    """
    Helper to plot forecast results

    :return: None
    :rtype: None
    """
    st.markdown("**Forecast vs Actual & Trend**")
    st.line_chart(fc[["prediction", "actual", "trend"]])


def plot_seasonality(fc):
    """
    Helper to plot seasonality results

    :return: None
    :rtype: None
    """
    st.markdown("**Seasonality Coefficients**")
    st.line_chart(fc)


def render_predictions(
        model: BaseForecaster, predictions: pd.DataFrame) -> None:
    """
    Helper function to render predictions

    :param model: model instance
    :type model: BaseForecaster
    :param predictions: prediction dataframe
    :type predictions: pd.DataFrame
    :return: None
    :rtype: None
    """
    forecast = model.get_forecast(predictions)
    seasonality = model.get_seasonality(predictions)

    # Render metrics
    metrics = calculate_metrics(forecast)
    columns: list[st.container] = st.columns(len(metrics))
    for i, (index, row) in enumerate(metrics.iterrows()):
        columns[i].metric(label=index, value=row["error"])

    plot_forecasts(forecast.set_index("ds"))
    plot_seasonality(seasonality.set_index("ds"))
    render_download_callback(predictions)


if __name__ == "__main__":
    model_klass, model_args = render_sidebar()
    header_config, render_download_callback = render_header()
    if header_config["run_btn"]:
        with st.spinner("Training model"):
            model: BaseForecaster = model_klass(model_args)
            predictions = run_model(
                model,
                header_config["dataset"],
                header_config["period"],
                header_config["horizon"]
            )
            # push predictions to state
            st.session_state["predictions"] = predictions
            st.session_state["model"] = model
            render_predictions(model, predictions)
    elif "predictions" in st.session_state:
        render_predictions(
            st.session_state["model"], st.session_state["predictions"])
