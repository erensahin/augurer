import pandas as pd
import streamlit as st

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


def render_sidebar():
    st.sidebar.markdown("**Models**")
    model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
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


def render_header():
    with st.container():
        columns: list[st.container] = st.columns(5)

        # datasets
        datasets = list_datasets()
        with columns[0]:
            dataset = st.selectbox("Dataset", datasets)

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
                file_name = dataset.replace(".csv", "") + "_forecast.csv"
                st.download_button(
                    label="Download Forecasts",
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
        dataset: pd.DataFrame,
        period: int,
        horizon: int
):
    data = read_data(dataset)
    train_df, holidays_df = decompose_data(data, period)
    test_df = model.make_test_dataframe(train_df, period, horizon)
    fc = model.fit_predict(train_df, test_df, period=period, horizon=horizon)
    return fc


def plot_forecasts(fc):
    st.markdown("**Forecast vs Actual & Trend**")
    st.line_chart(fc[["prediction", "actual", "trend"]])


def plot_seasonality(fc):
    st.markdown("**Seasonality Coefficients**")
    st.line_chart(fc)


model_klass, model_args = render_sidebar()
header_config, render_download_callback = render_header()
if header_config["run_btn"]:
    with st.spinner("Training model"):
        model: BaseForecaster = model_klass(model_args)
        forecast = run_model(
            model,
            header_config["dataset"],
            header_config["period"],
            header_config["horizon"]
        )
        predictions = model.get_predictions(forecast)
        seasonality = model.get_seasonality(forecast)
        plot_forecasts(predictions.set_index("ds"))
        plot_seasonality(seasonality.set_index("ds"))
