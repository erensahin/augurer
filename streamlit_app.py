from src.model import fit_predict
from src.utils import list_datasets, read_data, decompose_data

import pandas as pd
import streamlit as st


# Initial page config

st.set_page_config(
    page_title="Prophet experimentation tool",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar():
    st.sidebar.markdown("**Prophet Parameters**")

    # seasonality_mode
    seasonality_mode = st.sidebar.radio(
        "seasonality_mode", ["multiplicative", "additive"])

    # growth
    growth = st.sidebar.selectbox("Growth", ["linear", "logistic", "flat"])
    cap = None
    if growth == "logistic":
        cap = int(st.sidebar.number_input("Logistic Growth Capacity"))

    # n_changepoints
    st.sidebar.markdown("n_changepoints")
    n_changepoints = int(st.sidebar.number_input("n_changepoints", value=25))

    # yearly seasonality
    st.sidebar.markdown("yearly_seasonality")

    if st.sidebar.checkbox("Use fourier yearly_seasonality", True):
        yearly_seasonality = st.sidebar.number_input(
            "yearly_seasonality", value=5)
    else:
        yearly_seasonality = st.sidebar.radio(
            "yearly_seasonality", ["auto", True, False])

    # weekly seasonality
    if st.sidebar.checkbox("Use fourier weekly_seasonality"):
        weekly_seasonality = st.sidebar.number_input(
            "weekly_seasonality", value=5)
    else:
        weekly_seasonality = st.sidebar.radio(
            "weekly_seasonality", ["auto", True, False])

    # daily seasonality
    if st.sidebar.checkbox("Use fourier daily_seasonality"):
        daily_seasonality = st.sidebar.number_input(
            "daily_seasonality", value=5)
    else:
        daily_seasonality = st.sidebar.radio(
            "daily_seasonality", ["auto", True, False])

    # seasonality prior scale
    st.sidebar.markdown("seasonality_prior_scale")
    seasonality_prior_scale = st.sidebar.number_input(
        "seasonality_prior_scale",
        min_value=float(0),
        max_value=float(1000),
        value=0.05,
        step=0.05
    )

    # holidays prior scale
    st.sidebar.markdown("holidays_prior_scale")
    holidays_prior_scale = st.sidebar.number_input(
        "holidays_prior_scale",
        min_value=float(0),
        max_value=float(1000),
        value=10.0,
        step=0.50
    )

    # changepoint prior scale
    st.sidebar.markdown("changepoint_prior_scale")
    changepoint_prior_scale = st.sidebar.number_input(
        "changepoint_prior_scale",
        min_value=float(0),
        max_value=float(1000),
        value=0.005,
        step=0.05
    )

    # mcmc samples
    st.sidebar.markdown("mcmc_samples")
    mcmc_samples = st.sidebar.number_input("mcmc_samples", value=0)

    # interval_width
    st.sidebar.markdown("interval width")
    interval_width = st.sidebar.number_input("interval_width", value=0.80)

    # uncertainty_samples
    st.sidebar.markdown("uncertainty_samples")
    uncertainty_samples = st.sidebar.number_input(
        "uncertainty_samples", value=1000)

    # parameter configuration

    sidebar_conf = {
        "prophet_args": {
            "seasonality_mode": seasonality_mode,
            "growth": growth,
            "n_changepoints": n_changepoints,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "changepoint_prior_scale": changepoint_prior_scale,
            "mcmc_samples": mcmc_samples,
            "interval_width": interval_width,
            "uncertainty_samples": uncertainty_samples
        },
        "logistic_cap": cap
    }
    return sidebar_conf


def render_page(forecast: pd.DataFrame, plot, comp_plot):
    fc = forecast.set_index("ds")
    plot_forecasts(fc)
    plot_coefficients(fc)
    plot_prophet_components(plot, comp_plot)


def plot_forecasts(fc):
    st.markdown("**Forecast vs Actual & Trend**")
    st.line_chart(fc[["y", "yhat", "trend"]])


def plot_coefficients(fc):
    st.markdown("**Seasonality Coefficients**")
    st.line_chart(fc[["yearly", "holidays"]])


def plot_prophet_components(plot, comp_plot):
    st.markdown("**Prophet Components**")
    st.container()
    columns: list[st.container] = st.columns(2)
    columns[0].markdown("Forecast vs Actual")
    columns[0].plotly_chart(plot, use_container_width=True)

    columns[1].markdown("Seasonal Components")
    columns[1].plotly_chart(comp_plot, use_container_width=True)


def run_model(dataset, period, horizon, prophet_args, normalize_coeffs, cap=None):
    data = read_data(dataset)
    train, holidays = decompose_data(data, period)
    forecast, plot, comp_plot = fit_predict(
        train, holidays, prophet_args, period=period, horizon=horizon, cap=cap)
    if normalize_coeffs:
        for col in ["yearly", "weekly", "daily", "holidays"]:
            if col in forecast.columns:
                forecast[col] = forecast[col] + 1
    return forecast, plot, comp_plot


def render_header():
    with st.container():
        columns: list[st.container] = st.columns(6)

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

        # normalize coeffs
        with columns[3]:
            st.markdown("Normalize Coefficients")
            normalize_coeffs = st.checkbox("Normalize Coefficients", True)

        with columns[4]:
            run_btn = st.button("Run!")

        def render_download_btn(fc):
            with columns[5]:
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
        "normalize_coeffs": normalize_coeffs,
        "run_btn": run_btn
    }

    return header_conf, render_download_btn


header_conf, render_download_callback = render_header()
sidebar_conf = render_sidebar()
if header_conf["run_btn"]:
    with st.spinner("Training prophet model"):
        forecast, plot, comp_plot = run_model(
            header_conf["dataset"],
            header_conf["period"],
            header_conf["horizon"],
            sidebar_conf["prophet_args"],
            header_conf["normalize_coeffs"],
            cap=sidebar_conf["logistic_cap"]
        )
        render_download_callback(forecast)
    render_page(forecast, plot, comp_plot)
