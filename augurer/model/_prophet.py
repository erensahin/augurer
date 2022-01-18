import pandas as pd
import numpy as np

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

from .base import BaseForecaster


class ProphetForecast(BaseForecaster):

    def __init__(self, model_args):
        super().__init__(model_args)
        self.model = None

    def get_estimator(self, **kwargs):
        args = self.model_args.copy()
        args.update(kwargs)
        return Prophet(**args)

    def fit(self, train_df, **kwargs):
        model = self.get_estimator(**kwargs)
        model.fit(train_df)
        self.model = model
        return model

    def make_test_dataframe(self, train_df, period, horizon):
        if self.model:
            test_df = self.model.make_future_dataframe(horizon, freq=period)
        else:
            last_date = train_df["ds"].max()
            dates = pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq=period)
            dates = dates[dates > last_date]
            dates = pd.DataFrame(dates[:horizon], columns=["ds"])
            dates["ds"] = dates["ds"].astype(str)
            test_df = pd.concat([train_df[["ds"]], dates[["ds"]]])

        return pd.merge(
            test_df["ds"], train_df[["ds", "y"]], on="ds", how="left")

    def predict(self, test_df):
        forecast = self.model.predict(test_df)
        forecast["ds"] = forecast["ds"].astype(str)
        forecast = pd.merge(
            forecast, test_df[["ds", "y"]], on="ds", how="left")

        columns = [
            "yhat",
            "yhat_upper",
            "yhat_lower",
            "trend",
            "yearly",
            "weekly",
            "daily",
            "holidays"
        ]
        columns = [c for c in columns if c in forecast.columns]

        forecast = forecast[["ds", "y", *columns]]
        return forecast.rename(columns={
            "y": "actual",
            "yhat": "prediction",
            "yhat_upper": "prediction_upper",
            "yhat_lower": "prediction_lower"
        })

    def fit_predict(self, train_df, test_df, **kwargs):
        horizon = kwargs.pop("horizon", None)
        period = kwargs.pop("period", None)
        model = self.fit(train_df, **kwargs)
        if test_df is None:
            test_df = model.make_future_dataframe(horizon, freq=period)
        return self.predict(test_df)

    def get_predictions(self, forecast):
        columns = [
            "actual",
            "prediction",
            "prediction_upper",
            "prediction_lower",
            "trend"
        ]
        columns = [c for c in columns if c in forecast.columns]
        return forecast[["ds", *columns]]

    def get_seasonality(self, forecast):
        columns = [
            "yearly",
            "weekly",
            "daily",
            "holidays"
        ]
        columns = [c for c in columns if c in forecast.columns]
        return forecast[["ds", *columns]]

    @staticmethod
    def get_options():
        return {
            "seasonality_mode": {
                "label": "seasonality mode",
                "value": ["multiplicative", "additive"],
            },
            "yearly_seasonality_type": {
                "label": "yearly seasonality type",
                "value": ["fourier", "auto", True, False],
                "no_return": True
            },
            "yearly_seasonality": {
                "label": "yearly seasonality",
                "value": 5,
                "is_text_input": True,
                "data_type": int,
                "prereq": lambda opts: (opts["yearly_seasonality_type"] == "fourier", False)
            },
            "weekly_seasonality_type": {
                "label": "weekly seasonality type",
                "value": [False, True, "fourier", "auto"],
                "no_return": True
            },
            "weekly_seasonality": {
                "label": "weekly seasonality",
                "value": 5,
                "is_text_input": True,
                "data_type": int,
                "prereq": lambda opts: (opts["weekly_seasonality_type"] == "fourier", False)
            },
            "daily_seasonality_type": {
                "label": "daily seasonality type",
                "value": [False, True, "fourier", "auto"],
                "no_return": True
            },
            "daily_seasonality": {
                "label": "daily seasonality",
                "value": 5,
                "is_text_input": True,
                "data_type": int,
                "prereq": lambda opts: (opts["daily_seasonality_type"] == "fourier", False)
            },
            "growth": {
                "label": "growth type",
                "value": ["linear", "flat"]
            },
            "n_changepoints": {
                "label": "n_changepoints",
                "value": 25,
                "is_text_input": True,
                "data_type": int
            },
            "seasonality_prior_scale": {
                "label": "seasonality_prior_scale",
                "min_value": 0.0,
                "max_value": 1000.0,
                "value": 0.05,
                "step": 0.05,
                "is_text_input": True,
                "data_type": float
            },
            "changepoint_prior_scale": {
                "label": "changepoint_prior_scale",
                "min_value": 0.0,
                "max_value": 1000.0,
                "value": 0.005,
                "step": 0.05,
                "is_text_input": True,
                "data_type": float
            },
            "holidays_prior_scale": {
                "label": "holidays_prior_scale",
                "min_value": 0.0,
                "max_value": 1000.0,
                "value": 10.0,
                "step": 0.50,
                "is_text_input": True,
                "data_type": float
            },
            "mcmc_samples": {
                "label": "mcmc_samples",
                "value": 0,
                "is_text_input": True,
                "data_type": float
            },
            "interval_width": {
                "label": "interval_width",
                "value": 0.80,
                "is_text_input": True,
                "data_type": float
            },
            "uncertainty_samples": {
                "label": "uncertainty_samples",
                "value": 0,
                "is_text_input": True,
                "data_type": float
            },
        }
