from typing import Any, Dict, Literal

import pandas as pd
from prophet import Prophet

from .base import BaseForecaster


class ProphetForecast(BaseForecaster):
    """
    Prophet model wrapper class

    :param model_args: model arguments
    :type model_args: Dict[str, Any]
    """

    def __init__(self, model_args: Dict[str, Any]):
        super().__init__(model_args)
        self.model = None

    def get_estimator(self, **kwargs: Dict[str, Any]) -> Prophet:
        """
        :return: Prophet model instance
        :rtype: Prophet
        """
        args = self.model_args.copy()
        args.update(kwargs)
        return Prophet(**args)

    def fit(self, train_df: pd.DataFrame, **kwargs: Dict[str, Any]) -> Prophet:
        """
        Trains a prophet model and returns trained model instance

        :param train_df: train dataframe
        :type train_df: pd.DataFrame
        :param kwargs: additional arguments that will be passed to estimator
        :type kwargs: Dict[str, Any]
        :return: trained model instance
        :rtype: Prophet
        """
        model = self.get_estimator(**kwargs)
        model.fit(train_df)
        self.model = model
        return model

    def make_test_dataframe(
        self,
        train_df: pd.DataFrame,
        period: Literal["D", "W"],
        horizon: int
    ) -> pd.DataFrame:
        """
        Creates a test dataframe to forecast on.

        :param train_df: train dataframe
        :type train_df: pd.DataFrame
        :param period: periodicity of input data. should be "D" for day or
            "W" for week.
        :type period: Literal["D", "W"]
        :param horizon: forecast horizon of future test data
        :type horizon: int
        :return: test dataframe
        :rtype: pd.DataFrame
        """
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

    def predict(self, model: Prophet, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts on a trained model

        :param model: trained prophet model instance
        :type model: Prophet
        :param test_df: test dataframe
        :type test_df: pd.DataFrame
        :return: predictions
        :rtype: pd.DataFrame
        """
        forecast = model.predict(test_df)
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

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        **kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Trains a prophet model and returns trained model instance

        :param train_df: train dataframe
        :type train_df: pd.DataFrame
        :param test_df: test dataframe
        :type test_df: pd.DataFrame
        :param kwargs: additional arguments that will be passed to estimator
            or will be used as helpers to create test dataframe:
            `horizon` and `period`
        :type kwargs: Dict[str, Any]
        :return: prediction dataframe
        :rtype: pd.DataFrame
        """
        horizon = kwargs.pop("horizon", None)
        period = kwargs.pop("period", None)
        model = self.fit(train_df, **kwargs)
        if test_df is None:
            test_df = model.make_future_dataframe(horizon, freq=period)
        return self.predict(model, test_df)

    def get_forecast(self, prediction: pd.DataFrame) -> pd.DataFrame:
        """
        Returns forecast related information from prediction output.

        :param prediction: prediction DataFrame
        :type prediction: pd.DataFrame
        :return: forecast DataFrame
        :rtype: pd.DataFrame
        """
        columns = [
            "actual",
            "prediction",
            "prediction_upper",
            "prediction_lower",
            "trend"
        ]
        columns = [c for c in columns if c in prediction.columns]
        return prediction[["ds", *columns]]

    def get_seasonality(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Returns seasonality related information from prediction output.

        :param prediction: prediction DataFrame
        :type prediction: pd.DataFrame
        :return: seasonality DataFrame
        :rtype: pd.DataFrame
        """
        columns = [
            "yearly",
            "weekly",
            "daily",
            "holidays"
        ]
        columns = [c for c in columns if c in forecast.columns]
        return forecast[["ds", *columns]]

    @staticmethod
    def get_options() -> Dict[str, Any]:
        """
        :return: dictionary of model parameters and their widget
            dictionaries that will be rendered
        :rtype: Dict[str, Any]
        """
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
