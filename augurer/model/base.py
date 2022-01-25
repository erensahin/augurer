from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base forecaster class

    :param model_args: dictionary of arguments that will be passed to
        the estimator
    :type model_args: Dict
    """

    def __init__(self, model_args: Dict):
        self.model_args = model_args

    @abstractmethod
    def get_estimator(self, **kwargs: Dict[str, Any]):
        """
        Abstract base method that returns the estimator instance 
        """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs: Dict[str, Any]):
        """
        Abstract base method that trains the forecasting model.
        Should return the model instance

        :param train_df: train dataframe
        :type train_df: pd.DataFrame
        :param kwargs: additional arguments that will be passed to estimator
        :type kwargs: Dict[str, Any]
        :return: trained model instance
        :rtype: Any
        """

    @abstractmethod
    def make_test_dataframe(
        self,
        train_df: pd.DataFrame,
        period: str,
        horizon: int
    ) -> pd.DataFrame:
        """
        Abstract base method that generates a test dataset that will be
        forecasted.

        :param train_df: train dataframe
        :type train_df: pd.DataFrame
        :param period: periodicity of input data. should be "D" for day or
            "W" for week.
        :type period: str
        :param horizon: forecast horizon of future test data
        :type horizon: int
        :return: test dataframe
        :rtype: pd.DataFrame
        """

    @abstractmethod
    def predict(self, model, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract base method that predicts on the trained forecasting model.
        Returns a pd.DataFrame instance

        :param model: trained prophet model instance
        :type model: Any
        :param test_df: test dataframe
        :type test_df: pd.DataFrame
        :return: predictions
        :rtype: pd.DataFrame
        """

    @abstractmethod
    def fit_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        **kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Abstract base method that fits on a forecasting model, and then
        predicts on it.

        :param train_df: train dataframe
        :type train_df: pd.DataFrame
        :param test_df: test dataframe
        :type test_df: pd.DataFrame
        :param kwargs: additional arguments
        :type kwargs: Dict[str, Any]
        :return: prediction dataframe
        :rtype: pd.DataFrame
        """

    @abstractmethod
    def get_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract base method that decomposes forecasting output to
        only forecasts for future and backtest period

        :param prediction: prediction DataFrame
        :type prediction: pd.DataFrame
        :return: forecast DataFrame
        :rtype: pd.DataFrame
        """

    @abstractmethod
    def get_seasonality(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract base method that decomposes forecasting output to
        seasonality coefficients

        :param prediction: prediction DataFrame
        :type prediction: pd.DataFrame
        :return: seasonality DataFrame
        :rtype: pd.DataFrame
        """

    @staticmethod
    @abstractmethod
    def get_options() -> Dict[str, Any]:
        """
        Abstract base method that returns a dictionary of model parameters
        and their widget dictionaries that will be rendered

        :return: dictionary of model parameters and their widget
            dictionaries that will be rendered
        :rtype: Dict[str, Any]
        """
