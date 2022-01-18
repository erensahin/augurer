from abc import ABC, abstractmethod
from typing import Any


class BaseForecaster(ABC):

    def __init__(self, model_args):
        self.model_args = model_args

    @abstractmethod
    def get_options(self):
        pass

    @abstractmethod
    def get_estimator(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, train_df, **kwargs):
        pass

    @abstractmethod
    def make_test_dataframe(self, train_df, period, horizon):
        pass

    @abstractmethod
    def predict(self, test_df):
        pass

    @abstractmethod
    def fit_predict(self, train_df, test_df, **kwargs):
        pass

    @abstractmethod
    def get_predictions(self, forecast):
        pass

    @abstractmethod
    def get_seasonality(self, forecast):
        pass