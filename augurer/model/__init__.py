from typing import Dict

from .base import BaseForecaster
from ._prophet import ProphetForecast

MODELS: Dict[str, BaseForecaster] = {
    "ProphetForecast": ProphetForecast
}
