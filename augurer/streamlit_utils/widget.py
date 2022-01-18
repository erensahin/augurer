from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Type

from streamlit.delta_generator import DeltaGenerator


@dataclass
class InputWidgetOption:
    """Class for keeping information of an input widget"""

    label: str
    value: Optional[Union[List[Any], Any]] = None
    is_text_input: bool = False
    data_type: Optional[Type] = None
    prereq: Optional[Callable] = None
    min_value: float = None
    max_value: float = None
    step: float = None
    no_return: bool = False

    def render(self, st_container: DeltaGenerator, **kwargs):
        """
        Helper method to render a widget
        """
        if self.prereq:
            condition, return_value = self.prereq(kwargs)
            if not condition:
                return return_value

        if isinstance(self.value, list):
            return st_container.selectbox(
                label=self.label,
                options=self.value
            )
        elif self.is_text_input:
            number_input = st_container.number_input(
                label=self.label,
                min_value=self.min_value,
                max_value=self.max_value,
                value=self.value,
                step=self.step,
            )
            return (
                number_input if self.data_type is None
                else self.data_type(number_input)
            )
        else:
            raise ValueError(f"Cant render {self.label}")
