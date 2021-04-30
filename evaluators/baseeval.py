import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

class BaseEval:

  def __init__(self, settings): # **kwargs
    self._settings = settings

  def _compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    raise NotImplementedError()

  def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    return self._compute(real_data, synthetic_data)