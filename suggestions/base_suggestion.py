import logging
import pandas as pd

from models import Training

LOGGER = logging.getLogger(__name__)

class BaseSuggestion:

  _name = None

  def __init__(self):
    self._name = 'Base'

  def get_suggestions(self, tables: dict, training: Training):
    return self._get_suggestions(tables, training)

  def _get_suggestions(self, tables: dict, training: Training):
    return NotImplementedError()