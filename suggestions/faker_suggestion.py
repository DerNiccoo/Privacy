import logging
import pandas as pd

from suggestions.base_suggestion import BaseSuggestion
from models import Training

LOGGER = logging.getLogger(__name__)

class FakerSuggestion(BaseSuggestion):

  _faker_column_names = ['name', 'postcode', 'job']

  def __init__(self):
    super().__init__()
    self._name = 'Faker'

  def _column_check(self, table_name, column_list):
    suggestions = []

    for col in column_list:
      for faker_name in self._faker_column_names:
        if faker_name in col.lower():
          suggestions.append({'table': table_name, 'attribute': col, 'category': 'Faker', 'solution': faker_name})

    return suggestions

  def _get_suggestions(self, tables: dict, training: Training):
    suggestions = []

    for table_name, df in tables.items():
      suggs = self._column_check(table_name, df.columns.tolist())

      if suggs:
        suggestions.extend(suggs)

    return suggestions