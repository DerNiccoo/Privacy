import logging
import pandas as pd

from suggestions.base_suggestion import BaseSuggestion
from models import Training, Table

LOGGER = logging.getLogger(__name__)

class SDVSuggestion(BaseSuggestion):

  def __init__(self):
    super().__init__()
    self._name = 'SDV'

  def _category_check(self, table_name: str, df: pd.DataFrame, table_meta: Table):
    suggestions = []

    for attribute in table_meta.attributes:
      uniques = len(df[attribute.name].unique())
      if attribute.dtype == 'categorical':
        if uniques / len(df) > 0.03: #Wenn es viele davon gibt, besser label_encoding da besser für performance
          suggestions.append({'table': table_name, 'attribute': attribute.name, 'category': 'Transformer', 'solution': 'label_encoding'})
      elif attribute.dtype == 'numerical':
        if uniques / len(df) < 0.6:
          suggestions.append({'table': table_name, 'attribute': attribute.name, 'category': 'Datatype', 'solution': 'categorical'})

          if uniques / len(df) > 0.03:
            suggestions.append({'table': table_name, 'attribute': attribute.name, 'category': 'Transformer', 'solution': 'label_encoding'})

    return suggestions

  def _check_int(self, attr):
    if str(attr).endswith('.0'):
      return True
    else:
      return False

  def _float_check(self, table_name: str, df: pd.DataFrame, table_meta: Table):
    suggestions = []

    for col in df.columns.tolist():
      if df[col].dtypes == 'float64':
        uniques = df[col].apply(lambda x: self._check_int(x)).unique()
        if len(uniques) == 1:
          if uniques[0] == True:
            suggestions.append({'table': table_name, 'attribute': col, 'category': 'Transformer', 'solution': 'integer'})

    return suggestions

  def _column_name_check(self, table_name, df):
    suggestions = []

    for col in df.columns.tolist():
      if 'id' in col.lower():
        if len(df[col].unique()) == len(df): # Nur wenn die auch wirklich in den OG Datentyp eindeutig sind, handelt es sich wohl um eine ID
          suggestions.append({'table': table_name, 'attribute': col, 'category': 'Datatype', 'solution': 'ID'})

    return suggestions

  def _get_suggestions(self, tables: dict, training: Training):
    suggestions = []

    for (table_name, df), table_meta in zip(tables.items(), training.tables):
      if table_name is not table_meta.name:
        LOGGER.warning('Missmatch with table names in SDV Suggestions')
        return []

      suggs = self._category_check(table_name, df, table_meta)

      if suggs:
        suggestions.extend(suggs)

      suggs = self._float_check(table_name, df, table_meta)

      if suggs:
        suggestions.extend(suggs)

      suggs = self._column_name_check(table_name, df)

      if suggs:
        suggestions.extend(suggs)

    return suggestions