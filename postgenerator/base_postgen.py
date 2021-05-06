import logging
import pandas as pd

from models import Training

LOGGER = logging.getLogger(__name__)

class BasePostGen:
  _name = None

  def __init__(self):
    self._name = 'Base'

  def apply_post_process(self, df_real: pd.DataFrame, df_fake: pd.DataFrame, training: Training, table_name: str):
    return self._apply_post_process(df_real, df_fake, training, table_name)

  def _apply_post_process(self, df_real: pd.DataFrame, df_fake: pd.DataFrame, training: Training, table_name: str):
    return NotImplementedError()

  def _get_columns_without_faker(self, training: Training, table_name: str):
    col_names = []
    
    for table in training.tables:
      if table.name == table_name:
        for attr in table.attributes:
          if attr.field_anonymize == None:
            col_names.append(attr.name)

    return col_names