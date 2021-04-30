
from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE
from sdv.relational import HMA1
from sdv import Metadata

from models import Training

import pandas as pd
import logging


LOGGER = logging.getLogger(__name__)

class Generator:

  _model = None
  _training = None
  _performance_mode = False

  def __init__(self, training: Training, metadata: Metadata):
    self._training = training

    distribution, transformer, anonymize = self._get_settings()

    if len(self._training.tables) > 1:
      LOGGER.warning(f"Using HMA1")
      self._model = HMA1(metadata)
    else:
      if training.tables[0].model == "GaussianCopula":
        self._model = GaussianCopula(anonymize_fields = anonymize, field_transformers = transformer, field_distributions = distribution)
      elif training.tables[0].model == "CTGAN": 
        self._model = CTGAN(anonymize_fields = anonymize, field_transformers = transformer)
      elif training.tables[0].model == "CopulaGAN": 
        self._model = CopulaGAN(anonymize_fields = anonymize, field_transformers = transformer, field_distributions = distribution)
      elif training.tables[0].model == "TVAE": 
        self._model = TVAE(anonymize_fields = anonymize, field_transformers = transformer)

    LOGGER.warning(f'Using Model: {self._model}')

  def fit(self, tables: dict):
    data = tables

    if len(tables) == 1:
      data = tables[list(tables.keys())[0]] #Da nur eine Tabelle kein dict von Tabellen übergeben
      if self._performance_mode:
        data = data.sample(n=100)
    else:
      #Reduction in dataset only. Debugging only. Hardcodeing for given Dataset:
      data['player'] = tables['player'].sample(n=10)
      boolean_series = tables['player_attributes'].player_api_id.isin(data['player'].player_api_id)
      data['player_attributes'] = tables['player_attributes'][boolean_series]

      LOGGER.warning(f"For debugging reason only allow training with 100 or less Datapoints")
      LOGGER.warning(f"Started fitting with {len(data)} data points")

    self._model.fit(data)

  def sample(self, count: int):
    LOGGER.warning("start sampling of data")
    return self._model.sample(num_rows = count)

  def save(self, tables):
    path_split = self._training.path.split('/')
    new_path = "/".join(path_split[:-1])

    if isinstance(tables, pd.DataFrame):
      t_name = path_split[-1].split(".")[0]
      tables.to_csv(path_or_buf=new_path + "/" + t_name + "_gen" + ".csv", index=False)
    else:
      for t_name, t_value in tables.items():
        t_value.to_csv(path_or_buf=new_path + "/" + t_name + "_gen" + ".csv", index=False)

    return None

  def _get_settings(self):
    table_distribution = {}
    table_transformer = {}
    table_anonymize = {}

    for table in self._training.tables:
      field_distribution = {}
      field_transformer = {}
      field_anonymize = {}

      for attr in table.attributes:
        if attr.field_anonymize is not None:
          field_anonymize[attr.name] = attr.field_anonymize

        if attr.field_transformer is not None:
          field_transformer[attr.name] = attr.field_transformer

        if attr.field_distribution is not None:
          field_distribution[attr.name] = attr.field_distribution

      table_distribution[table.name] = field_distribution
      table_transformer[table.name] = field_transformer
      table_anonymize[table.name] = field_anonymize

    LOGGER.warning(f"Table_distribution: {table_distribution}")
    LOGGER.warning(f"Table_transformer: {table_transformer}")
    LOGGER.warning(f"Table_anonymize: {table_anonymize}")

    if len(self._training.tables) > 1:
      return (table_distribution, table_transformer, table_anonymize)
    else:
      return (field_distribution, field_transformer, field_anonymize)
