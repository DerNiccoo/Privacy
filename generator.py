import warnings
warnings.simplefilter(action='ignore')

from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE
from sdv.relational import HMA1
from sdv import Metadata

from models import Training
from fakers import FakerFactory
from generators import StatisticalGenerator

import pandas as pd
import logging
import pathlib

LOGGER = logging.getLogger(__name__)

class Generator:

  _epochs = 300
  _model = None
  _model_name = None
  _training = None
  _anonymize = None
  _performance_mode = False

  def __init__(self, training: Training, metadata: Metadata):
    self._training = training

    distribution, transformer, anonymize, types = self._get_settings()
    self._anonymize = anonymize

    if len(self._training.tables) > 1:
      LOGGER.warning(f"Using HMA1")
      self._model = HMA1(metadata)
    else:
      if training.tables[0].model == "GaussianCopula":
        self._model = GaussianCopula(anonymize_fields = anonymize, field_transformers = transformer, field_distributions = distribution, field_types= types)
        self._model_name = 'gc'
      elif training.tables[0].model == "CTGAN": 
        self._model = CTGAN(anonymize_fields = anonymize, field_transformers = transformer, field_types= types, epochs= self._epochs)
        self._model_name = 'ct'
      elif training.tables[0].model == "CopulaGAN": 
        self._model = CopulaGAN(anonymize_fields = anonymize, field_transformers = transformer, field_distributions = distribution, field_types= types, epochs= self._epochs)
        self._model_name = 'cg'
      elif training.tables[0].model == "TVAE": 
        self._model = TVAE(field_transformers = transformer, field_types= types, epochs= self._epochs)
        self._model_name = 'tv'
      elif training.tables[0].model == "Statistical":
        self._model = StatisticalGenerator(anonymize_fields = anonymize, field_transformers = transformer, field_distributions = distribution, field_types= types)
        self._model_name = 'st'        

    LOGGER.warning(f'Using Model: {self._model}')

  def fit(self, tables: dict):
    data = tables

    columns = []

    for key, value in self._anonymize.items():
      columns.append(key)

    if len(tables) == 1:
      data = tables[list(tables.keys())[0]].copy() #Da nur eine Tabelle kein dict von Tabellen übergeben
      data.drop(columns=columns, inplace=True)
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

  def sample(self, count: int, column_names):
    LOGGER.warning("start sampling of data")
    df_faker = FakerFactory.apply(self._anonymize, num_rows = count)
    df_gen = self._model.sample(num_rows = count)

    #TODO: Hier fehlt auch wieder die Unterscheidung zwischen multi und single
    if type(column_names) == list:
      df = pd.concat([df_gen, df_faker], axis=1)
      df = df.reindex(columns=column_names)

    if len(self._training.tables) == 1:
      return {self._training.tables[0].name: df}
    else:
      return df

  def save(self, tables, appendix = [], new_folder = None):
    path_split = self._training.path.split('/')
    new_path = "/".join(path_split[:-1])

    if new_folder is not None:
      new_path += '/' + new_folder

    appen = ""
    for apx in appendix:
        appen += str(apx) + "_"
        
    appen = appen[:-1]

    if tables == isinstance(tables, pd.DataFrame):
      t_name = path_split[-1].split(".")[0]
      if appen != "":
        t_name += "_" + str(appen)
      self._training.path_gen = new_path + "/" + t_name + "_gen" + ".csv"
      LOGGER.warning(f'Saving Data to: {self._training.path_gen}')
      tables.to_csv(path_or_buf=self._training.path_gen, index=False, encoding='utf-8-sig')
    else:
      for t_name, t_value in tables.items():
        if appen != "":
          t_name += "_" + str(appen)
        self._training.path_gen = new_path + "/" + t_name + "_gen" + ".csv"
        LOGGER.warning(f'Saving Data to: {self._training.path_gen}')
        t_value.to_csv(path_or_buf=self._training.path_gen, index=False, encoding='utf-8-sig')

    return None

  def _get_settings(self):
    table_distribution = {}
    table_transformer = {}
    table_anonymize = {}
    table_field_types = {}

    for table in self._training.tables:
      field_distribution = {}
      field_transformer = {}
      field_anonymize = {}
      field_types = {}

      for attr in table.attributes:
        if attr.field_anonymize is not None:
          field_anonymize[attr.name] = attr.field_anonymize

        if attr.field_transformer is not None:
          field_transformer[attr.name] = attr.field_transformer

        if attr.field_distribution is not None:
          field_distribution[attr.name] = attr.field_distribution

        if attr.dtype is not None:
          field_types[attr.name] = {'type': attr.dtype}

      table_distribution[table.name] = field_distribution
      table_transformer[table.name] = field_transformer
      table_anonymize[table.name] = field_anonymize
      table_field_types[table.name] = field_types

    LOGGER.warning(f"Table_distribution: {table_distribution}")
    LOGGER.warning(f"Table_transformer: {table_transformer}")
    LOGGER.warning(f"Table_anonymize: {table_anonymize}")
    LOGGER.warning(f"Table_types: {table_field_types}")

    if len(self._training.tables) > 1:
      return (table_distribution, table_transformer, table_anonymize, table_field_types)
    else:
      return (field_distribution, field_transformer, field_anonymize, field_types)

