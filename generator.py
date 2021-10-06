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
from pathlib import Path


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
  _loadedModelPath = None

  def __init__(self, training: Training, metadata: Metadata, loadedModelPath: str = None):
    self._training = training
    self._loadedModelPath = loadedModelPath

    distribution, transformer, anonymize, types = self._get_settings()
    self._anonymize = anonymize
    self._epochs = int(training.epoch)

    if training.tables[0].model == "GaussianCopula":
      if loadedModelPath is not None:
        self._model = GaussianCopula.load(loadedModelPath)
      else:
        self._model = GaussianCopula(field_transformers = transformer, field_distributions = distribution, field_types= types)
      self._model_name = 'gc'
    elif training.tables[0].model == "CTGAN": 
      if loadedModelPath is not None:
        self._model = CTGAN.load(loadedModelPath)
      else:
        self._model = CTGAN(field_transformers = transformer, field_types= types, epochs= self._epochs)
      self._model_name = 'ct'
    elif training.tables[0].model == "CopulaGAN": 
      if loadedModelPath is not None:
        self._model = CopulaGAN.load(loadedModelPath)
      else:
        self._model = CopulaGAN(field_transformers = transformer, field_distributions = distribution, field_types= types, epochs= self._epochs)
      self._model_name = 'cg'
    elif training.tables[0].model == "TVAE": 
      if loadedModelPath is not None:
        self._model = TVAE.load(loadedModelPath)
      else:
        self._model = TVAE(field_transformers = transformer, field_types= types, epochs= self._epochs)
      self._model_name = 'tv'
    elif training.tables[0].model == "Statistical":
      self._model = StatisticalGenerator(field_transformers = transformer, field_distributions = distribution, field_types= types)
      self._model_name = 'st'
    elif training.tables[0].model == "HMA":
      if loadedModelPath is not None:
        self._model = HMA1.load(loadedModelPath)
      else:
        self._model = self._model = HMA1(metadata)
      print(metadata.to_dict())
      
      self._model_name = 'hm'

    LOGGER.warning(f'Using Model: {self._model}')

  def fit(self, tables: dict, new_folder = None):
    data = tables

    columns = []

    for key, value in self._anonymize.items():  # Hier müsste man prüfen, ob es Faker+ ist wenn ja: NICHT entfernen, da es sonst nicht mitgelernt wird!
      if '+' not in value:  # Faker+ nicht löschen!
        columns.append(key)

    if len(tables) == 1:
      data = tables[list(tables.keys())[0]].copy() #Da nur eine Tabelle kein dict von Tabellen übergeben
      if self._training.dataFactor != 1:
        data = data.sample(int(len(data) * self._training.dataFactor)) #data.sample(n = 1000)

      self.save(data, new_folder=new_folder, generated=False)
      data.drop(columns=columns, inplace=True)
    else:
      #Reduction in dataset only. Debugging only. Hardcodeing for given Dataset:
      #try:
      #  data['player'] = tables['player'].sample(n=1000)
      #  boolean_series = tables['player_attributes'].player_api_id.isin(data['player'].player_api_id)
      #  data['player_attributes'] = tables['player_attributes'][boolean_series]
      #  data['player_attributes'] = data['player_attributes'].drop_duplicates(subset=['player_api_id'], keep='first')
      #except Exception as e:
      data = tables
      self.save(data, new_folder=new_folder, generated=False)


      LOGGER.warning(f"For debugging reason only allow training with 100 or less Datapoints")
      LOGGER.warning(f"Started fitting with {len(data)} data points")

    self._model.fit(data)
    self._model.save(self._training.temp_folder_path + "\\model.pkl")

  def sample(self, count: int, column_names):
    LOGGER.warning("start sampling of data")
    LOGGER.warning('Generating rows: ' + str(count))
    df_gen = self._model.sample(num_rows = count)
    df_faker, faker_plus_dict, faker_plus_cols = FakerFactory.apply(self._anonymize, num_rows = count, df_gen = df_gen)


    #TODO: Hier fehlt auch wieder die Unterscheidung zwischen multi und single
    if type(column_names) == list:
      df = pd.concat([df_gen, df_faker], axis=1)
      df = df.reindex(columns=column_names)

      for faker_col in faker_plus_cols:
        df = df.replace({faker_col: faker_plus_dict})
    else:
      result = {}
      for table, cols in column_names.items():
        try:
          faker_cols = [i for i in cols if i in list(df_faker.columns)] # z.B. name, job
          if len(faker_cols) > 0:
            faker_df = pd.DataFrame()
            
            for f_cols in faker_cols:
              faker_df[f_cols] = df_faker[f_cols]

            result[table] = pd.concat([df_gen[table], faker_df], axis=1)
            result[table] = result[table].reindex(columns=cols)
          else:
            result[table] = df_gen[table]
        except KeyError as e:
          LOGGER.warning('Table ' + str(e) + ' not generated. This isnt always a error')

    if len(self._training.tables) == 1:
      return {self._training.tables[0].name: df}
    else:
      return result

  def save(self, tables, appendix = [], new_folder = None, absolut=False, generated=True):
    if absolut == False:
      path = Path(self._training.path)
      new_path = str(path.parent)

      if new_folder is not None:
        new_path += '/' + new_folder
    else:
      new_path = new_folder

    appen = ""
    for apx in appendix:
        appen += str(apx) + "_"
        
    appen = appen[:-1]

    gen_app = ""
    if generated:
      gen_app = "_gen"

    if isinstance(tables, pd.DataFrame):
      t_name = path.stem
      if appen != "":
        t_name += "_" + str(appen)
      self._training.path_gen = new_path + "/" + t_name + gen_app + ".csv"
      LOGGER.warning(f'Saving Data to: {self._training.path_gen}')
      tables.to_csv(path_or_buf=self._training.path_gen, index=False, encoding='utf-8-sig')
    else:
      for t_name, t_value in tables.items():
        if appen != "":
          t_name += "_" + str(appen)
        self._training.path_gen = new_path + "/" + t_name + gen_app + ".csv"
        LOGGER.warning(f'Saving Data to: {self._training.path_gen}')
        t_value.to_csv(path_or_buf=self._training.path_gen)

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

