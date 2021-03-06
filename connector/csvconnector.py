import logging
import os.path

import pandas as pd
from sdv import Metadata
from models import Training, Table, Attribute, DataEvaluator
import operator
from pathlib import Path

from connector.baseconnector import BaseConnector


LOGGER = logging.getLogger(__name__)

class CSVConnector(BaseConnector):

  _tables = None
  _pk_relation = None
  _fk_relation = None
  _table_name = ''


  def __init__(self, path : str):
    super().__init__(path)
    self._table_name = str(Path(self.path).name).split('.')[0]
    #self._table_name = self.path.split('/')[-1].split('.')[0]

  def _get_schema(self):
    self._table_name = self.path.split('/')[-1].split('.')[0]

    return ([self._table_name], {}, {})

  def _get_metadata(self):
    df = self._get_dataframe()

    metadata = Metadata()

    metadata.add_table(name=self._table_name, data=df)
    meta_dict = metadata.get_table_meta(self._table_name)

    attributes_list = []
    for name, dtype in meta_dict['fields'].items():
      attributes_list.append(Attribute(**{'name': name, 'dtype': str(dtype['type'])}))

    table = Table(**{'name': self._table_name, 'attributes': attributes_list, 'model': 'TVAE'})
    data_evaluator = DataEvaluator(**{'config': {'extras': 'none'}})
    training = Training(**{'path': self.path, 'tables': [table], 'evaluators': {'closeness': data_evaluator}})

    return training

  def _get_trainable_columns(self, attributes: [Attribute]):
    columns = []
    for attr in attributes:
      columns.append(attr.name)

    return columns

  def _get_training_data(self, training: Training):
    df = self._get_dataframe()

    if self._table_name == '':
      self._table_name = self.path.split('/')[-1].split('.')[0]

    train_columns = self._get_trainable_columns(training.tables[0].attributes) #Bei CSV kann es eh immer nur eine Tabelle geben. 
    return ({self._table_name: df[train_columns]}, None) # Da nur eine Tabelle werden keine Metadaten benötigt...(da nur bei HMA benötigt)

  def _get_column_names(self):
    df = self._get_dataframe()

    return df.columns.tolist()

  def _get_tables(self, replace_na: bool = True):
    df = self._get_dataframe(replace_na)
    return {self._table_name: df}

  def _get_dataframe(self, replace_na: bool = True):
    with open(self.path) as f:
      first_line = f.readline()
        
    seperator = [',', ';']
    best_sep = None
    best_sep_count = -1
    for sep in seperator:
      count = first_line.count(sep)
      if count > best_sep_count:
        best_sep_count = count
        best_sep = sep


    df = pd.read_csv(self.path, sep=best_sep)

    if replace_na:
      for col in df.columns.tolist():
        if df[col].dtypes == 'object':
          df[col].fillna('', inplace=True)
        else:
          df[col].fillna(0, inplace=True)

    return df