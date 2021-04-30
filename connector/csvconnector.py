import logging
import os.path

import pandas as pd
from sdv import Metadata
from models import Training, Table, Attribute, DataEvaluator
import operator

from connector.baseconnector import BaseConnector


LOGGER = logging.getLogger(__name__)

class CSVConnector(BaseConnector):

  _tables = None
  _pk_relation = None
  _fk_relation = None
  _table_name = ''


  def __init__(self, path : str):
    super().__init__(path)

  def _get_schema(self):
    self._table_name = self.path.split('/')[-1].split('.')[0]

    return ([self._table_name], {}, {})

  def _get_metadata(self):
    df = pd.read_csv(self.path)

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
    df = pd.read_csv(self.path)

    if self._table_name == '':
      self._table_name = self.path.split('/')[-1].split('.')[0]

    train_columns = self._get_trainable_columns(training.tables[0].attributes) #Bei CSV kann es eh immer nur eine Tabelle geben. 
    return ({self._table_name: df[train_columns]}, None) # Da nur eine Tabelle werden keine Metadaten benötigt...(da nur bei HMA benötigt)