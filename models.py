from pydantic import BaseModel, PrivateAttr
import typing
from typing import List, Optional
import pandas as pd


class Attribute(BaseModel):
  name: str
  dtype: str
  field_distribution: Optional[str] = None
  field_transformer: Optional[str] = None
  field_anonymize: Optional[str] = None
  sensible: Optional[bool] = False

class Table(BaseModel):
  name: str
  attributes: List[Attribute]
  model: Optional[str] = None

class DataEvaluator(BaseModel):
  config: typing.Dict[str, str]

class Training(BaseModel):
  path: str
  path_gen: Optional[str]
  tables: List[Table]
  evaluators: typing.Dict[str, DataEvaluator]
  epoch: Optional[float]
  dataFactor: Optional[float] # Wie viele Daten betracht werden sollen [0.01, 1]
  dataAmount: Optional[float] # Wie viele Daten generiert werden sollen
  
  temp_folder_path: str = Optional[str]
  train_tables : Optional[List[str]] #TODO: Ugly workarround da es einfach mit _ davor nicht geht und zuweisungen...
  train_attr : Optional[typing.Dict[str, List[str]]]

  def get_tables(self):
    if self.train_tables is None:
      self._get_table_and_attr()

    return self.train_tables

  def get_attr(self):
    if self.train_attr is None:
      self._get_table_and_attr()

    return self.train_attr

  def _get_table_and_attr(self):
    train_tables = []
    train_attr = {}

    for table in self.tables:
      train_tables.append(table.name)

      list_attr = []
      for attr in table.attributes:
        list_attr.append(attr.name)

      train_attr[table.name] = list_attr

    self.train_tables = train_tables
    self.train_attr = train_attr
