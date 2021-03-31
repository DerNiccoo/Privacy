from pydantic import BaseModel
import typing
from typing import List, Optional


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

class Training(BaseModel):
  path: str
  tables: List[Table]

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
