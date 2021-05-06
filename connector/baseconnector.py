import logging
import os.path
from models import Training, Table

LOGGER = logging.getLogger(__name__)

class BaseConnector:

  path = None

  def __init__(self, path : str):
    if os.path.isfile(path):
      self.path = path
    else:
      LOGGER.error('Unable to open path. Not a file.')


  def _get_schema(self):
    raise NotImplementedError()

  def get_schema(self):
    '''
    Returns the automatic deteced metadata of a given dataset.
    '''
    return self._get_schema()

  def _get_metadata(self):
    raise NotImplementedError()

  def get_metadata(self):
    return self._get_metadata()

  def _get_column_names(self):
    raise NotImplementedError()

  def get_column_names(self):
    return self._get_column_names()

  def _get_tables(self, replace_na: bool = True):
    raise NotImplementedError()

  def get_tables(self, replace_na: bool = True):
    return self._get_tables(replace_na)

  def _get_training_data(self, training: Training):
    raise NotImplementedError()

  def get_training_data(self, training: Training):
    return self._get_training_data(training)