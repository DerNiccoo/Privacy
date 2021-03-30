import logging
from connector.sqlconnector import SQLConnector

LOGGER = logging.getLogger(__name__)

class DataConnector:
  """ Genetic class for connecting to all different kind of database systems

  """

  @classmethod
  def load(cls, path : str):
    """Load a database from a given path.

    Args:
        path (str):
            Path from which to load the database.
    """
    _format = path.split('.')

    if len(_format) < 2:
      LOGGER.error('Unable to detect file format.')

    if _format[-1] == 'sqlite':
      return SQLConnector(path)
    #else:
    #  return CSVConnector(path)
    