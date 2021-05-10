import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

class BaseGenerator:
  _field_anonymizers = None
  _field_transformers = None
  _field_distributions = None
  _field_types = None
  
  def __init__(self, anonymize_fields, field_transformers, field_distributions, field_types):
    self._field_anonymizers = anonymize_fields
    self._field_transformers = field_transformers
    self._field_distributions = field_distributions
    self._field_types = field_types
      
  def _fit(self, df):
    raise NotImplementedError()
      
  def fit(self, df):
    return self._fit(df)
  
  def _sample(self, num_rows: int):
    raise NotImplementedError()
      
  def sample(self, num_rows: int):
    return self._sample(num_rows)