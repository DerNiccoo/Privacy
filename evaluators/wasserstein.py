import logging
import numpy as np
import pandas as pd
from rdt import HyperTransformer
from scipy.stats import wasserstein_distance

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class Wasserstein(BaseEval):

  _settings = None

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

  def _compute(self, real_data, synthetic_data):
    ht = HyperTransformer()
    real_data = real_data.append(pd.Series(dtype='object'), ignore_index=True)
    real = ht.fit_transform(real_data)

    synthetic = ht.transform(synthetic_data)

    values = []

    for col in real.columns.tolist():
      score = wasserstein_distance(real[col], synthetic[col])
      values.append(1 / (1 + score))

    return [{'type': 'quality', 'source': 'wasserstein', 'metric': 'Wasserstein', 'name': 'Wasserstein / Earth Movers Distance', 'result': {'Score': np.nanmean(values)}}]
