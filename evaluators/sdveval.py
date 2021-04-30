import logging

from sdv.metrics.tabular import CSTest, KSTest
from sdv.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood
from sdv.metrics.tabular import LogisticDetection, SVCDetection
from sdv.evaluation import evaluate

import numpy as np

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class SDVEval(BaseEval):

  _settings = None

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

  def _compute(self, real_data, synthetic_data):
    result = []

    df = evaluate(real_data, synthetic_data, aggregate=False)

    df = df.replace([np.inf], 1)
    df = df.replace([-np.inf], 0)

    for index, row in df.iterrows():
      result.append({'type': 'quality', 'source': 'sdv', 'metric': row[0], 'name': row[1], 'result': {'raw_score': row[2], 'normalized_score': row[3], 'min_value': row[4], 'max_value': row[5], 'goal': row[6]}})

    return result

