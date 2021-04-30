import logging

from sdv.metrics.tabular import CSTest, KSTest
from sdv.metrics.tabular import BNLikelihood, BNLogLikelihood, GMLogLikelihood
from sdv.metrics.tabular import LogisticDetection, SVCDetection

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class SDVQ(BaseEval):

  _settings = None
  _tests = []
  _names = []

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

    if 'tests' in self._settings.config:
      tests = self._settings.config['tests'].split(',')
      for name in tests:
        self._tests.append(self._get_model(name))
        self._names.append(name)
    """
    if 'SDVQ' in kwargs:
      self._settings = kwargs['SDVQ']

      tests = kwargs['SDVQ']['config']['tests'].split(',')
      for name in tests:
        self._tests.append(_get_model(name))
    """

  def _get_model(self, name):
    if name == 'CSTest':
      return CSTest
    elif name == 'KSTest':
      return KSTest
    elif name == 'BNLikelihood':
      return BNLikelihood
    elif name == 'BNLogLikelihood':
      return BNLogLikelihood
    elif name == 'GMLogLikelihood':
      return GMLogLikelihood
    elif name == 'LogisticDetection':
      return LogisticDetection
    elif name == 'SVCDetection':
      return SVCDetection

  def _compute(self, real_data, synthetic_data):
    for test, name in zip(self._tests, self._names):
      score = test.compute(real_data.fillna(0), synthetic_data.fillna(0))
      print(f'Test: {name:30} -> {score}')