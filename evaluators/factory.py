import logging

from evaluators.sdvq import SDVQ
from evaluators.closeness import Closeness
from evaluators.similarity import Similarity
from evaluators.sdveval import SDVEval
from evaluators.anonymity import Anonymity
from evaluators.wasserstein import Wasserstein
from evaluators.backgroundanon import BackgroundAnonymity

LOGGER = logging.getLogger(__name__)

class EvalFactory:

  @classmethod
  def create(cls, name: str, settings):

    if name == 'SDVQ':
      return SDVQ(settings)
    elif name == 'closeness':
      return Closeness(settings)
    elif name == 'similarity':
      return Similarity(settings)
    elif name == 'sdveval':
      return SDVEval(settings)
    elif name == 'anonymity':
      return Anonymity(settings)
    elif name == 'wasserstein':
      return Wasserstein(settings)
    elif name == 'backgroundanonymity':
      return BackgroundAnonymity(settings)