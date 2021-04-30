import logging

from evaluators.sdvq import SDVQ
from evaluators.closeness import Closeness
from evaluators.similarity import Similarity
from evaluators.sdveval import SDVEval
from evaluators.anonymity import Anonymity

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
    