import logging
from suggestions.faker_suggestion import FakerSuggestion
from suggestions.sdv_suggestion import SDVSuggestion

from models import Training

import pandas as pd

LOGGER = logging.getLogger(__name__)

class SuggestionFactory:

  @classmethod
  def create(cls, tables: dict, training: Training):
    suggestions = []

    suggestors = [FakerSuggestion(), SDVSuggestion()]

    for suggestor in suggestors:
      result = suggestor.get_suggestions(tables, training)

      if result:
        suggestions.extend(result)

    return suggestions