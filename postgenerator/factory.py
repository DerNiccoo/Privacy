import logging
from models import Training

from postgenerator.corr_postgen import CorrPostGen

import pandas as pd

class PostGenFactory:

  @classmethod
  def apply(cls, df_real: pd.DataFrame, df_fake: pd.DataFrame, training: Training, table_name: str):
    post_processor = [CorrPostGen()]

    for processor in post_processor:
      df_fake = processor.apply_post_process(df_real, df_fake, training, table_name)

    return df_fake