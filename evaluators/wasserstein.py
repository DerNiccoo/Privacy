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

  def _compute(self, df_real, df_fake):
    values = []
    
    for col in df_real.columns.tolist():
      df_r_c = pd.DataFrame(df_real[col], columns=[col])
      df_f_c = pd.DataFrame(df_fake[col], columns=[col])
      
      ht = HyperTransformer()
      try:
        real_data = ht.fit_transform(df_r_c)
        synthetic_data = ht.transform(df_f_c)
      except:  
        df_r_c = df_r_c.append(pd.Series(dtype='object'), ignore_index=True)
          
        try:
          real_data = ht.fit_transform(df_r_c)
          synthetic_data = ht.transform(df_f_c)            
        except:
          LOGGER.warning(f'Error: Unknown Values in "{col}" Column will be skipped')
          continue

      for r_col in real_data.columns:
        score = wasserstein_distance(real_data[r_col], synthetic_data[r_col])
        values.append(1 / (1 + score))

    res = np.nanmean(values)

    return [{'type': 'quality', 'source': 'wasserstein', 'metric': 'Wasserstein', 'name': 'Wasserstein / Earth Movers Distance', 'result': {'Score': res}}]
