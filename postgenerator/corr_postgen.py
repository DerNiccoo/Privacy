import logging
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from postgenerator.base_postgen import BasePostGen
from models import Training, Table

LOGGER = logging.getLogger(__name__)

class CorrPostGen(BasePostGen):

  def __init__(self):
    super().__init__()
    self._name = 'SDV'

  def _dataframe_cat(self, df, col_names):
    df_c = df.copy()
    
    for col in col_names:
      if is_numeric_dtype(df[col]):
        continue

      df_c[col] = df[col].astype('category').cat.codes
        
    return df_c

  def _apply_post_process(self, df_real: pd.DataFrame, df_fake: pd.DataFrame, training: Training, table_name: str):
    col_names = self._get_columns_without_faker(training, table_name)
    df_real_c = self._dataframe_cat(df_real, col_names)
    df_fake_c = self._dataframe_cat(df_fake, col_names)

    for i, col1 in enumerate(col_names):
      for k, col2 in enumerate(col_names[i:]):
        if col1 == col2:
          continue

        corr = df_real_c[col1].corr(df_real_c[col2])
        
        if abs(corr) > 0.90:
          print(f'Real: {col1:25}:{col2:25} ==> {corr}')
          corr_fake = df_fake_c[col1].corr(df_fake_c[col2])
          print(f'Fake: {col1:25}:{col2:25} ==> {corr_fake}')
          
          df_fake = self._correlation_fix(col1, col2, df_real, df_fake)
          
          df_fake_c = self._dataframe_cat(df_fake, col_names)
          corr_fake = df_fake_c[col1].corr(df_fake_c[col2])
          print(f'Fake After: {col1:25}:{col2:25} ==> {corr_fake}')
          print('---'* 30)

    return df_fake


  def _correlation_fix(self, col1, col2, df_real, df_fake):
    col_main = col1
    col_replace = col2
    if is_numeric_dtype(df_real[col_main]):      
      col_main = col2
      col_replace = col1
        
    uniques_cat = df_fake[col_main].unique().tolist()
    
    for unique_cat in uniques_cat:
      uniques_real = df_real.loc[df_real[col_main] == unique_cat][col_replace].unique().tolist()
      
      if not uniques_real or len(uniques_real) < 1:
        continue
          
      row_count_by_value = len(df_real.loc[df_real[col_main] == unique_cat])
      
      weight = []
      for unique_real in uniques_real:
        weight.append(len( df_real.loc[df_real[col_main] == unique_cat].loc[df_real[col_replace] == unique_real] ) / row_count_by_value)
      
      df_fake.loc[df_fake[col_main] == unique_cat, col_replace] = np.random.choice(uniques_real, size=len(df_fake.loc[df_fake[col_main] == unique_cat]), p=weight)
        
    return df_fake