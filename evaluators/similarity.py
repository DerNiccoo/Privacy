import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class Similarity(BaseEval):

  _settings = None

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

  def _get_col_quantile(self, df, col, buckets):
      step = 1 / buckets
      container = []
      
      for i in range(buckets):
          container.append(df[col].quantile(step * i))
      
      return container

  def _compare_two_dataframe(self, df1, df2, buckets = 10):
      res = []
      dif_total = 0

      col_names = df1.columns.tolist()

      for col in col_names:
          q1 = self._get_col_quantile(df1, col, buckets)
          q2 = self._get_col_quantile(df2, col, buckets)
          
          dif = []
          for i in range(buckets):
              dif.append(q1[i] - q2[i])
              
          percent = 1
          if abs(sum(q1)) > abs(sum(q2)):
              percent = abs(sum(q2)) / abs(sum(q1))
          elif abs(sum(q1)) < abs(sum(q2)):
              percent = abs(sum(q1)) / abs(sum(q2))
              
          dif_total += sum(dif)
          entry = {
              'name': col,
              'q1': q1,
              'q2': q2,
              'dif': dif,
              'dif_total': sum(dif),
              'percent': percent,
          }
          #res.append(entry)

          res.append(percent)
          
      return (res, dif_total)

  def _preprocess(self, real_df, synth_df):
    real_df = real_df.select_dtypes(include="number").dropna()
    synth_df = synth_df[real_df.columns.tolist()].dropna()

    return (real_df, synth_df)

  def _cast_categorical(self, row, rev_mapping):
      if row in rev_mapping:
          return rev_mapping[row]
      else:
          return -1

  def _dataframe_cat(self, df_real, df_fake, col_names):
      df_real_c = df_real.copy()
      df_fake_c = df_fake.copy()
      
      for col in col_names:
          if is_numeric_dtype(df_real[col]):
              continue

          mapping = dict( enumerate(df_real[col].astype('category').cat.categories ) )
          rev_mapping = dict((v,k) for k,v in mapping.items())
              
          df_real_c[col] = df_real[col].apply(lambda x: self._cast_categorical(x, rev_mapping))
          df_fake_c[col] = df_fake[col].apply(lambda x: self._cast_categorical(x, rev_mapping))
      return (df_real_c, df_fake_c)

  def _compute(self, real_data, synthetic_data):
    buckets = 100

    df_real_c, df_fake_c = self._dataframe_cat(real_data, synthetic_data, real_data.columns.tolist())

    error = []
    for col in real_data.columns.tolist():
        quantile_fake = self._get_col_quantile(df_fake_c, col, buckets)
        quantile_real = self._get_col_quantile(df_real_c, col, buckets)

        dev = 0
        for q_f, q_r in zip(quantile_fake, quantile_real):
            dev += abs(q_f - q_r)

        maxima = abs(max(quantile_real, key=abs))
        if abs(max(quantile_fake, key=abs)) > maxima:
            maxima = abs(max(quantile_fake, key=abs))

        error.append(dev / (buckets * maxima))

    corr_new = real_data.corr()
    corr_og = synthetic_data.corr()
    df_diff = corr_new.subtract(corr_og)
    df = df_diff.where(np.triu(np.ones(df_diff.shape)).astype(np.bool))
    correlation_diff = df.sum().abs().sum()/df.shape[0]


    return [{'type': 'quality', 'source': 'similarity', 'metric': 'Value comparison', 'name': 'Ähnlichkeitsbestimmung', 'result': {'Avg_bucket_similarity': 1 - (sum(error) / len(error)), 'Max_bucket_similarity': 1 - min(error), 'Min_bucket_similarity': 1 - max(error), 'correlation_difference': correlation_diff}}]

  def _compute_old(self, real_data, synthetic_data):
    real_df, synth_df = self._preprocess(real_data, synthetic_data)

    res, dif_total = self._compare_two_dataframe(real_df, synth_df, 100) # die 100 könnten durch die settings geladen werden

    #print(f'Avg bucket similarity: {sum(res) / len(res)}')
    #print(f'Max bucket similarity: {max(res)}')
    #print(f'Min bucket similarity: {min(res)}')


    corr_new = real_data.corr()
    corr_og = synthetic_data.corr()
    df_diff = corr_new.subtract(corr_og)
    df = df_diff.where(np.triu(np.ones(df_diff.shape)).astype(np.bool))
    correlation_diff = df.sum().abs().sum()/df.shape[0]

    #print(f'Total correlation difference: {correlation_diff}')

    return [{'type': 'quality', 'source': 'similarity', 'metric': 'Value comparison', 'name': 'Ähnlichkeitsbestimmung', 'result': {'Avg_bucket_similarity': sum(res) / len(res), 'Max_bucket_similarity': max(res), 'Min_bucket_similarity': min(res), 'correlation_difference': correlation_diff}}]