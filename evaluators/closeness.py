import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class Closeness(BaseEval):

  _settings = None

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

  def _preprocess(self, real_df, synth_df):
    real_df = real_df.select_dtypes(include="number").dropna()
    synth_df = synth_df[real_df.columns.tolist()].dropna()

    return (real_df, synth_df)
    #return df.apply(pd.to_numeric, errors='coerce').fillna(df)

  def _compute(self, real_data, synthetic_data):
    neigh = NearestNeighbors(n_neighbors=1)

    real_df, synth_df = self._preprocess(real_data, synthetic_data)

    neigh.fit(real_df)
    total_dist = 0
    
    neighbors = neigh.kneighbors(synth_df)
    
    #values = np.array(neighbors)
    #idx = neighbors[1][np.argmin(values)][0]
    
    idx = np.arange(len(neighbors[0]))
    result = neighbors[0][:, 0]
    idx_point = np.array([result, idx])
    idx_point[:, idx_point[0].argsort()]# Sorted liste der Distanzen mit Index
    
    total_dist = sum(result)
    
    return [{'type': 'quality', 'source': 'closeness', 'metric': 'Nächste-Nachbarn', 'name': 'Nächste-Nachbarn-Klassifikation', 'result': { 'total_distance': total_dist, 'avg_distance': total_dist / len(result)}}]