import logging
import pandas as pd
import numpy as np
import random

LOGGER = logging.getLogger(__name__)

from generators.basegenerator import BaseGenerator

class StatisticalGenerator(BaseGenerator):

  _security_threshold = 0.1

  def __init__(self, anonymize_fields, field_transformers, field_distributions, field_types):
    super().__init__(anonymize_fields, field_transformers, field_distributions, field_types)

  def _fit(self, df):
    self._df = df
    self._df_fake = pd.DataFrame()

  def _sample(self, num_rows: int):
    # Finden des besten gleichverteilten Attr. mit den wenigsten Ausprägungen aber mehr als 2
    # Loop over DF.
      # Bestimmen der Ausprägungen, wenn 1 skip
      # Wenn mehr: Verteilung bestimmen und mit: 2 / VERTEILUNG beurteiln. Dadurch wäre 2 beschte und die anderen LOWER
      # Besten Wert zurück geben
      
    # Auf basis der Verteilung passend viele Werte bestimmen

    # Iteration: Solange bis neue DF.columns len < übergebene df cols len
      # Sonderfall für NUMERIC Werte einbauen. Zu beginn einfach "random?" Ziehen
      # Prüfe welches Attr. am meisten korrekt vorhergesagt werden kann. (Groupby[cond][attr] / len(df) -> sum der prob)
      # Verteilung von NUR dem Attr bestimmen und dafür default werte verteilen (damit ggf. nicht passende dennoch zugeordnet werden können)
      # Dann auf Basis der wshl. neue Ausprägungen ziehen
    self._df_fake = pd.DataFrame()
    col, weight = self._find_best_fit()

    uniques = self._df[col].unique().tolist()
    self._df_fake[col] = np.random.choice(uniques, size=num_rows, p=weight)

    with_uniques = False

    while len(self._df_fake.columns) < len(self._df.columns):
      condition = self._df_fake.columns.tolist()
      print(len(condition))
      col, weight = self._find_best_fit(condition, with_uniques)

      if col == None and not with_uniques:
        with_uniques = True
        col, weight = self._find_best_fit(condition, with_uniques)
        if col == None:
          break
      elif col == None:
        break

      # Sets the default value if for some reasons no case existed 
      uniques = self._df[col].unique().tolist()
      groupby = self._df.groupby([col])[col].value_counts() / len(self._df)
      
      weight = []
      for attr, score in groupby.items():
        weight.append(score)

      self._df_fake[col] = np.random.choice(uniques, size=num_rows, p=weight)

      # Here the real value should be set
      self._set_probability_values(condition, col)

    self._post_fix_uniques()
    LOGGER.warning('Done Generating')
    LOGGER.warning(f'Difference: {len(self._df_fake.columns)} -- vs -- {len(self._df.columns)}')

    return self._df_fake


  def _post_fix_uniques(self):
    missing = list(set(self._df.columns.tolist()).symmetric_difference(set(self._df_fake.columns.tolist())))

    for miss in missing:
      if self._field_types[miss]['type'] == 'id':
        self._df_fake[miss] = np.arange(start=1, stop=len(self._df_fake) + 1, step=1, dtype=np.int32)

  def _set_probability_values(self, cond, col):
    cond_values, all_probs, all_values = self._get_probabilities(cond, col)
    changed_values = 0

    for comb, weights, values in zip(cond_values, all_probs, all_values):
        query = []
        
        for index, attr in enumerate(comb):
            query.append(f'({cond[index]} == {repr(attr)})')
            
        query = ' & '.join(query)
        mask = self._df_fake.query(query, engine='python').index
        changed_values += len(mask)

        if 1 - random.random() < self._security_threshold:
          res = self._df[col].value_counts() / len(self._df)

          col_values = []
          col_weight = []
          for value, weight in res.items():
            col_values.append(value)
            col_weight.append(weight)

          self._df_fake.loc[mask, col] = np.random.choice(col_values, size=len(mask), p=col_weight)

        else:
          self._df_fake.loc[mask, col] = np.random.choice(values, size=len(mask), p=weights)    

    LOGGER.warning(f'Changed: {changed_values} from {len(self._df_fake)} entries.')

  def _get_probabilities(self, cond, col):
    # Hier könnte man noch prüfen wv elemente genau zugeordnet werden können. Aka. 1 zu 1. Denn diese möchte man vll nicht aus anony gründe. In solchen fällen macht man ein fallback auf default verteilung
    cond_values = []
    set_value = []
    idx = -1

    groupby = self._df.groupby(cond)[col].value_counts() / self._df.groupby(cond)[col].count()

    for attr, score in groupby.items():
      entry = list(attr)
      if entry[:-1] not in cond_values:
        cond_values.append(entry[:-1])
        set_value.append([[entry[-1], score]])
        idx += 1
      else:
        set_value[idx].append([entry[-1], score])
            
    all_values = []
    all_probs = []

    for value in set_value:
      values = []
      prob = []
      
      for v, p in value:
        values.append(v)
        prob.append(p)
          
      all_values.append(values)
      all_probs.append(prob)

    return (cond_values, all_probs, all_values)

  def _find_best_fit(self, condition = None, with_unique = False):
    best_col = None
    best_score = 99999999
    best_weight = []

    if condition is not None:
      cond = condition

    for col in self._df.columns.tolist():
      uniques = self._df[col].unique().tolist()
      
      if len(uniques) == 1 or len(uniques) == len(self._df):
        continue
          
      if not with_unique and len(uniques) / len(self._df) > 0.8:
        continue

      if condition is None:
        cond = [col]
      elif col in cond:
        continue

      groupby = self._df.groupby(cond)[col].value_counts() / len(self._df)
      
      weight = []
      deviation = 0
      for attr, score in groupby.items():
        deviation += abs(score - 1 / len(groupby))
        weight.append(score)
          
      factor = 2 / len(uniques) # Hier will ich einfach die beste Verteilung für die wenigsten Attr
      deviation = deviation / factor
      
      if deviation < best_score:
        best_score = deviation
        best_col = col
        best_weight = weight

    return (best_col, best_weight)