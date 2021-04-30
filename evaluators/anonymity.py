import logging
import itertools

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class Anonymity(BaseEval):

  _settings = None

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

  def _compute(self, real_data, synthetic_data):
    result = self._check_anonymity(real_data, synthetic_data)
    return [{'type': 'anonymity', 'source': 'anonymity', 'metric': 'anonymity Attack', 'name': 'Hintergrundswissen Angriff', 'result': result}]

  def _check_anonymity(self, df_real, df_fake, bin_size = 100):
      # Bins einführen
      # Wenn auftreten eines wertes < bin_size dann als Category interpretieren und nach genaue werte suchen wie mit dtype = 'O'
      # Alle Columns auflisten die verwendet werden sollen
      # Alle Kombinationen davon erstellen
      # Für jede Kombination compare_row aufrufen (modifizieren, dass es mit bins geht)
      # Merken welche Kombinationen bzw Attribute die höchste Wshl. der Zuordnung haben
      df_r, df_f, columns = self._create_bins(df_real, df_fake, bin_size)
      
      column_sets = [[x] for x in columns]
      result = []
      len_fake = len(df_fake)
      
      anon = self._check_column_set(column_sets, df_r, df_f)
      result.extend(anon.copy())

      columns = []
      for entry in anon:
          if entry[0] > 0.5:
              for attr in entry[1]:
                  if attr not in columns:
                      columns.append(attr)


      column_sets = []
      for i in range(2, len(columns) + 1):
          for subset in itertools.combinations(columns, i):
              column_sets.append(list(subset))        

      anon = self._check_column_set(column_sets, df_r, df_f)
      result.extend(anon.copy())
      
      return result

  def _check_column_set(self, column_sets, df_r, df_f):
      anon = []
      for col in column_sets:
          score, identified = self._compare_row(df_f, df_r, col)
          anon.append([score, col, identified, score * identified / len(df_f)])

      anon.sort(key=lambda x: x[0], reverse=True)
      
      return anon

  def _create_bins(self, df_real, df_fake, bin_size = 100):
    columns = df_real.columns.tolist()
    
    columns_binned = columns.copy()
    
    df_r = df_real.copy()
    df_f = df_fake.copy()
    
    for col in columns:
      if len(df_real[col].unique()) > bin_size:
        if df_real[col].dtype == 'int64' or df_real[col].dtype == 'float64':
          #df_r[col + '_AutoAnonBin'] = pd.qcut(df_real[col], q=bin_size, precision=3, duplicates='drop')
          #df_f[col + '_AutoAnonBin'] = pd.qcut(df_real[col], q=bin_size, precision=3, duplicates='drop')
          split = df_real.shape[0] / bin_size
          groups = []
          
          sorted_values = df_real[col].sort_values().tolist()
          
          for i in range(bin_size):
              items = sorted_values[int(i*split):int((i+1)*split)]
              if items:
                  groups.append([min(items), max(items)])
              
          df_r[col + '_AutoAnonBin'] = df_real.apply(lambda row: self._label_bin(row, col, groups, True), axis=1)
          df_f[col + '_AutoAnonBin'] = df_fake.apply(lambda row: self._label_bin(row, col, groups, True), axis=1)
          
          columns_binned.remove(col)
          columns_binned.append(col + '_AutoAnonBin')

    return (df_r, df_f, columns_binned)

  def _label_bin(self, row, col, bin_list, binned = False):
      for bin_group in bin_list:
          if binned:
              if row[col] >= bin_group[0] and row[col] <= bin_group[1]:
                  return str(bin_group)
          else:
              if row[col] in bin_group:
                  return str(bin_group)

  def _compare_row(self, df_real, df_fake, col_name):
      # Echte daten wurden geleaked, wie wahrscheinlich ist es, dass ich diese wiederfinde?
      # Dazu alle möglichen Kombinationen der Attribute bilden und schauen welche am anfälligsten ist.
      # Zur Anon überprüfung muss in die Methode real und fake vertauscht werden. Ein Angreifer verfügt z.B. über die ECHTE Spalten Position und Geburtstag. 
      #  Dann muss in den Fake Daten geschaut werden: wie viele Personen besitzen genau diese Werte. Wenn nur eine kann aus den Fake Daten die Person zugeordnet werden
      
      unique_attributes = df_fake[col_name].dropna().drop_duplicates() # vor drop_dupes #das droppen von NaN sorgt für schlechtere Ergebnisse
      susceptibility = []
      
      unique_identification = 0
      
      for index, row in unique_attributes.iterrows():
          eval_list = []
          for col in col_name:
              if type(row[col]) == str:
                  eval_list.append("(df_real['{0}'] == '{1}')".format(col, row[col]))
              else:
                  eval_list.append("(df_real['{0}'] == {1})".format(col, row[col]))

                  
          df = df_real[eval(" & ".join(eval_list))]            
          entries = len(df)

          if entries == 1:
              unique_identification += 1
          
          if entries > 0:
              susceptibility.append(1 / entries)
              
      result = 0
      if susceptibility:
          result = sum(susceptibility) / len(susceptibility)

      return (result, unique_identification)