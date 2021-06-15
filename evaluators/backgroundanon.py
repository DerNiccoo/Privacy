import logging
import pandas as pd

from evaluators.baseeval import BaseEval

LOGGER = logging.getLogger(__name__)

class BackgroundAnonymity(BaseEval):

  _settings = None

  def __init__(self, settings):
    super().__init__(settings)

    self._settings = settings

  def _compute(self, real_data, synthetic_data):
    if len(real_data) != len(synthetic_data):
      raise ValueError('Missmatch of dataset sizes. Skipping messurement.')

    result = self._messure_anonymity(real_data, synthetic_data, top_k=10)
    return [{'type': 'anonymity', 'source': 'background_anonymity', 'metric': 'background anonymity Attack', 'name': 'Hintergrundswissen Angriff', 'result': result}]

  def _label_bin(self, row, col, bin_list, binned = False):
    for bin_group in bin_list:
      if binned:
        if row[col] >= bin_group[0] and row[col] <= bin_group[1]:
          return str(bin_group)
      else:
        if row[col] in bin_group:
          return str(bin_group)

  def _create_bins(self, df_real, df_fake, bin_size = 100):
    columns = df_real.columns.tolist()    
    columns_binned = columns.copy()
    
    df_r = df_real.copy()
    df_f = df_fake.copy()
    
    for col in columns:
      if len(df_real[col].unique()) > bin_size:
          if df_real[col].dtype == 'int64' or df_real[col].dtype == 'float64':
            
            df_r[col + '_AutoAnonBin'], bins = pd.qcut(df_real[col], q=100, retbins=True, labels=False, duplicates='drop')
            bins[0] = float('-inf')
            bins[-1] = float('inf')
            df_f[col + '_AutoAnonBin'] = pd.cut(df_fake[col], bins=bins, labels=False, include_lowest=True)
            
            columns_binned.remove(col)
            columns_binned.append(col + '_AutoAnonBin')
          else:
            split = df_real.shape[0] / bin_size
            groups = []
            
            sorted_values = df_real[col].sort_values().unique().tolist()
            
            for i in range(bin_size):
                groups.append(sorted_values[int(i*split):int((i+1)*split)])
                
            df_r[col + '_AutoAnonBin'] = df_real.apply(lambda row: self._label_bin(row, col, groups), axis=1)
            df_f[col + '_AutoAnonBin'] = df_fake.apply(lambda row: self._label_bin(row, col, groups), axis=1)
            columns_binned.remove(col)
            columns_binned.append(col + '_AutoAnonBin')
              
          df_r.drop(col, axis=1)
          df_f.drop(col, axis=1)
          
    return (df_r[columns_binned], df_f[columns_binned], columns_binned)

  def _conditional_sensitive_find(self, df_real_bin, df_fake_bin, fields, sensitive):
    # Schauen welche OG Combis es gibt
    columns = fields.copy()
    columns.extend(sensitive)
    
    df_r = df_real_bin[columns]
    df_f = df_fake_bin[columns]
    
    df = pd.concat([df_r, df_f])
    df = df.reset_index(drop=True)
    
    df_gpby = df.groupby(list(df.columns))
    scaling_factor = len(df_real_bin) / len(df_fake_bin)
    
    idx = []
    for x in df_gpby.groups.values():
      if len(x) > 1 and min(x) < len(df_real_bin) and max(x) > len(df_real_bin):
        # Zählen wie viele hinzugefügt werden dürfen. min(count(real), count(fake))
        count_real = sum(i < len(df_real_bin) for i in x)
        count_fake = int(sum(i >= len(df_real_bin) for i in x) * scaling_factor)
        counter = min([count_real, count_fake])
        for v in x:
          if v < len(df_real_bin):
            idx.append(v)
            counter -= 1
            if counter == 0:
              break

    df = df.reindex(idx)
    
    result = {}
    result['accuracy'] = (df.groupby(fields).size() / df_real_bin.groupby(fields).size()).sum() / len(df_fake_bin[fields].drop_duplicates())
    result['count'] = int(df.groupby(fields).size().sum())
    result['total_accuracy'] = df.groupby(fields).size().sum() / len(df_real_bin)
    result['fields'] = fields
    result['sortby'] = (result['accuracy'] + result['total_accuracy']) / 2
    #result['risk_level'] = max(result['accuracy'] - (1 / len(df_fake_bin.groupby(fields).size())), 0)
    #result['risk_level2'] = max(result['sortby'] - (1 / len(df_fake_bin.groupby(fields).size())), 0)
    #result['risk_level3'] = max(result['accuracy'] * result['sortby'] - (1 / len(df_fake_bin.groupby(fields).size())), 0)
    
    return result

  def _get_unique_list_of_lists(self, list_with_lists):
    seen = set()
    unique = []

    for x in list_with_lists:
      srtd = tuple(sorted(x))
      if srtd not in seen:
        unique.append(x)
        seen.add(srtd)
    return unique

  def _cleanup_columns(self, df_real, df_fake):
    columns = df_fake.columns.tolist()
    columns_return = columns.copy()
    
    for col in columns:
      if len(df_fake.groupby([col]).size()) == 1:
        columns_return.remove(col)
      elif len(df_real.groupby([col]).size()) / df_real.groupby([col]).size().sum() > 0.8: #Unique Attribute check
        columns_return.remove(col)
            
    return columns_return

  def _get_security_for_attr(self, df_real_bin, df_fake_bin, columns, sensible, top_k = 3):
    cols = []
    columns_raw = []
    for col in columns:
      if col not in sensible:
        cols.append([col])
        columns_raw.append(col)

    combination = cols
    old_combi_len = 0
    
    while old_combi_len < len(combination):
      old_combi_len = len(combination)
      #print(f'Possible Combis: {old_combi_len:6} with an attribute length of: {len(combination[0])}')
      columns_left = combination.copy()

      results = []
      for col in combination:
        res = self._conditional_sensitive_find(df_real_bin, df_fake_bin, col, sensible)
        
        if res['accuracy'] < 0.5 and res['total_accuracy'] < 0.5:
          columns_left.remove(col)
            
          if res['accuracy'] <= 0.15 and res['total_accuracy'] <= 0.15:
            if col[-1] in columns_raw:
                columns_raw.remove(col[-1])
        else:
          results.append(res)
              
      
      sorted_results = sorted(results, key=lambda k: k['sortby'], reverse=True)
      if len(sorted_results) > top_k:
        for res in sorted_results[top_k:]:
          columns_left.remove(res['fields'])
          
      columns_left = self._get_unique_list_of_lists(columns_left)
      
      combination = []
      for col_left in columns_left:
        for col in columns_raw:
          if col in col_left:
            continue

          combi = col_left.copy()
          combi.append(col)
          combination.append(combi)
      
      combination = self._get_unique_list_of_lists(combination)
        
    return sorted_results

  def _messure_anonymity(self, df_real, df_fake, top_k = 10):
    df_real_bin, df_fake_bin, cols = self._create_bins(df_real, df_fake)
    columns = self._cleanup_columns(df_real_bin, df_fake_bin)
    
    data = []
    for col in df_real_bin.columns.tolist():
      sensitive = [col]
      res = self._get_security_for_attr(df_real_bin, df_fake_bin, columns, sensitive, top_k=10)
      if res:
        top_res = res[0]
        #top_res['risk_level'] = max(top_res['accuracy'] - (1 / len(df_fake_bin.groupby(top_res['fields']).size())), 0)
        #top_res['risk_level2'] = max(top_res['sortby'] - (1 / len(df_fake_bin.groupby(top_res['fields']).size())), 0)
        top_res['risk_level'] = max(top_res['accuracy'] * top_res['sortby'] - (1 / len(df_fake_bin.groupby(top_res['fields']).size())), 0)
        if top_res['risk_level'] >= 0.8:
          top_res['risk'] = 'high'
        elif top_res['risk_level'] < 0.8 and top_res['risk_level'] >= 0.5:
          top_res['risk'] = 'medium'
        else:
          top_res['risk'] = 'low'
        data.append([col, top_res['accuracy'], top_res['count'], top_res['total_accuracy'], top_res['fields'], top_res['risk_level'], top_res['risk']])
      else:
        data.append([col, 0, 0, 0, 0, 0, 'low'])
            
    
    #prepare_return_dict(data)
    return self._prepare_return_dict(data)

  def _prepare_return_dict(self, data):
    attributes = []
    
    for col in data:
      attr = {}
      attr['name'] = col[0]
      attr['accuracy'] = col[1]
      attr['count'] = col[2]
      attr['total_accuracy'] = col[3]
      attr['fields'] = col[4]
      attr['risk_level'] = col[5]
      attr['risk'] = col[6]
      
      attributes.append(attr)
        
    col_names = ['Attribute', 'Accuracy', 'Number', 'Total Accuracy', 'Fields', 'Risk_Level', 'Risiko']
    df = pd.DataFrame(data, columns=col_names)    
    high, medium, low = self._print_risk_level(df)
    
    risk = {}
    risk['high'] = int(high)
    risk['medium'] = int(medium)
    risk['low'] = int(low)
    
    return {'risk': risk, 'attributes': attributes}

  def _get_df(self, data):
    col_names = ['Attribute', 'Accuracy', 'Number', 'Total Accuracy', 'Fields', 'Risk_Level', 'Risiko']
    return pd.DataFrame(data, columns=col_names)

  def _print_risk_level(self, dataframe):
    risk = dataframe.groupby(['Risiko']).size()    
    try:
      high = risk['high']
    except:
      high = 0        
    try:
      medium = risk['medium']
    except:
      medium = 0        
    try:
      low = risk['low']
    except:
      low = 0
        
    print(f"Risiko Score: {high}/{medium}/{low}")
    return (high, medium, low)