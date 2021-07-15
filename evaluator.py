import warnings
warnings.simplefilter(action='ignore')

from models import Training
import pandas as pd
import logging


from connector import DataConnector
from evaluators import EvalFactory

LOGGER = logging.getLogger(__name__)

class Evaluator:

  _training = None
  _methods = [] # Liste die alle Verfahren umfasst, die angewendet werden sollen

  def __init__(self, training: Training):
    self._training = training
    self._methods = []

    for method, config in training.evaluators.items():
      self._methods.append(EvalFactory.create(method, config))

  def _get_anonymized_fields(self):
    t_anon = {}

    for table in self._training.tables:
      t_anon[table.name] = []
      for attribute in table.attributes:
        if attribute.field_anonymize is not None:
          t_anon[table.name].append(attribute.name)
        #if attribute.dtype == 'id':
        #  t_anon[table.name].append(attribute.name) # ID fields werden komplett neu erstellt, sollen nicht mit einbezogen werden in Messungen

    return t_anon

  def _join_tables(self, tables_gen, tables_og):
    dc = DataConnector.load(path=self._training.path)
    _, metadata = dc.get_training_data(self._training)
    metadata = metadata.to_dict()

    df_real = pd.DataFrame()
    df_fake = pd.DataFrame()
    added_tables = []

    for key, table in metadata['tables'].items():
      for key_at, attr in table['fields'].items():
        if 'ref' in attr:
          if key in added_tables:
            df_real.join(tables_og[attr['ref']['table']].set_index(attr['ref']['field']), on=attr['ref']['field'], lsuffix='_caller', rsuffix='_other')
            df_fake.join(tables_gen[attr['ref']['table']].set_index(attr['ref']['field']), on=attr['ref']['field'], lsuffix='_caller', rsuffix='_other')
            added_tables.append(attr['ref']['table'])
          elif attr['ref']['table'] in added_tables:
            df_real.join(tables_og[key].set_index(attr['ref']['field']), on=attr['ref']['field'], lsuffix='_caller', rsuffix='_other')
            df_fake.join(tables_gen[key].set_index(attr['ref']['field']), on=attr['ref']['field'], lsuffix='_caller', rsuffix='_other')
            added_tables.append(key)
          else:
            df_real = tables_og[attr['ref']['table']].join(tables_og[key].set_index(attr['ref']['field']), on=attr['ref']['field'], lsuffix='_caller', rsuffix='_other')
            df_fake = tables_gen[attr['ref']['table']].join(tables_gen[key].set_index(attr['ref']['field']), on=attr['ref']['field'], lsuffix='_caller', rsuffix='_other')
            added_tables.append(key)
            added_tables.append(attr['ref']['table'])

    return (df_real, df_fake)

  def run(self):
    if self._training.path_gen == None:
      path_split = self._training.path.split('/')
      new_path = "/".join(path_split[:-1])
    else:
        path_split = self._training.path_gen.split('/')
        new_path = "/".join(path_split[:-1])

    field_anonymize = self._get_anonymized_fields()

    evaluation_result = []

    tables_gen = {}
    tables_og = {}

    for table in self._training.tables:
      real_table = pd.read_csv(new_path + "/" + table.name + ".csv")
      synthetic_table = pd.read_csv(new_path + "/" + table.name + "_gen.csv")

      real = real_table.drop(field_anonymize[table.name], axis=1)
      synthetic = synthetic_table.drop(field_anonymize[table.name], axis=1)

      tables_gen[table.name] = synthetic
      tables_og[table.name] = real

      for method in self._methods: # Entfernen der Faker erstellten Attribute, da diese offensichtlich random sind
        print(method)
        try:
          results = method.compute(real, synthetic)
          evaluation_result.extend(results)
        except:
          LOGGER.warning(f'Error compute eval.{method}')

    if len(self._training.tables) > 1:
      dc = DataConnector.load(path=self._training.path)
      real = dc.join_table(self._training, tables_og)
      synthetic = dc.join_table(self._training, tables_gen)
      eval = EvalFactory.create('backgroundanonymity', {})
      results = eval.compute(real, synthetic)
      evaluation_result.extend(results)

    return evaluation_result


  """
  Deine Aufgabe ist es, die benötigten Evaluation Methoden zu instanzieren. Die benötigten Dataframes zu laden und ggf. auch mit mehreren Umgehene können.

  Idee: Laden der neuen CSV Datei mittels DataConnector? (oder so laden?)
  Erhalten der Spalten, die benötigt // generiert wurden
  Laden der anderen Tabelle(n) mit dem DataConnector --> Beschränken dabei auf Attribute die nur auch in den neuen Daten drinne sind (sind die einzigen, die trainiert wurden)

  Beachte: Irgendwie/wo müssen noch die Faker Attribute beachtet werden. Diese sollten NICHT mit in die Bewertung einfließen, da diese logischerweiße Random sind. 
  """