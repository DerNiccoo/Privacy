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
        if attribute.dtype == 'id':
          t_anon[table.name].append(attribute.name) # ID fields werden komplett neu erstellt, sollen nicht mit einbezogen werden in Messungen

    return t_anon

  def run(self):
    path_split = self._training.path.split('/')
    new_path = "/".join(path_split[:-1])

    dc = DataConnector.load(path=self._training.path)
    real_tables, _ = dc.get_training_data(self._training)

    field_anonymize = self._get_anonymized_fields()

    evaluation_result = []

    for table in self._training.tables:
      if self._training.path_gen == None:        
        synthetic_table = pd.read_csv(new_path + "/" + table.name + "_gen.csv")
      else:
        synthetic_table = pd.read_csv(self._training.path_gen)

      real = real_tables[table.name].drop(field_anonymize[table.name], axis=1)
      synthetic = synthetic_table.drop(field_anonymize[table.name], axis=1)

      for method in self._methods: # Entfernen der Faker erstellten Attribute, da diese offensichtlich random sind
        print(method)
        try:
          results = method.compute(real, synthetic)
          evaluation_result.extend(results)
        except:
          LOGGER.warning(f'Error compute eval.{method}')

    return evaluation_result


  """
  Deine Aufgabe ist es, die benötigten Evaluation Methoden zu instanzieren. Die benötigten Dataframes zu laden und ggf. auch mit mehreren Umgehene können.

  Idee: Laden der neuen CSV Datei mittels DataConnector? (oder so laden?)
  Erhalten der Spalten, die benötigt // generiert wurden
  Laden der anderen Tabelle(n) mit dem DataConnector --> Beschränken dabei auf Attribute die nur auch in den neuen Daten drinne sind (sind die einzigen, die trainiert wurden)

  Beachte: Irgendwie/wo müssen noch die Faker Attribute beachtet werden. Diese sollten NICHT mit in die Bewertung einfließen, da diese logischerweiße Random sind. 
  """