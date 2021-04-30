import logging
import os.path

import pandas as pd
import sqlite3
from sdv import Metadata
from models import Training, Table, Attribute, DataEvaluator
import operator

from connector.baseconnector import BaseConnector


LOGGER = logging.getLogger(__name__)

class SQLConnector(BaseConnector):

  _tables = None
  _pk_relation = None
  _fk_relation = None

  def __init__(self, path : str):
    super().__init__(path)

  def _sql_identifier(self, s : str):
    return '"' + s.replace('"', '""') + '"'

  def _get_schema(self):    
    '''This method returns the order of tables to be generated. Tables have relations between each other (FK) that needs to be created first in order to generate consistend data. 
    TODO: Remove pk; fk stuff from here. Method already big enough, will get own method to call when needed. 
    TODO: Vll alle Methoden so erweitern, dass die ihr Ergebnis zwischenspeichern und bei mehrfachen Aufruf einfach wiedergeben können. 

    TODO: Das Framework SDV erlaubt es nur einen Primary Key bei relationalen Tabellen zu haben. Dieser MUSS der gezeigte FK Schlüssel der relationalen Tabellen sein. 
    '''

    LOGGER.debug('Executing get_schema')

    con = sqlite3.connect(self.path)

    rows = con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    tables = [row[0].lower() for row in rows]

    if "sqlite_sequence" in tables:
      tables.remove('sqlite_sequence')
    
    pfkey_count = {}

    for table in tables:
      pfkey_count[table.lower()] = {} #init für die prim-foreign-key Count dict

    fk_relation = {}
    pk_relation = {}
    
    for table in tables:
      rows = con.execute("PRAGMA table_info({})".format(self._sql_identifier(table)))
      attributes = rows.fetchall()
        
      pk = []
      for attr in attributes:
        if attr[5] == 1:
          pk.append(attr[1])    
      pk_relation[table] = pk # Default bestimmung des PK, wird infolge dessen überschrieben wenn nötig
      
      rows = con.execute("PRAGMA foreign_key_list({})".format(self._sql_identifier(table)))
      foreign_key_list = rows.fetchall()
      fkeys = []      

      for fk in foreign_key_list:
        fkeys.append({'table': fk[2].lower(), 'origin': fk[3], 'dest': fk[4]})

        if fk[4] in pfkey_count[fk[2].lower()]:
          pfkey_count[fk[2].lower()][fk[4]] += 1
        else:
          pfkey_count[fk[2].lower()][fk[4]] = 0
      
      fk_relation[table] = fkeys
      
    # Updating primary Key of tables based on highest count of foreign key relations
    for table in tables:
      if len(pfkey_count[table]) > 0:
        new_pk = max(pfkey_count[table].items(), key=operator.itemgetter(1))[0] # Getting key with highest value from dict
        pk_relation[table] = [new_pk]

        
    con.close()

    self._tables = tables
    self._pk_relation = pk_relation
    self._fk_relation = fk_relation

    LOGGER.warning('Finished get_schema')
    LOGGER.warning(self._pk_relation)

    return (tables, pk_relation, fk_relation)

  def _get_metadata(self):
    if self._tables is None or self._pk_relation is None or self._fk_relation is None:
      self._get_schema()

    metadata = Metadata()
    con = sqlite3.connect(self.path)

    for table in self._tables:
      df = pd.read_sql_query('SELECT * FROM ' + table + ' LIMIT 5', con)

      metadata.add_table(
        name=table,
        data=df,
        primary_key=self._pk_relation[table][0] #TODO: Könnte sein, dass es mehr prim keys gibt. Dann schauen.
      )

    table_list = []
    for table in self._tables:
      meta_dict = metadata.get_table_meta(table)

      attributes_list = []
      for name, dtype in meta_dict['fields'].items():
        attributes_list.append(Attribute(**{'name': name, 'dtype': str(dtype['type'])}))

      table = Table(**{'name': table, 'attributes': attributes_list, 'model': 'TVAE'})
      table_list.append(table)


    con.close()

    data_evaluator = DataEvaluator(**{'config': {'extras': 'none'}})
    training = Training(**{'path': self.path, 'tables': table_list, 'evaluators': {'closeness': data_evaluator}})


    return training

  def _get_selected_attributes(self, table: Table):
    """
      TODO: Ist ggf. redundant, da es umgebaut wurde, dass die Attr. Namen alle in einer Liste verfügbar sind. 
    """
    statement = ""

    for attr in table.attributes:
      statement += attr.name + ', '

    statement = statement[:-2]
    
    return statement

  def _get_training_data(self, training: Training):
    tables = {}
    metadata = Metadata()

    table_order, pk_relation, fk_relation = self._get_schema()
    con = sqlite3.connect(self.path)

    for table in training.tables:
      df = pd.read_sql_query('SELECT '+ self._get_selected_attributes(table) +' FROM ' + table.name, con)
      tables[table.name] = df

      metadata.add_table(
        name=table.name,
        data=df,
        primary_key=pk_relation[table.name][0] #TODO: Könnte sein, dass es mehr prim keys gibt. Dann schauen.
      )

      #TODO: FK Relation zu den Metadaten hinzufügen
      for fk_rel in fk_relation[table.name]:
        #TODO: Prüfen ob Tabelle und Attrs mit trainiert werden sollen
        if fk_rel["table"] in training.get_tables():
          if fk_rel["origin"] in training.train_attr[table.name] and fk_rel["dest"] in training.train_attr[fk_rel["table"]]:
            LOGGER.info("adding valid relation")
            
            #TODO: Die Metadaten hier adden den falschen Key zur Tabelle als FK. Muss möglicherweise manuell angepasst werden
            metadata.add_relationship(
              parent=fk_rel["table"],
              child=table.name,
              foreign_key=fk_rel["dest"]
            )

          else:
            LOGGER.warning("Missing attr. for adding relations to train")
        else:
          LOGGER.warning("Table should not be trained. Skipped relation")

      #TODO: Vergleichen ob die ausgewählten Trainingssettings passen

    con.close()
    return (tables, metadata)

