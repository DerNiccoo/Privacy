import uvicorn
import os

from fastapi import FastAPI, HTTPException, Query
from typing import List
from models import Training, Table
from pathlib import Path

from connector import DataConnector
from generator import Generator
from evaluator import Evaluator
from suggestions import SuggestionFactory
from postgenerator import PostGenFactory

from pathlib import Path
from datetime import datetime
from shutil import copyfile
import json
import random

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://localhost:4200",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

debug = False
sizes = [50, 100, 311, 622, 3110, 6220, 31100]

@app.get("/schema/{db_path:path}")
async def get_database_schema(db_path: str):  
  try:
    try:
      dc = DataConnector.load(path=db_path)  
    except Exception as e:
      raise Exception("Schema: Konnte angegebene Datei nicht finden. Error:" + str(e))

    table_order, pk_relation, fk_relation = dc.get_schema()
    metadata = dc.get_metadata()
    metadata.temp_folder_path=None

    try:
      suggestions = SuggestionFactory.create(dc.get_tables(replace_na = False), metadata)
    except Exception as e:
      raise Exception("Schema: Konnte empfehlungen nicht Erstellen, möglicherweise leere Tabelle angegeben. Error:" + str(e))

    return {"db_path": db_path, "table_order": table_order, "pk_relation": pk_relation, "fk_relation": fk_relation, 'metadata': metadata, 'suggestions': suggestions}
  except Exception as e:
    raise HTTPException(status_code=404, detail="Schema: " + str(e)) 

@app.post("/training/")
async def start_training(training: Training):
  try:
    if '_gen' in training.tables[0].name:
      training.path_gen = training.path
      return training

    dc = DataConnector.load(path=training.path)
    tables, metadata = dc.get_training_data(training)

    dt = datetime.now()
    path = Path(training.path)
    folder_name = str(dt.strftime("%d.%m, %H.%M")) + 'Uhr ' + path.stem
    new_dir = Path(str(path.parent), folder_name)
    new_dir.mkdir(parents=True, exist_ok=True)

    training.temp_folder_path = str(new_dir)
    copyfile(path, str(new_dir) + '\\' + path.name)
    with open(str(new_dir) + '\\settings.json', 'w+') as outfile:
      json.dump(training.dict(), outfile)
    
    gen = Generator(training, metadata)
    gen.fit(tables)

    real_data = dc.get_tables()
    length = int(len(real_data[training.tables[0].name]) * training.dataAmount)

    new_data = gen.sample(length, dc.get_column_names())

    # Post Processing:
    try:
      real_data = dc.get_tables()

      for table_name, table_df in new_data.items():
        new_data[table_name] = PostGenFactory.apply(real_data[table_name], table_df, training, table_name)
    except Exception as e:
      raise Exception(detail="Generierung: Konnte post processing nicht anwenden. Error: " + str(e))

    gen.save(new_data, new_folder=folder_name)

    if debug:
      for size in sizes:
        new_data = gen.sample(size, dc.get_column_names())
        gen.save(new_data, size)      

    return training
  except Exception as e:
    raise HTTPException(status_code=404, detail="Generierung: " + str(e)) 

@app.post("/evaluate/")
async def start_evaluation(training: Training):
  try:
    p = Path(training.path)
    evaluator = Evaluator(training)
    result = evaluator.run()

    with open(training.temp_folder_path + '\\evaluation.json', 'w+') as outfile:
      json.dump(result, outfile)

    return [{'name': str(p.stem), 'evaluations': result}]
  except Exception as e:
    raise HTTPException(status_code=404, detail="Evaluation: " + str(e)) 

@app.post("/evaluate/all")
async def start_evaluation_all(training: Training):
  folders = [
      r'E:\GitHub Repos\Masterarbeit\Evaluation_2\HR\11.06, 08.53Uhr\HRD.csv',
  ]

  for f_folder in folders:
    training.path = f_folder.replace('\\', '/')
    p = Path(training.path)

    folder_path = str(p.parent)
    file_name = str(p.stem)

    results = []

    for file in os.listdir(folder_path):
      p = Path(folder_path + '\\' + file)
      if str(p.stem).startswith(file_name + '_'):
        path = (str(p.parent) + '\\' + str(p.name)).replace('\\', '/')
        training.path_gen = path
        print(f'Current File: {file}')

        print(training.path)
        evaluator = Evaluator(training)
        results.append({'name': str(p.stem), 'evaluations': evaluator.run()})

    with open(folder_path + '\\evaluation3.json', 'w+') as outfile:
      json.dump(results, outfile)  

  return results

@app.post("/debug")
async def start_debug(training: Training):
  generators = ['TVAE', 'GaussianCopula', 'CTGAN', 'CopulaGAN']
  #sizes = [50, 100, 311, 622, 3110, 6220, 31100]
  sizes = [1000, 10000]
  results = []

  dt = datetime.now()
  folder_name = str(dt.strftime("%d.%m, %H.%M")) + 'Uhr'
  path = Path(training.path)

  new_dir = Path(str(path.parent), folder_name)
  new_dir.mkdir(parents=True, exist_ok=True)

  with open(str(new_dir) + '\\settings.json', 'w+') as outfile:
    json.dump(training.dict(), outfile)

  dc = DataConnector.load(path=training.path)
  tables, metadata = dc.get_training_data(training)
  real_data = dc.get_tables()
  
  for g in generators:
    training.tables[0].model = g

    gen = Generator(training, metadata)
    gen.fit(tables)

    for s in sizes:
      new_data = gen.sample(s, dc.get_column_names())
      gen.save(new_data, appendix=[gen._model_name, s, 'NP'], new_folder=folder_name)

      # Post Processing:
      for table_name, table_df in new_data.items():
        new_data[table_name] = PostGenFactory.apply(real_data[table_name], table_df, training, table_name)

      gen.save(new_data, appendix=[gen._model_name, s, 'PP'], new_folder=folder_name)

  for file in os.listdir(str(new_dir)):
    if "settings" in file:
      continue

    p = Path(str(new_dir) + '\\' + file)
    path = (str(p.parent) + '\\' + str(p.name)).replace('\\', '/')
    training.path_gen = path

    evaluator = Evaluator(training)
    results.append({'name': str(p.stem), 'evaluations': evaluator.run()})

  with open(str(new_dir) + '\\evaluation.json', 'w+') as outfile:
    json.dump(results, outfile)  

  return results

@app.get("/reset")
async def hard_reset():
  with open("reset.txt", "w+") as f:
    f.write(str(random.getrandbits(128)))

@app.get("/load/{save_file:path}")
async def load_model(save_file: str):
  with open(save_file,) as jfile:    
    jload = json.load(jfile)
  
  training = Training.parse_obj(jload)

  dc = DataConnector.load(path=training.path)  
  table_order, pk_relation, fk_relation = dc.get_schema()

  return {"db_path": save_file, "table_order": table_order, "pk_relation": pk_relation, "fk_relation": fk_relation, 'metadata': training, 'suggestions': []}


@app.get("/loadedModel/{load_path:path}/{amount:float}")
async def start_evaluation(load_path: str, amount: float):
#  try:
    with open(load_path,) as jfile:    
      jload = json.load(jfile)
    
    training = Training.parse_obj(jload)

    parentPath = Path(load_path).parents[0]
    fileCount = [y for y in parentPath.rglob(f'*')] 
    parentPath = str(parentPath)

    fileName = Path(training.path).name

    dc = DataConnector.load(parentPath + '\\' + fileName)
    tables = dc.get_tables()
    gen = Generator(training, None, parentPath+ '\\model.pkl')

    dc = DataConnector.load(parentPath + '\\' + fileName.split('.')[0] + '_gen.' + fileName.split('.')[1])
    tables_gen = dc.get_tables()

    new_data = gen.sample(int(len(tables[fileName.split('.')[0]]) * amount), list(tables_gen[fileName.split('.')[0] + '_gen'].columns))

    gen.save(new_data, [len(fileCount)], str(parentPath), absolut=True)
    return True
#  except Exception as e:
#    raise HTTPException(status_code=404, detail="LoadedModel: " + str(e)) 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
