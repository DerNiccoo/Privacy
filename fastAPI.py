import uvicorn
import os

from fastapi import FastAPI, HTTPException
from models import Training, Table
from pathlib import Path

from connector import DataConnector
from generator import Generator
from evaluator import Evaluator
from suggestions import SuggestionFactory
from postgenerator import PostGenFactory

from pathlib import Path
from datetime import datetime
import json

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
    dc = DataConnector.load(path=db_path)
  except Exception as e:
    raise HTTPException(status_code=404, detail=str(e)) 
  
  table_order, pk_relation, fk_relation = dc.get_schema()
  metadata = dc.get_metadata()

  suggestions = SuggestionFactory.create(dc.get_tables(replace_na = False), metadata)

  return {"db_path": db_path, "table_order": table_order, "pk_relation": pk_relation, "fk_relation": fk_relation, 'metadata': metadata, 'suggestions': suggestions}


@app.post("/training/")
async def start_training(training: Training):
  if '_gen' in training.tables[0].name:
    training.path_gen = training.path
    return training

  dc = DataConnector.load(path=training.path)
  tables, metadata = dc.get_training_data(training)
  
  gen = Generator(training, metadata)
  gen.fit(tables)

  real_data = dc.get_tables()
  length = len(real_data[training.tables[0].name]) # * 10

  new_data = gen.sample(length, dc.get_column_names())

  # Post Processing:
  real_data = dc.get_tables()

  for table_name, table_df in new_data.items():
    new_data[table_name] = PostGenFactory.apply(real_data[table_name], table_df, training, table_name)

  gen.save(new_data)

  if debug:
    for size in sizes:
      new_data = gen.sample(size, dc.get_column_names())
      gen.save(new_data, size)      

  return training

@app.post("/evaluate/")
async def start_evaluation(training: Training):
  p = Path(training.path)
  evaluator = Evaluator(training)
  result = evaluator.run()
  
  return [{'name': str(p.stem), 'evaluations': result}]

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
