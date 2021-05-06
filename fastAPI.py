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
    return training

  dc = DataConnector.load(path=training.path)
  tables, metadata = dc.get_training_data(training)
  
  gen = Generator(training, metadata)
  gen.fit(tables)

  new_data = gen.sample(311, dc.get_column_names())

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
  p = Path(training.path)

  folder_path = str(p.parent)
  file_name = str(p.stem)

  results = []

  for file in os.listdir(folder_path):
    p = Path(folder_path + '\\' + file)
    if str(p.stem).startswith(file_name + '_'):
      path = (str(p.parent) + '\\' + str(p.name)).replace('\\', '/')
      training.path_gen = path

      evaluator = Evaluator(training)
      results.append({'name': str(p.stem), 'evaluations': evaluator.run()})

  return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)