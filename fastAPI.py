import uvicorn

from fastapi import FastAPI, HTTPException
from models import Training, Table

from connector import DataConnector
from generator import Generator
from evaluator import Evaluator

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

@app.get("/schema/{db_path:path}")
async def get_database_schema(db_path: str):  
  try:
    dc = DataConnector.load(path=db_path)
  except Exception as e:
    raise HTTPException(status_code=404, detail=str(e)) 
  
  table_order, pk_relation, fk_relation = dc.get_schema()
  metadata = dc.get_metadata()

  return {"db_path": db_path, "table_order": table_order, "pk_relation": pk_relation, "fk_relation": fk_relation, 'metadata': metadata}


@app.post("/training/")
async def start_training(training: Training):
  if '_gen' in training.tables[0].name:
    return training

  dc = DataConnector.load(path=training.path)
  tables, metadata = dc.get_training_data(training)
  
  gen = Generator(training, metadata)
  gen.fit(tables)

  new_data = gen.sample(311)
  gen.save(new_data)

  return training

@app.post("/evaluate/")
async def start_evaluation(training: Training):
  evaluator = Evaluator(training)
  result = evaluator.run()

  return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)