from fastapi import FastAPI
from models import Training, Table

from connector import DataConnector
from generator import Generator


app = FastAPI()

@app.get("/schema/{db_path:path}")
async def get_database_schema(db_path: str):
  dc = DataConnector.load(path=db_path)

  table_order, pk_relation, fk_relation = dc.get_schema()
  metadata = dc.get_metadata()

  return {"db_path": db_path, "table_order": table_order, "pk_relation": pk_relation, "fk_relation": fk_relation, 'metadata': metadata.to_dict()}


@app.post("/training/")
async def start_training(training: Training):
  dc = DataConnector.load(path=training.path)

  tables, metadata = dc.get_training_data(training)
  
  gen = Generator(training, metadata)
  gen.fit(tables)

  gen.sample(100)

  return training