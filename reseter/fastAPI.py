import uvicorn
import os

from fastapi import FastAPI, HTTPException, Query
from pathlib import Path

import random

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://localhost:4200",
    "http://localhost:4200",
    "https://0.0.0.0:4200",
    "http://0.0.0.0:4200",
    "https://172.24.105.27:4200",
    "http://172.24.105.27:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/reset")
async def hard_reset():
  with open("reset.py", "w+") as f:
    f.write(str(random.getrandbits(128)))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
