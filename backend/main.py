import io
from typing import List, Union
from pydantic import BaseModel

import pandas as pd

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from gptfuncbk import similarity_search

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class InputData(BaseModel):
    query: str
    number: int
    option: str

@app.post("/search")
async def search(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option

    # Process the input data and generate a response
    result_df = similarity_search(query, topk=number, industry=option, items=[])

    response_data = result_df.to_dict(orient='records')

    return {"response_data": response_data}
