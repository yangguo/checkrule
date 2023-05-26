import io
from typing import List, Union

import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from gptfuncbk import gpt_answer, similarity_search
from pydantic import BaseModel
from checkrule import searchByItem,searchByName
import asyncio

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
    result_df = await asyncio.to_thread(similarity_search,query, topk=number, industry=option, items=[])

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}


@app.post("/gptanswer")
async def gptanswer(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option

    # Process the input data and generate a response
    answer, sourcedf = await asyncio.to_thread(gpt_answer,question=query, industry=option, top_k=number)

    source = sourcedf.to_dict(orient="records")

    return {"answer": answer, "source": source}


@app.post("/keywords")
async def keywords(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option

    ruledf,rulels= await asyncio.to_thread(searchByName,"", option)
    # Process the input data and generate a response
    result_df = await asyncio.to_thread(searchByItem,ruledf, rulels, '', query)

    if number != 0:
        result_df = result_df.head(number)

    response_data = result_df.to_dict(orient="records")

    
    return {"response_data": response_data}
