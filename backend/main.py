import asyncio
import io
from datetime import date, datetime
from typing import List, Union

import pandas as pd
from dbcsrc2 import searchcsrc2
from dbpboc import searchpboc
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from gptfuncbk import gpt_answer, similarity_search
from pydantic import BaseModel

from checkrule import searchByItem, searchByName

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class InputData(BaseModel):
    query: str
    number: int
    option: str


class CsrcItem(BaseModel):
    filename: str = ""
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2023, 1, 1)
    wenhao: str = ""
    # caseval: str = ""
    org: List[str] = []
    people: str = ""
    event: str = ""
    penalty: str = ""
    legal_basis: str = ""
    violation_type: str = ""


class PbocItem(BaseModel):
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2023, 1, 1)
    wenhao_text: str = ""
    people_text: str = ""
    event_text: str = ""
    penalty_text: str = ""
    org_text: str = ""
    province: List[str] = []


@app.post("/search")
async def search(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option

    # Process the input data and generate a response
    result_df = await asyncio.to_thread(
        similarity_search, query, topk=number, industry=option, items=[]
    )

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}


@app.post("/gptanswer")
async def gptanswer(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option

    # Process the input data and generate a response
    answer, sourcedf = await asyncio.to_thread(
        gpt_answer, question=query, industry=option, top_k=number
    )

    source = sourcedf.to_dict(orient="records")

    return {"answer": answer, "source": source}


@app.post("/keywords")
async def keywords(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option

    ruledf, rulels = await asyncio.to_thread(searchByName, "", option)
    # Process the input data and generate a response
    result_df = await asyncio.to_thread(searchByItem, ruledf, rulels, "", query)

    if number != 0:
        result_df = result_df.head(number)

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}


# search csrc2 by filename, date, wenhao,case,org,law,label
@app.post("/pensearchcsrc2")
async def pensearchcsrc2(item: CsrcItem):
    filename = item.filename
    start_date = item.start_date
    end_date = item.end_date
    wenhao = item.wenhao
    # caseval = item.caseval
    org = item.org
    people = item.people
    event = item.event
    penalty = item.penalty
    legal_basis = item.legal_basis
    violation_type = item.violation_type

    result_df = await asyncio.to_thread(
        searchcsrc2,
        filename,
        start_date,
        end_date,
        wenhao,
        org,
        people,
        event,
        penalty,
        legal_basis,
        violation_type,
    )

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}


@app.post("/pensearchpboc")
async def pensearchpboc(item: PbocItem):
    start_date = item.start_date
    end_date = item.end_date
    wenhao_text = item.wenhao_text
    people_text = item.people_text
    event_text = item.event_text
    penalty_text = item.penalty_text
    org_text = item.org_text
    province = item.province

    # Process the input data and generate a response
    result_df = await asyncio.to_thread(
        searchpboc,
        start_date,
        end_date,
        wenhao_text,
        people_text,
        event_text,
        penalty_text,
        org_text,
        province,
    )

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}
