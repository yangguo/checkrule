import asyncio
from datetime import date
from typing import List

from dbcbirc import searchcbirc
from dbcsrc2 import searchcsrc2
from dbpboc import searchpboc
from fastapi import FastAPI
from gptfuncbk import get_audit_steps, gpt_answer, similarity_search
from pydantic import BaseModel

from checkrule import searchByIndustrysupa, searchByItem, searchByName

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class InputData(BaseModel):
    query: str
    number: int
    option: str
    make_choice: List[str] = []


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


class CbircItem(BaseModel):
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2023, 1, 1)
    wenhao_text: str = ""
    people_text: str = ""
    event_text: str = ""
    law_text: str = ""
    penalty_text: str = ""
    org_text: str = ""


class AuditRequest(BaseModel):
    query: str = ""


class IndustryInput(BaseModel):
    industry_choice: str = ""


# class InputData(BaseModel):
#     point: str
#     params: dict = {}


# @app.post("/testing")
# async def dify_receive(data: InputData = Body(...), authorization: str = Header(None)):
#     """
#     Receive API query data from Dify.
#     """
#     expected_api_key = "123456"  # TODO Your API key of this API
#     auth_scheme, _, api_key = authorization.partition(" ")

#     if auth_scheme.lower() != "bearer" or api_key != expected_api_key:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     print(data)

#     point = data.point

#     # for debug
#     print(f"point: {point}")

#     if point == "ping":
#         return {"result": "pong"}

#     elif point == "keywords":
#         return await keywords_dify(params=data.params)

#     raise HTTPException(status_code=400, detail="Not implemented")


# async def keywords_dify(params: dict):
#     inputs = params.get("inputs", {})

#     query = inputs.get("query", "")
#     number = inputs.get("number", 0)
#     option = inputs.get("option", "")
#     make_choice = inputs.get("make_choice", [])
#     choicels = await asyncio.to_thread(searchByIndustrysupa, option)
#     if make_choice == []:
#         make_choice = choicels

#     ruledf, rulels = await asyncio.to_thread(searchByName, "", option)
#     # Process the input data and generate a response
#     result_df = await asyncio.to_thread(searchByItem, ruledf, make_choice, "", query)

#     if number != 0:
#         result_df = result_df.head(number)

#     response_data = result_df.to_dict(orient="records")

#     return {"response_data": response_data}


@app.post("/search")
async def search(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option
    make_choice = input_data.make_choice

    # Process the input data and generate a response
    result_df = await asyncio.to_thread(
        similarity_search, query, topk=number, industry=option, items=make_choice
    )

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}


@app.post("/gptanswer")
async def gptanswer(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option
    make_choice = input_data.make_choice

    # print input data
    print(query, number, option)
    # Process the input data and generate a response
    answer, sourcedf = await asyncio.to_thread(
        gpt_answer, question=query, industry=option, top_k=number, items=make_choice
    )
    # print answer
    print(answer)
    # print sourcedf
    print(sourcedf)
    # convert sourcedf to dict
    source = sourcedf.to_dict(orient="records")

    return {"answer": answer, "source": source}


@app.post("/keywords")
async def keywords(input_data: InputData):
    query = input_data.query
    number = input_data.number
    option = input_data.option
    make_choice = input_data.make_choice

    ruledf, rulels = await asyncio.to_thread(searchByName, "", option)
    # Process the input data and generate a response
    result_df = await asyncio.to_thread(searchByItem, ruledf, make_choice, "", query)

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


@app.post("/pensearchcbirc")
async def pensearchcbirc(item: CbircItem):
    start_date = item.start_date
    end_date = item.end_date
    wenhao_text = item.wenhao_text
    people_text = item.people_text
    event_text = item.event_text
    law_text = item.law_text
    penalty_text = item.penalty_text
    org_text = item.org_text

    result_df = await asyncio.to_thread(
        searchcbirc,
        start_date,
        end_date,
        wenhao_text,
        people_text,
        event_text,
        law_text,
        penalty_text,
        org_text,
    )

    response_data = result_df.to_dict(orient="records")

    return {"response_data": response_data}


@app.post("/audit_steps")
async def generate_audit_steps(audit_request: AuditRequest):
    # print input data
    print(audit_request.query)
    try:
        audit_response = await asyncio.to_thread(get_audit_steps, audit_request.query)
        # print audit_response
        print(audit_response)
        return {"status": "success", "audit_response": audit_response}
    except Exception as e:
        # print error
        print(e)
        return {"status": "error", "message": str(e)}


@app.post("/plclist")
async def get_industry_list(input_data: IndustryInput):
    choicels = await asyncio.to_thread(searchByIndustrysupa, input_data.industry_choice)
    return {"choicels": choicels}
