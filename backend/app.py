from flask import Flask, request, jsonify
import pandas as pd
from typing import List, Union
from pydantic import BaseModel
from gptfuncbk import similarity_search

app = Flask(__name__)

@app.route("/")
def read_root():
    return {"Hello": "World"}

@app.route("/search", methods=["POST"])
def search():
    input_data = request.get_json()
    query = input_data["query"]
    number = input_data["number"]
    option = input_data["option"]

    # Process the input data and generate a response
    result_df = similarity_search(query, topk=number, industry=option, items=[])

    response_data = result_df.to_dict(orient='records')

    return jsonify({"response_data": response_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
