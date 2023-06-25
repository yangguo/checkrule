from flask import Flask, request, jsonify
from flask.json import JSONEncoder
from datetime import datetime, date
from werkzeug.exceptions import BadRequest

from gptfuncbk import get_audit_steps, gpt_answer, similarity_search
from checkrule import searchByItem, searchByName

from dbcbirc import searchcbirc
from dbcsrc2 import searchcsrc2
from dbpboc import searchpboc


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, date):
                return obj.isoformat()
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

@app.route("/")
def read_root():
    return {"Hello": "World"}

@app.route('/search', methods=['POST'])
def search():
    try:
        input_data = request.get_json()
        query = input_data.get('query')
        number = input_data.get('number')
        option = input_data.get('option')

        result_df = similarity_search(query, topk=number, industry=option, items=[])

        response_data = result_df.to_dict(orient="records")
        return jsonify(response_data=response_data)
    except Exception as e:
        raise BadRequest(str(e))


@app.route("/keywords", methods=["POST"])
def keywords():
    try:
        input_data = request.get_json()
        query = input_data.get("query")
        number = input_data.get("number")
        option = input_data.get("option")

        ruledf, rulels = searchByName("", option)
        # Process the input data and generate a response

        result_df = searchByItem(ruledf, rulels, "", query)
        if number != 0:
            result_df = result_df.head(number)

        response_data = result_df.to_dict(orient="records")
        return jsonify(response_data=response_data)
    except Exception as e:
        raise BadRequest(str(e))





if __name__ == "__main__":
    app.run(debug=True)
