from datetime import datetime
from typing import List

import pandas as pd
from database import get_collection


def searchpboc(
    start_date: str,
    end_date: str,
    wenhao_text: str,
    people_text: str,
    event_text: str,
    penalty_text: str,
    org_text: str,
    province: List[str],
):
    # split words
    if wenhao_text != "":
        wenhao_text = split_words(wenhao_text)
    if people_text != "":
        people_text = split_words(people_text)
    if event_text != "":
        event_text = split_words(event_text)
    if penalty_text != "":
        penalty_text = split_words(penalty_text)
    if org_text != "":
        org_text = split_words(org_text)

    # print all input text
    # print("wenhao_text:", wenhao_text)
    # print("people_text:", people_text)
    # print("event_text:", event_text)
    # print("penalty_text:", penalty_text)
    # print("org_text:", org_text)
    # print("province:", province)
    # print("start_date:", start_date)
    # print("end_date:", end_date)

    # Define the columns
    col = [
        "企业名称",
        "处罚决定书文号",
        "违法行为类型",
        "行政处罚内容",
        "作出行政处罚决定机关名称",
        "作出行政处罚决定日期",
        "备注",
        "区域",
        "link",
        "发布日期",
        "name",
    ]

    # Connect to your MongoDB
    collection = get_collection("penpboc", "pbocdtl")

    # Build the MongoDB query dynamically
    query = {
        "发布日期": {
            "$gte": datetime.combine(start_date, datetime.min.time()),
            "$lte": datetime.combine(end_date, datetime.max.time()),
        }
    }
    if wenhao_text != "":
        query["处罚决定书文号"] = {"$regex": wenhao_text}
    if people_text != "":
        query["企业名称"] = {"$regex": people_text}
    if event_text != "":
        query["违法行为类型"] = {"$regex": event_text}
    if penalty_text != "":
        query["行政处罚内容"] = {"$regex": penalty_text}
    if org_text != "":
        query["作出行政处罚决定机关名称"] = {"$regex": org_text}
    if province:
        query["区域"] = {"$in": province}

    print("query:", query)
    # Execute the query and fetch the results
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    # print(df.head)
    if not df.empty:
        # Select only the desired columns
        searchdf = df[col]
        # fillna
        searchdf = searchdf.fillna("")
        # Format date
        searchdf.loc[:, "发布日期"] = pd.to_datetime(searchdf["发布日期"]).dt.date
        # Sort by date desc
        searchdf = searchdf.sort_values(by=["发布日期"], ascending=False)
        # Reset index
        searchdf.reset_index(drop=True, inplace=True)
    else:
        searchdf = pd.DataFrame(columns=col)

    return searchdf


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new
