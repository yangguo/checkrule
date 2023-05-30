from datetime import datetime
from typing import List

import pandas as pd
from database import get_collection, get_data


# search by filename, date, wenhao,org,law,label
def searchcsrc2(
    filename: str,
    start_date: str,
    end_date: str,
    wenhao: str,
    org: List[str],
    people: str = "",
    event: str = "",
    penalty: str = "",
    legal_basis: str = "",
    violation_type: str = "",
):
    col = [
        "名称",
        "发文日期",
        "文号",
        "内容",
        "链接",
        "机构",
        "当事人",
        "违法违规事实",
        "处罚结果",
        "处罚依据",
        "违法违规类型",
        "监管部门",
        "处罚时间",
        "罚款总金额",
        "没收总金额",
    ]

    # split words
    if filename != "":
        filename = split_words(filename)
    if wenhao != "":
        wenhao = split_words(wenhao)
    if people != "":
        people = split_words(people)
    if event != "":
        event = split_words(event)
    if penalty != "":
        penalty = split_words(penalty)
    if legal_basis != "":
        legal_basis = split_words(legal_basis)
    if violation_type != "":
        violation_type = split_words(violation_type)

    collection = get_collection("pencsrc2", "csrc2analysis")

    query = {
        "发文日期": {
            "$gte": datetime.combine(start_date, datetime.min.time()),
            "$lte": datetime.combine(end_date, datetime.max.time()),
        }
    }
    if filename != "":
        query["名称"] = {"$regex": filename}
    if wenhao != "":
        query["文号"] = {"$regex": wenhao}
    if people != "":
        query["当事人"] = {"$regex": people}
    if event != "":
        query["违法违规事实"] = {"$regex": event}
    if penalty != "":
        query["处罚结果"] = {"$regex": penalty}
    if legal_basis != "":
        query["处罚依据"] = {"$regex": legal_basis}
    if violation_type != "":
        query["违法违规类型"] = {"$regex": violation_type}
    if org:
        query["机构"] = {"$in": org}

    print(query)
    # execute the query and fetch the results
    cursor = collection.find(query)
    # searchdf = pd.DataFrame(list(cursor))[col]
    df = pd.DataFrame(list(cursor))
    if not df.empty:
        print(df.columns)  # prints the actual column names in the dataframe
        searchdf = df[col]
        # fillna
        searchdf = searchdf.fillna("")
        # format date
        searchdf.loc[:, "发文日期"] = pd.to_datetime(searchdf["发文日期"]).dt.date
        # sort by date desc
        searchdf = searchdf.sort_values(by=["发文日期"], ascending=False)
        # reset index
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
