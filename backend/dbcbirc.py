from datetime import datetime

import pandas as pd
from database import get_collection


def searchcbirc(
    start_date: str,
    end_date: str,
    wenhao_text: str,
    people_text: str,
    event_text: str,
    law_text: str,
    penalty_text: str,
    org_text: str,
):
    cols = [
        "标题",
        "文号",
        "发布日期",
        "行政处罚决定书文号",
        "被处罚当事人",
        "主要违法违规事实",
        "行政处罚依据",
        "行政处罚决定",
        "作出处罚决定的机关名称",
        "作出处罚决定的日期",
        "id",
    ]

    # split words
    if wenhao_text != "":
        wenhao_text = split_words(wenhao_text)
    if people_text != "":
        people_text = split_words(people_text)
    if event_text != "":
        event_text = split_words(event_text)
    if law_text != "":
        law_text = split_words(law_text)
    if penalty_text != "":
        penalty_text = split_words(penalty_text)
    if org_text != "":
        org_text = split_words(org_text)

    collection = get_collection("pencbirc", "cbircanalysis")

    query = {
        "发布日期": {
            "$gte": datetime.combine(start_date, datetime.min.time()),
            "$lte": datetime.combine(end_date, datetime.max.time()),
        }
    }
    if wenhao_text != "":
        query["行政处罚决定书文号"] = {"$regex": wenhao_text}
    if people_text != "":
        query["被处罚当事人"] = {"$regex": people_text}
    if event_text != "":
        query["主要违法违规事实"] = {"$regex": event_text}
    if law_text != "":
        query["行政处罚依据"] = {"$regex": law_text}
    if penalty_text != "":
        query["行政处罚决定"] = {"$regex": penalty_text}
    if org_text != "":
        query["作出处罚决定的机关名称"] = {"$regex": org_text}

    print(query)
    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))
    if not df.empty:
        searchdf = df[cols]
        # fillna
        searchdf = searchdf.fillna("")
        # format date
        searchdf.loc[:, "发布日期"] = pd.to_datetime(searchdf["发布日期"]).dt.date
        # sort by date desc
        searchdf = searchdf.sort_values(by=["发布日期"], ascending=False)
        # reset index
        searchdf.reset_index(drop=True, inplace=True)
    else:
        searchdf = pd.DataFrame(columns=cols)
    return searchdf


def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new
