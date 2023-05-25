# import scipy
import ast
import json
import os
import pandas as pd
from supabase import Client, create_client

from gptfuncbk import industry_name_to_code


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)



def searchByName(search_text, industry_choice):

    table_name = industry_name_to_code(industry_choice)

    # print(table_name)
    # Get all records from table and cast 'metadata' to text type
    result = supabase.table(table_name).select("content, metadata").execute()

    # print(result.data)
    # Convert the results to a DataFrame
    df = pd.json_normalize(result.data)
    df.columns = ["条款", "结构", "监管要求"]
    # print(df)
    # Filter DataFrame based on conditions
    filtered_results = df[df["监管要求"].str.contains(f".*{search_text}.*")]

    choicels = filtered_results["监管要求"].unique().tolist()

    return filtered_results, choicels



def searchByItem(searchresult, make_choice, column_text, item_text):

    # split words item_text
    item_text_list = split_words(item_text)
    column_text = fix_section_text(column_text)
    plcsam = searchresult[
        (searchresult["监管要求"].isin(make_choice))
        & (searchresult["结构"].str.contains(column_text))
        & (searchresult["条款"].str.contains(item_text_list))
    ]
    return plcsam


# fix section text with +
def fix_section_text(section_text):
    if "+" in section_text:
        section_text = section_text.replace("+", "\\+")
    return section_text


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new
