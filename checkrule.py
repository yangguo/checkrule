import os
import glob
import pandas as pd
import streamlit as st
import scipy
import numpy as np
from utils import split_words, roformer_encoder,get_csvdf,get_rulefolder

rulefolder='rules'

def rule2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf['监管要求'] = filename
    return docdf


def get_embedding(rulefolder, emblist):
    dflist = []
    for file in emblist:
        filepath = os.path.join(rulefolder, file + '.npy')
        embeddings = np.load(filepath)
        dflist.append(embeddings)
    alldf = np.concatenate(dflist)
    return alldf


def get_rule_data(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)

    selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
    emblist = selectdf['监管要求'].unique().tolist()
    rule_encode = get_embedding(rulefolder, emblist)
    return selectdf, rule_encode


def get_samplerule(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
    tb_sample = selectdf[['监管要求', '结构', '条款']]
    return tb_sample.reset_index(drop=True)


def searchrule(text, column_text, make_choice, industry_choice, top):
    queries = [text]
    query_embeddings = roformer_encoder(queries)

    searchdf = get_samplerule(make_choice, industry_choice)
    # search rule
    ruledf, _ = searchByItem(searchdf, make_choice, column_text, '')
    rulefolder = get_rulefolder(industry_choice)
    emblist = ruledf['监管要求'].drop_duplicates().tolist()
    subsearchdf = get_samplerule(emblist, industry_choice)
    # fix index
    fixruledf, _ = searchByItem(subsearchdf, emblist, column_text, '')
    # get index of rule
    rule_index = fixruledf.index.tolist()

    sentence_embeddings = get_embedding(rulefolder, emblist)
    # get sub embedding
    sub_embedding = sentence_embeddings[rule_index]

    avglist = []
    idxlist = []
    number_top_matches = top
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding],
                                                 sub_embedding, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:number_top_matches]:
            idxlist.append(idx)
            avglist.append(1 - distance)

    return fixruledf.iloc[idxlist]


def getresultls(results):
    resultnum = 3
    resultls = []
    for i in range(resultnum):
        resultstr = results["labels"][i] + "{0:.2f}".format(
            results["scores"][i])
        resultls.append(resultstr)
    return '| '.join(resultls)


def searchByName(search_text, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf['监管要求'].drop_duplicates().tolist()

    choicels = []
    for plc in plc_list:
        if search_text in plc:
            choicels.append(plc)

    plcsam = get_samplerule(choicels, industry_choice)

    return plcsam, choicels


def searchByItem(searchresult, make_choice, column_text, item_text):

    # split words item_text
    item_text_list = split_words(item_text)
    column_text = fix_section_text(column_text)
    plcsam = searchresult[(searchresult['监管要求'].isin(make_choice))
                          & (searchresult['结构'].str.contains(column_text)) &
                          (searchresult['条款'].str.contains(item_text_list))]
    total = len(plcsam)
    return plcsam, total


# fix section text with +
def fix_section_text(section_text):
    if '+' in section_text:
        section_text = section_text.replace('+', '\\+')
    return section_text
