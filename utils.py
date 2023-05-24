import glob
import os

import pandas as pd

# from keybert import KeyBERT
# from sentence_transformers import SentenceTransformer
# from transformers import RoFormerModel, RoFormerTokenizer
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

# import streamlit as st
# import torch
# import asyncio
# import spacy
# from textrank4zh import TextRank4Sentence
# from sklearn.cluster import AgglomerativeClustering


# modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'

# tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
# model = RoFormerModel.from_pretrained(modelfolder)

# smodel = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# nlp = spacy.load('zh_core_web_lg')

rulefolder = "rules"


# def async sent2emb(sentences):
# def sent2emb_async(sentences):
#     """
#     run sent2emb in async mode
#     """
#     # create new loop
#     loop = asyncio.new_event_loop()
#     # run async code
#     asyncio.set_event_loop(loop)
#     # run code
#     task = loop.run_until_complete(sent2emb(sentences))
#     # close loop
#     loop.close()
#     return task


# async def sent2emb(sents):
#     embls = []
#     for sent in sents:
#         # get summary of sent
#         summarize = get_summary(sent)
#         sentence_embedding = roformer_encoder(summarize)
#         embls.append(sentence_embedding)
#     # count += 1
#     all_embeddings = np.concatenate(embls)
#     return all_embeddings


# get summary of text
# def get_summary(text):
#     tr4s = TextRank4Sentence()
#     tr4s.analyze(text=text, lower=True, source='all_filters')
#     sumls = []
#     for item in tr4s.get_key_sentences(num=3):
#         sumls.append(item.sentence)
#     summary = ''.join(sumls)
#     return summary


# Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     # First element of model_output contains all token embeddings
#     token_embeddings = model_output[0]
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(
#         token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1), min=1e-9)


# def roformer_encoder(sentences):
#     # Tokenize sentences
#     encoded_input = tokenizer(sentences,
#                               max_length=512,
#                               padding=True,
#                               truncation=True,
#                               return_tensors='pt')

#     # Compute token embeddings
#     with torch.no_grad():
#         model_output = model(**encoded_input)

#     # Perform pooling. In this case, max pooling.
#     sentence_embeddings = mean_pooling(
#         model_output, encoded_input['attention_mask']).numpy()
#     return sentence_embeddings


# @st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + "**/*.csv", recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = rule2df(filename, filepath)[["监管要求", "结构", "条款"]]
        dflist.append(newdf)
    alldf = pd.concat(dflist, axis=0)
    return alldf


def rule2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf["监管要求"] = filename
    return docdf


# def get_embedding(folder, emblist):
#     dflist = []
#     for file in emblist:
#         filepath = os.path.join(folder, file + '.npy')
#         embeddings = np.load(filepath)
#         dflist.append(embeddings)
#     alldf = np.concatenate(dflist)
#     return alldf


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new


# get section list from df
def get_section_list(searchresult, make_choice):
    """
    get section list from df

    args: searchresult, make_choice
    return: section_list
    """
    df = searchresult[(searchresult["监管要求"].isin(make_choice))]
    conls = df["结构"].drop_duplicates().tolist()
    unils = []
    # print(conls)
    for con in conls:
        itemls = con.split("/")
        #     print(itemls[:-1])
        for item in itemls[:2]:
            unils.append(item)
    # drop duplicates and keep list order
    section_list = list(dict.fromkeys(unils))
    return section_list


# conver items to cluster
# def items2cluster(df, threshold):
#     corpus = df['条款'].tolist()
#     # get embedding
#     embeddings = sent2emb_async(corpus)
#     # Normalize the embeddings to unit length
#     corpus_embeddings = embeddings / np.linalg.norm(
#         embeddings, axis=1, keepdims=True)

#     # Perform kmean clustering
#     clustering_model = AgglomerativeClustering(n_clusters=None,
#                                                distance_threshold=threshold)
#     #                                            affinity='cosine',
#     #                                            linkage='complete',
#     #                                            distance_threshold=0.5)
#     clustering_model.fit(corpus_embeddings)
#     cluster_assignment = clustering_model.labels_
#     clustered_sentences = {}
#     clustered_idlist = {}
#     for sentence_id, cluster_id in enumerate(cluster_assignment):
#         if cluster_id not in clustered_sentences:
#             clustered_sentences[cluster_id] = []
#             clustered_idlist[cluster_id] = []
#         clustered_sentences[cluster_id].append(corpus[sentence_id])
#         clustered_idlist[cluster_id].append(sentence_id)

#     # reset index
#     dfbefore = df.reset_index(drop=True)
#     for key, value in clustered_idlist.items():
#         dfbefore.loc[value, '分组'] = str(key)

#     dfsort = dfbefore.sort_values(by='分组')
#     clusternum = len(clustered_idlist.keys())
#     return dfsort, clusternum


# get folder name list from path
def get_folder_list(path):
    folder_list = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    return folder_list


def get_rulefolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(rulefolder, industry_choice)
    return folder


# cut text into words using spacy
# def cut_sentences(text):
#     # nlp = spacy.load('zh_core_web_trf')
#     # cut text into words
#     doc = nlp(text)
#     sents = [t.text for t in doc]
#     # sents = jieba.lcut(text,use_paddle=True)
#     return sents


# get keyword list using keybert
# def keybert_keywords(text, top_n=3):
#     doc = ' '.join(cut_sentences(text))
#     bertModel = KeyBERT(model=smodel)
#     # keywords = bertModel.extract_keywords(doc,keyphrase_ngram_range=(1,1),stop_words=None,top_n=top_n)
#     #mmr
#     keywords = bertModel.extract_keywords(doc,
#                                           keyphrase_ngram_range=(1, 1),
#                                           stop_words='english',
#                                           use_mmr=True,
#                                           diversity=0.7,
#                                           top_n=top_n)
#     keyls = []
#     for (key, val) in keywords:
#         keyls.append(key)
#     return keyls


# convert text spacy to word embedding
# def text2emb(text):
#     doc = nlp(text)
#     return doc


# find similar words in doc embedding
# def find_similar_words(words, doc, threshold_key=0.5, top_n=3):
#     # nlp = spacy.load('zh_core_web_trf')

#     # compute similarity
#     similarities = {}
#     for word in words:
#         tok = nlp(word)
#         similarities[tok.text] = {}
#         for tok_ in doc:
#             similarities[tok.text].update({tok_.text: tok.similarity(tok_)})
#     # sort
#     topk = lambda x: {
#         k: v
#         for k, v in sorted(similarities[x].items(),
#                            key=lambda item: item[1],
#                            reverse=True)[:top_n]
#     }
#     result = {word: topk(word) for word in words}
#     # filter by threshold
#     result_filter = {
#         word: {k: v
#                for k, v in result[word].items() if v >= threshold_key}
#         for word in result
#     }
#     return result_filter


# get similarity using keywords between two docs
# def get_similar_keywords(keyls, audit_list, key_top_n=3, threshold_key=0.5):

#     audit_keywords = dict()
#     # emptyls = []
#     for idx, audit in enumerate(audit_list):

#         doc = text2emb(audit)
#         result = find_similar_words(keyls, doc, threshold_key, top_n=key_top_n)
#         # st.write(result)
#         subls = []
#         for key in keyls:
#             subls.append(list(result[key].keys()))
#         # flatten subls
#         subls = [item for sub in subls for item in sub]
#         # remove duplicates
#         subls = list(set(subls))
#         # st.write(subls)
#         audit_keywords[idx] = subls

#         # get audit_keywords keys sorted by value length
#         audit_keywords_sorted = sorted(audit_keywords.items(),
#                                        key=lambda x: len(x[1]),
#                                        reverse=True)
#         # st.write(audit_keywords_sorted)
#         # get keys of audit_keywords_sorted if length > 0
#         audit_keywords_keys = [
#             key for key, value in audit_keywords_sorted if len(value) > 0
#         ]
#         # audit_keywords_keys = [key for key, value in audit_keywords_sorted]

#         # get audit_list using audit_keywords_keys
#         # audit_list_sorted = [audit_list[key] for key in audit_keywords_keys]
#     return audit_keywords_keys


# get most similar from list of sentences
# def get_most_similar(keyls, audit_list, top_n=3):
#     # st.write(keyls)
#     # st.write(audit_list)
#     # st.write(top_n)
#     audit_list_sorted = get_similar_keywords(keyls,
#                                              audit_list,
#                                              key_top_n=3,
#                                              threshold_key=0.5)
#     # st.write(audit_list_sorted)
#     return audit_list_sorted[:top_n]


# combine df columns into one field and return a list
def combine_df_columns(df, cols):
    df_combined = df[cols].apply(lambda x: " ".join(x), axis=1)
    # return list
    return df_combined.tolist()


def df2aggrid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    # gb.configure_auto_height()
    gb.configure_default_column(
        groupable=True,
        value=True,
        # resizable=True,
        # wrap_text=True,
        enableRowGroup=True,
        aggFunc="sum",
        editable=True,
    )
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    # configure column visibility
    gb.configure_column(field="lawid", hide=True)
    gb.configure_column(field="id", hide=True)

    gridOptions = gb.build()
    ag_grid = AgGrid(
        df,
        theme="blue",
        #  height=800,
        fit_columns_on_grid_load=True,  # fit columns to grid width
        gridOptions=gridOptions,  # grid options
        #  key='select_grid', # key is used to identify the grid
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        #  update_mode=GridUpdateMode.NO_UPDATE,
        enable_enterprise_modules=True,
    )
    return ag_grid
