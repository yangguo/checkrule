# from langchain.llms import OpenAI
import json
import os
import pickle
from pathlib import Path
import pandas as pd
import faiss
import pinecone

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import VectorDBQA
from langchain.chains.question_answering import load_qa_chain

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma, Pinecone, Qdrant,Milvus
from qdrant_client import QdrantClient

# import requests
# from llama_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader


# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

# PINECONE_API_KEY = ''
# PINECONE_API_ENV = 'us-west1-gcp'

qdrant_host = "127.0.0.1"
# qdrant_api_key = ""


os.environ["OPENAI_API_KEY"] = api_key

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "ruleidx"
backendurl = "http://localhost:8000"

openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    print("请设置OPENAI_API_KEY")
else:
    print("已设置OPENAI_API_KEY" + openai_api_key)

# initialize pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_API_ENV
# )

# gpt_model="text-davinci-003"
# gpt_model='gpt-3.5-turbo'


# llm_predictor = LLMPredictor(
#     llm=OpenAI(temperature=0, model_name=gpt_model, max_tokens=1024)
# )

# use ChatGPT [beta]
# from gpt_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor

# llm_predictor = ChatGPTLLMPredictor()


# def build_index():
#     documents = SimpleDirectoryReader(filerawfolder, recursive=True).load_data()
#     # index = GPTSimpleVectorIndex(documents)
#     index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor)
#     index.save_to_disk(os.path.join(fileidxfolder, "filedata.json"))


# def gpt_answer(question):
#     filepath = os.path.join(fileidxfolder, "filedata.json")
#     index = GPTSimpleVectorIndex.load_from_disk(filepath, llm_predictor=llm_predictor)

#     # prompt = f'You are a helpful support agent. You are asked: "{question}". Try to use only the information provided. Format your answer nicely as a Markdown page.'
#     prompt = f'您是一位专业顾问。您被问到："{question}"。请尽可能使用提供的信息。'
#     # response = index.query(prompt).response.strip()
#     response=index.query(prompt,llm_predictor=llm_predictor)
#     return response


def build_ruleindex(df,industry=""):
    """
    Ingests data into LangChain by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """
    collection_name = industry_name_to_code(industry)

    # get text list from df
    docs = df["条款"].tolist()
    # build metadata
    metadata = df[['监管要求','结构']].to_dict(orient="records")

    embeddings = OpenAIEmbeddings()
    # Create vector store from documents and save to disk
    # store = FAISS.from_texts(docs, embeddings,metadatas=metadata)
    # # store = FAISS.from_documents(docs, embeddings)
    # store.save_local(fileidxfolder)


    # use chroma
    store = Chroma(persist_directory=fileidxfolder, embedding_function=OpenAIEmbeddings(),collection_name=collection_name)
    store.delete_collection()

    store = Chroma.from_texts(docs, embeddings,metadatas=metadata,persist_directory=fileidxfolder,collection_name=collection_name)
    store.persist()
    # store=None

    # use qdrant
    # collection_name = "filedocs"
    # # Create vector store from documents and save to qdrant
    # Qdrant.from_documents(docs, embeddings, host=qdrant_host, prefer_grpc=True, collection_name=collection_name)

    # use pinecone
    # Create vector store from documents and save to pinecone
    # index_name = "langchain1"
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    # return docsearch

    # use milvus
    # vector_db = Milvus.from_texts(
    # docs,
    # embeddings,
    # connection_args={"host": "127.0.0.1", "port": "19530"},
    # # metadatas=metadata
    # )


# def split_text(text, chunk_chars=4000, overlap=50):
#     """
#     Pre-process text file into chunks
#     """
#     splits = []
#     for i in range(0, len(text), chunk_chars - overlap):
#         splits.append(text[i : i + chunk_chars])
#     return splits


# create function to add new documents to the index
def add_ruleindex(df,industry=""):
    """
    Adds new documents to the LangChain index by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """
    collection_name = industry_name_to_code(industry)
    # loader = DirectoryLoader(filerawfolder, glob="**/*.txt")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # use tiktoken
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    # print("docs",docs)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, OpenAIEmbeddings())

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host, prefer_grpc=True)
    # collection_name = "filedocs"
    # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=OpenAIEmbeddings().embed_query)

    # Create vector store from documents and save to disk
    # store.add_documents(docs)
    # store.save_local(fileidxfolder)

    # get text list from df
    docs = df["条款"].tolist()
    # build metadata
    metadata = df[['监管要求','结构']].to_dict(orient="records")

    embeddings = OpenAIEmbeddings()

    # get chroma
    store = Chroma(persist_directory=fileidxfolder, embedding_function=embeddings,collection_name=collection_name)
    # add to chroma
    store.add_texts(docs,metadatas=metadata)
    store.persist()

# list all indexes using qdrant
def list_indexes():
    """
    Lists all indexes in the LangChain index.
    """

    # get qdrant client
    qdrant_client = QdrantClient(host=qdrant_host)
    # get collection names
    collection_names = qdrant_client.list_aliases()
    return collection_names


def gpt_answer(question, chaintype="stuff",industry=""):
    collection_name = industry_name_to_code(industry)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, OpenAIEmbeddings())

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host, prefer_grpc=True)

    # collection_name = "filedocs"
    # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=OpenAIEmbeddings().embed_query)

    embeddings=OpenAIEmbeddings()
    # get chroma
    store = Chroma(persist_directory=fileidxfolder, embedding_function=embeddings,collection_name=collection_name)

    prefix_messages = [
        {
            "role": "system",
            # "content": "You are a helpful assistant that is very good at problem solving who thinks step by step.",
            "content": "你是一位十分善于解决问题、按步骤思考的专业顾问。",
        }
    ]
    llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)

    chain = VectorDBQA.from_chain_type(llm, chain_type=chaintype, vectorstore=store)

    result = chain.run(question)

    return result


def similarity_search(question,topk=4,industry="",items=[]):
    collection_name = industry_name_to_code(industry)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, OpenAIEmbeddings())

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host, prefer_grpc=True)
    # collection_name = "filedocs"
    # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=OpenAIEmbeddings().embed_query)
    # get chroma
    store = Chroma(persist_directory=fileidxfolder, embedding_function=OpenAIEmbeddings(),collection_name=collection_name)

    # get milvus
    # store = Milvus(
    # embedding_function=OpenAIEmbeddings(),
    # connection_args={"host": "127.0.0.1", "port": "19530"},
    # )
    filter={
    "监管要求": {
        "$eq": "商业银行信息科技风险管理指引"
    }
    }
    # filter={
    # "$or": [
    #     {
    #         "监管要求": "银行业金融机构重要信息系统投产及变更管理办法"
    #     },
    #     {
    #         "监管要求": "商业银行信息科技风险管理指引"
    #     }
    # ]
    # }
    filter = convert_list_to_dict(items)
    # print(filter)
    docs = store.similarity_search(question,k=topk,filter=filter)
    df = docs_to_df(docs)
    return df


# convert document list to pandas dataframe
def docs_to_df(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc=metadata['监管要求']
        sec=metadata['结构']
        row = {"条款": page_content, "监管要求": plc, "结构": sec}
        data.append(row)
    df = pd.DataFrame(data)
    return df
    

# convert industry chinese name to english name
def industry_name_to_code(industry_name):
    """
    Converts an industry name to an industry code.
    """
    industry_name = industry_name.lower()
    if industry_name == "银行":
        return "bank"
    elif industry_name == "保险":
        return "insurance"
    elif industry_name == "证券":
        return "securities"
    elif industry_name == "基金":
        return "fund"
    elif industry_name == "期货":
        return "futures"
    else:
        return "other"


def convert_list_to_dict(lst):
    if len(lst) == 1:
        return {"监管要求": lst[0]}
    else:
        return {"$or": [{"监管要求": item} for item in lst]}