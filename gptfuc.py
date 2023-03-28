# from langchain.llms import OpenAI
import json
import os
import pickle
from pathlib import Path

import faiss
import pandas as pd
import pinecone
import chromadb
from chromadb.config import Settings
# from chromadb.utils import embedding_functions

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma, Milvus, Pinecone, Qdrant,OpenSearchVectorSearch
from qdrant_client import QdrantClient
# from qdrant_client.http.models import Filter, FieldCondition

# from opensearchpy import OpenSearch
# import requests
# from llama_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader

# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

PINECONE_API_KEY = '***REMOVED***'
PINECONE_API_ENV = 'us-west1-gcp'

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

host = 'localhost'
port = 9200
auth = ('admin', 'admin') # For testing only. Don't store credentials in code.
# ca_certs_path = '/full/path/to/root-ca.pem' # Provide a CA bundle if you use intermediate CAs with your root CA.

# model_name='shibing624/text2vec-base-chinese'
# model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
embeddings = OpenAIEmbeddings()


# Create the client with SSL/TLS enabled, but hostname verification disabled.
# client = OpenSearch(
#     hosts = [{'host': host, 'port': port}],
#     http_compress = True, # enables gzip compression for request bodies
#     http_auth = auth,
#     use_ssl = True,
#     verify_certs = True,
#     ssl_assert_hostname = False,
#     ssl_show_warn = False,
#     # ca_certs = ca_certs_path
# )


# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

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


def build_ruleindex(df, industry=""):
    """
    Ingests data into LangChain by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """
    collection_name = industry_name_to_code(industry)

    # get text list from df
    docs = df["条款"].tolist()
    # build metadata
    metadata = df[["监管要求", "结构"]].to_dict(orient="records")
    # change the key names
    # for i in range(len(metadata)):
    #     metadata[i]["regulation"] = metadata[i].pop("监管要求")
    #     metadata[i]["structure"] = metadata[i].pop("结构")

    # embeddings = OpenAIEmbeddings()
    # embeddings =HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Create vector store from documents and save to disk
    # store = FAISS.from_texts(docs, embeddings,metadatas=metadata)
    # # store = FAISS.from_documents(docs, embeddings)
    # store.save_local(fileidxfolder)

    # use chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )

    # collections = store._client.list_collections()
    # for collection in collections:
    #     print(collection.name)
    # store.reset()
    # store.delete_collection()
    # store.persist()

    # store=Chroma.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     persist_directory=fileidxfolder,
    #     collection_name=collection_name,
    # )
    # store.persist()
    # store=None

    # use qdrant
    # collection_name = "filedocs"
    # Create vector store from documents and save to qdrant
    Qdrant.from_texts(docs, embeddings,metadatas=metadata, host=qdrant_host,  collection_name=collection_name)

    # use pinecone
    # Create vector store from documents and save to pinecone
    index_name = "ruledb"
    Pinecone.from_texts(docs, embeddings,metadatas=metadata,namespace=collection_name, index_name=index_name)

    # use milvus
    # vector_db = Milvus.from_texts(
    # docs,
    # embeddings,
    # connection_args={"host": "127.0.0.1", "port": "19530"},
    # metadatas=metadata,
    # collection_name=collection_name,
    # text_field="text",
    # )

    # use opensearch
    # docsearch = OpenSearchVectorSearch.from_texts(docs, embeddings, opensearch_url="http://localhost:9200")

# def split_text(text, chunk_chars=4000, overlap=50):
#     """
#     Pre-process text file into chunks
#     """
#     splits = []
#     for i in range(0, len(text), chunk_chars - overlap):
#         splits.append(text[i : i + chunk_chars])
#     return splits


# create function to add new documents to the index
def add_ruleindex(df, industry=""):
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
    # qdrant_client = QdrantClient(host=qdrant_host)
    # # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=embeddings.embed_query)

    # Create vector store from documents and save to disk
    # store.add_documents(docs)
    # store.save_local(fileidxfolder)

    # get pinecone
    index=pinecone.Index("ruledb")
    store = Pinecone(index, embeddings.embed_query,text_key="text",namespace=collection_name)

    # get text list from df
    docs = df["条款"].tolist()
    # build metadata
    metadata = df[["监管要求", "结构"]].to_dict(orient="records")

    # get chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )
    # add to chroma
    store.add_texts(docs, metadatas=metadata)
    # store.persist()


# list all indexes using qdrant
# def list_indexes():
#     """
#     Lists all indexes in the LangChain index.
#     """

#     # get qdrant client
#     qdrant_client = QdrantClient(host=qdrant_host)
#     # get collection names
#     collection_names = qdrant_client.list_aliases()
#     return collection_names


def gpt_answer(question, chaintype="stuff", industry="", top_k=4,model_name="gpt-3.5-turbo"):
    collection_name = industry_name_to_code(industry)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, OpenAIEmbeddings())

    # get qdrant client
    qdrant_client = QdrantClient(host=qdrant_host)

    # get qdrant docsearch
    store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=embeddings.embed_query)

    # embeddings = OpenAIEmbeddings()
    # get chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )

    # prefix_messages = [
    #     {
    #         "role": "system",
    #         # "content": "You are a helpful assistant that is very good at problem solving who thinks step by step.",
    #         "content": "你是一位十分善于解决问题、按步骤思考的专业顾问。",
    #     }
    # ]
    # llm = OpenAIChat(temperature=0, prefix_messages=prefix_messages)

    system_template = """Use the following pieces of context to answer the users question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}
    llm = ChatOpenAI(model_name=model_name,max_tokens=512)
    # chain = VectorDBQA.from_chain_type(
    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type=chaintype,
        # vectorstore=store,
        retriever=store.as_retriever(),
        # k=top_k,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    result = chain({"query": question})

    # docs = store.similarity_search(question)
    # qa_chain=load_qa_with_sources_chain(llm,chain_type=chaintype)
    # qa_chain=load_qa_chain(llm,chain_type=chaintype)
    # result = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    answer = result["result"]
    # sourcedf=None
    source = result["source_documents"]
    sourcedf = docs_to_df(source)
    return answer, sourcedf


def similarity_search(question, topk=4, industry="", items=[]):
    collection_name = industry_name_to_code(industry)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, embeddings)

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host)
    # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=embeddings.embed_query)
 
    # get pinecone
    index=pinecone.Index("ruledb")
    store = Pinecone(index, embeddings.embed_query,text_key="text",namespace=collection_name)
    # get chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )

    # collections = store._client.list_collections()
    # for collection in collections:
    #     print(collection.name)
    
    # # get milvus
    # store = Milvus(
    # embedding_function=OpenAIEmbeddings(),
    # connection_args={"host": "127.0.0.1", "port": "19530"},
    # # collection_name=collection_name,
    # # text_field="text",
    # )

    # List all collections
    # collections = store.list_collections()

    # print(collections)

    filter = {"监管要求": "商业银行信息科技风险管理指引"}
#     filter ={
#   "key": "监管要求",
#   "type": "string",
#   "match": {
#     "value": "商业银行信息科技风险管理指引"
#   }
# }

    filter = convert_list_to_dict(items)

    # substore=collection.query(["query text"], {"where": flter})
    print(filter)
    # filter=None
    docs = store.similarity_search(query=question, k=topk,filter=filter)
    df = docs_to_df(docs)
    # df=None
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
        plc = metadata["监管要求"]
        sec = metadata["结构"]
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
        return {"监管要求": {"$in":[item for item in lst]}}
