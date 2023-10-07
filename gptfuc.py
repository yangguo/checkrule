# from langchain.llms import OpenAI
import json
import os

import openai

# import chromadb
# import faiss
import pandas as pd

# import streamlit as st
from dotenv import load_dotenv

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ErnieBotChat

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import (
    EmbaasEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    OpenAIEmbeddings,
)

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import Minimax, OpenAIChat
from langchain.prompts import load_prompt
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
from langchain.vectorstores import (
    FAISS,
    Chroma,
    Milvus,
    OpenSearchVectorSearch,
    Pinecone,
    Qdrant,
    SupabaseVectorStore,
)
from supabase.client import Client, create_client

# import pinecone


load_dotenv()


# from chromadb.utils import embedding_functions


# from qdrant_client.http.models import Filter, FieldCondition


api_key = os.environ.get("OPENAI_API_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# supabase: Client = create_client(supabase_url, supabase_key)


AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_TOKEN,
)
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_BASE"] = AZURE_BASE_URL
# os.environ["OPENAI_API_KEY"] = AZURE_API_KEY
# os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

# embeddings = OpenAIEmbeddings(
#     deployment="ada02",
#     model="text-embedding-ada-002",
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_type="azure",
#     openai_api_key=AZURE_API_KEY,
#     openai_api_version="2023-05-15",
# )

# embeddings = EmbaasEmbeddings(
#     model="paraphrase-multilingual-mpnet-base-v2",
#     instruction="",
# )

# openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
# openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
# openai.api_base = "https://az.139105.xyz/v1"
# openai.api_base = "https://op.139105.xyz/v1"


# llm = ChatOpenAI(model_name="gpt-3.5-turbo",
#                  openai_api_base="https://op.139105.xyz/v1",
#                  openai_api_key=api_key)

# llm = ChatOpenAI(model_name="gpt-3.5-turbo",
#                  openai_api_base="https://az.139105.xyz/v1",
#                  openai_api_key=AZURE_API_KEY)

# llm = ErnieBotChat(ernie_client_id='', ernie_client_secret='')

# llm = Minimax(minimax_api_key="", minimax_group_id="")

# use azure model
# llm = AzureChatOpenAI(
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_version="2023-07-01-preview",
#     deployment_name=AZURE_DEPLOYMENT_NAME,
#     # deployment_name=AZURE_DEPLOYMENT_NAME_16K,
#     # deployment_name=AZURE_DEPLOYMENT_NAME_GPT4,
#     # deployment_name=AZURE_DEPLOYMENT_NAME_GPT4_32K,
#     openai_api_key=AZURE_API_KEY,
#     openai_api_type = "azure",
# )

# convert gpt model name to azure deployment name
gpt_to_deployment = {
    "gpt-35-turbo": AZURE_DEPLOYMENT_NAME,
    "gpt-35-turbo-16k": AZURE_DEPLOYMENT_NAME_16K,
    "gpt-4": AZURE_DEPLOYMENT_NAME_GPT4,
    "gpt-4-32k": AZURE_DEPLOYMENT_NAME_GPT4_32K,
}

# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        openai_api_base=AZURE_BASE_URL,
        openai_api_version="2023-07-01-preview",
        deployment_name=deployment_name,
        openai_api_key=AZURE_API_KEY,
        openai_api_type="azure",
    )
    return llm


# use cohere model
# llm = Cohere(model="command-xlarge-nightly",cohere_api_key=COHERE_API_KEY)


# initialize pinecone
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


# @st.cache_resource
def init_supabase():
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase


supabase = init_supabase()


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
    # Qdrant.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     host=qdrant_host,
    #     collection_name=collection_name,
    # )

    # use pinecone
    # Create vector store from documents and save to pinecone
    # index_name = "ruledb"
    # Pinecone.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     namespace=collection_name,
    #     index_name=index_name,
    # )

    # use supabase
    # Create vector store from documents and save to supabase
    SupabaseVectorStore.from_texts(
        docs,
        embeddings,
        metadatas=metadata,
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
    )

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
    # index = pinecone.Index("ruledb")
    # store = Pinecone(
    #     index, embeddings.embed_query, text_key="text", namespace=collection_name
    # )

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings,
    )

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


def gpt_answer(
    question,
    chaintype="stuff",
    industry="",
    top_k=4,
    model_name="gpt-35-turbo",
    items=[],
):
    collection_name = industry_name_to_code(industry)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, OpenAIEmbeddings())

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host)

    # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=embeddings.embed_query)

    # embeddings = OpenAIEmbeddings()
    # get chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )

    # get pinecone
    # index = pinecone.Index("ruledb")
    # store = Pinecone(
    #     index, embeddings.embed_query, text_key="text", namespace=collection_name
    # )

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings,
    )

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

    # chat_template = load_prompt("prompt.json")
    # print(chat_template)
    # json_str=prompt.to_json()
    # print(json_str)
    # with open("prompt1.json", "w") as outfile:
    #     json.dump(json_str, outfile)

    # filter_value = {"监管要求": "信息技术管理办法"}

    filter = convert_list_to_dict(items)
    # retriever = store.as_retriever(search_kwargs={"k": top_k})
    retriever = store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k, "filter": filter}
    )

    chain = RetrievalQA.from_chain_type(
        # llm,
        get_azurellm(model_name),
        chain_type=chaintype,
        # vectorstore=store,
        retriever=retriever,
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
    # index = pinecone.Index("ruledb")
    # store = Pinecone(
    #     index, embeddings.embed_query, text_key="text", namespace=collection_name
    # )
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
    # print(collection_name)

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings,
    )
    # print(store.table_name)

    # List all collections
    # collections = store.list_collections()

    # print(collections)

    # filter_value = {"监管要求": "信息技术管理办法"}

    filter = convert_list_to_dict(items)
    print(filter)

    docs = store.similarity_search(query=question, k=topk, filter=filter)
    # retriever = store.as_retriever(search_type="similarity",search_kwargs={ "k":topk ,"filter":filter})
    # docs = retriever.get_relevant_documents(question)
    df = docs_to_df(docs)
    return df


def delete_db(industry="", items=[]):
    collection_name = industry_name_to_code(industry)

    filter = convert_list_to_dict(items)
    # convert dict to json
    filter_json = json.dumps(filter)
    # get pinecone
    # index = pinecone.Index("ruledb")
    # index.delete(filter=filter, namespace=collection_name)
    # print(filter)
    # print(filter_json)
    # delete all
    supabase.table(collection_name).delete().filter(
        "metadata", "cs", filter_json
    ).execute()


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
    # industry_name = industry_name.lower()
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
    elif industry_name == "投行":
        return "invbank"
    elif industry_name == "医药":
        return "pharma"
    else:
        return "other"


def convert_list_to_dict(lst):
    if len(lst) == 1:
        return {"监管要求": lst[0]}
    else:
        return {}
        # return {"监管要求": {"$in": [item for item in lst]}}
