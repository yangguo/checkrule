import json
import os

import pandas as pd
import pinecone

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAIChat
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import (
    FAISS,
    Chroma,
    Milvus,
    OpenSearchVectorSearch,
    Pinecone,
    Qdrant,
)

# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

PINECONE_API_KEY = config["pinecone_api_key"]
PINECONE_API_ENV = config["pinecone_api_env"]


os.environ["OPENAI_API_KEY"] = api_key

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "ruleidx"
backendurl = "http://localhost:8000"

# openai_api_key = os.environ.get("OPENAI_API_KEY")
# if openai_api_key is None:
#     print("请设置OPENAI_API_KEY")
# else:
#     print("已设置OPENAI_API_KEY" + openai_api_key)

host = "localhost"
port = 9200
auth = ("admin", "admin")  # For testing only. Don't store credentials in code.
# ca_certs_path = '/full/path/to/root-ca.pem' # Provide a CA bundle if you use intermediate CAs with your root CA.

model_name='shibing624/text2vec-base-chinese'
# model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()


# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


def gpt_answer(
    question, chaintype="stuff", industry="", top_k=4, model_name="gpt-3.5-turbo"
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
    index = pinecone.Index("ruledb")
    store = Pinecone(
        index, embeddings.embed_query, text_key="text", namespace=collection_name
    )

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
    llm = ChatOpenAI(model_name=model_name, max_tokens=512)
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
    result = chain({"query": question, "top_k_docs_for_context": top_k})

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
    index = pinecone.Index("ruledb")
    store = Pinecone(
        index, embeddings.embed_query, text_key="text", namespace=collection_name
    )
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
    docs = store.similarity_search(query=question, k=topk, filter=filter)
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
    if len(lst)==0:
        return None
    elif len(lst) == 1:
        return {"监管要求": lst[0]}
    else:
        return {"监管要求": {"$in": [item for item in lst]}}
