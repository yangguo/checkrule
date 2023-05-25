import json
import os

import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceHubEmbeddings,
    OpenAIEmbeddings,
)

# from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import Cohere, OpenAIChat
from langchain.prompts import PromptTemplate
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
    SupabaseVectorStore,
)
from supabase import Client, create_client

load_dotenv()

huggingfacehub_api_token = os.environ.get("HF_API_TOKEN")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "ruleidx"
backendurl = "http://localhost:8000"


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# model_name='shibing624/text2vec-base-chinese'
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=huggingfacehub_api_token,
)


# openai_api_key = os.environ.get("OPENAI_API_KEY")
# if openai_api_key is None:
#     print("请设置OPENAI_API_KEY")
# else:
#     print("已设置OPENAI_API_KEY" + openai_api_key)

# host = "localhost"
# port = 9200
# auth = ("admin", "admin")  # For testing only. Don't store credentials in code.
# ca_certs_path = '/full/path/to/root-ca.pem' # Provide a CA bundle if you use intermediate CAs with your root CA.

# model_name='shibing624/text2vec-base-chinese'
# model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()


# initialize pinecone
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


def gpt_answer(
    question, chaintype="stuff", industry="", top_k=4, model_name="gpt-3.5-turbo"
):
    collection_name = industry_name_to_code(industry)

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
    # system_template = """根据提供的背景信息，请准确和全面地回答用户的问题。
    # 如果您不确定或不知道答案，请直接说明您不知道，避免编造任何信息。
    # ----------------
    # {context}"""
    # messages = [
    #     SystemMessagePromptTemplate.from_template(system_template),
    #     HumanMessagePromptTemplate.from_template("{question}"),
    # ]
    # prompt = ChatPromptTemplate.from_messages(messages)

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Chinese:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": prompt}

    import openai

    # openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
    # openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
    openai.api_base = "https://az.139105.xyz/v1"

    # llm = ChatOpenAI(model_name=model_name )
    llm = ChatOpenAI(model_name=model_name, openai_api_key=AZURE_API_KEY)

    # use azure model
    #     llm = AzureChatOpenAI(
    #     openai_api_base=AZURE_BASE_URL,
    #     openai_api_version="2023-03-15-preview",
    #     deployment_name=AZURE_DEPLOYMENT_NAME,
    #     openai_api_key=AZURE_API_KEY,
    #     openai_api_type = "azure",
    # )
    # use cohere model
    # llm = Cohere(model="command-xlarge-nightly",cohere_api_key=COHERE_API_KEY)

    # chain = VectorDBQA.from_chain_type(
    receiver = store.as_retriever()
    receiver.search_kwargs["k"] = top_k

    chain = RetrievalQA.from_chain_type(
        llm,
        chain_type=chaintype,
        # vectorstore=store,
        retriever=receiver,
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

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings,
    )
    # List all collections
    # collections = store.list_collections()

    # print(collections)

    # filter = {"监管要求": "商业银行信息科技风险管理指引"}
    #     filter ={
    #   "key": "监管要求",
    #   "type": "string",
    #   "match": {
    #     "value": "商业银行信息科技风险管理指引"
    #   }
    # }

    # filter = convert_list_to_dict(items)

    # substore=collection.query(["query text"], {"where": flter})
    # print(filter)
    # filter=None
    docs = store.similarity_search(query=question, k=topk)
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
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return {"监管要求": lst[0]}
    else:
        return {"监管要求": {"$in": [item for item in lst]}}
