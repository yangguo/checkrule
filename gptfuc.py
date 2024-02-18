# from langchain.llms import OpenAI
import json
import os
import time
from operator import itemgetter

import jwt

# import chromadb
# import faiss
import pandas as pd

# import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import (
    CohereRerank,
    LLMChainExtractor,
    LLMChainFilter,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import StrOutputParser
from langchain_community.chat_models import (
    ChatBaichuan,
    ChatOllama,
    QianfanChatEndpoint,
)
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.vectorstores import (  # FAISS,; Chroma,; Milvus,; OpenSearchVectorSearch,; Pinecone,; Qdrant,
    Neo4jVector,
    SupabaseVectorStore,
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from supabase.client import Client, create_client

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
AZURE_DEPLOYMENT_NAME_GPT4_TURBO = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_TURBO")
AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ.get("AZURE_DEPLOYMENT_NAME_EMBEDDING")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

BAICHUAN_API_KEY = os.environ.get("BAICHUAN_API_KEY")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY")
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

# model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

# embeddings = HuggingFaceHubEmbeddings(
#     repo_id=model_name,
#     task="feature-extraction",
#     huggingfacehub_api_token=HF_API_TOKEN,
# )

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_BASE_URL,
    azure_deployment=AZURE_DEPLOYMENT_NAME_EMBEDDING,
    openai_api_version="2023-08-01-preview",
    openai_api_key=AZURE_API_KEY,
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
    "gpt-4-turbo": AZURE_DEPLOYMENT_NAME_GPT4_TURBO,
}


def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


# choose chatllm base on model name
def get_chatllm(model_name):
    if (
        model_name == "qwen-turbo"
        or model_name == "qwen-plus"
        or model_name == "qwen-max"
        or model_name == "chatglm-6b-v2"
        or model_name == "chatglm3-6b"
        or model_name == "baichuan2-13b-chat-v1"
    ):
        llm = ChatTongyi(
            # streaming=True,
            model_name=model_name,
        )
    elif (
        model_name == "ERNIE-Bot-4"
        or model_name == "ERNIE-Bot-turbo"
        or model_name == "ChatGLM2-6B-32K"
        or model_name == "Yi-34B-Chat"
        or model_name == "Mixtral-8x7B-Instruct"
    ):
        llm = QianfanChatEndpoint(
            model=model_name,
            # streaming=True,
        )
    elif model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(
            model=model_name, convert_system_message_to_human=True
        )
    elif model_name == "mistral" or model_name == "qwen:7b":
        llm = ChatOllama(
            model=model_name,
        )
    elif (
        model_name == "Baichuan2-Turbo"
        or model_name == "Baichuan2-Turbo-192k"
        or model_name == "Baichuan2-53B"
    ):
        llm = ChatOpenAI(
            model=model_name,
            api_key=BAICHUAN_API_KEY,
            base_url="https://api.baichuan-ai.com/v1",
        )
    elif (
        model_name == "moonshot-v1-8k"
        or model_name == "moonshot-v1-32k"
        or model_name == "moonshot-v1-128k"
    ):
        llm = ChatOpenAI(
            model=model_name,
            api_key=MOONSHOT_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )
    elif model_name == "glm-3-turbo" or model_name == "glm-4":
        llm = ChatOpenAI(
            model=model_name,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=generate_token(ZHIPUAI_API_KEY, 3600),
        )
    else:
        llm = get_azurellm(model_name)
    return llm


# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_BASE_URL,
        openai_api_version="2023-12-01-preview",
        azure_deployment=deployment_name,
        openai_api_key=AZURE_API_KEY,
        temperature=0.0,
        # streaming=True,
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
    # SupabaseVectorStore.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     client=supabase,
    #     table_name=collection_name,
    #     query_name="match_" + collection_name,
    # )

    # use neo4j
    Neo4jVector.from_texts(
        docs,
        embeddings,
        metadatas=metadata,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=collection_name,
        keyword_index_name=collection_name + "keyword",
        search_type="hybrid",
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
    industry="",
    top_k=4,
    model_name="gpt-35-turbo",
    items=[],
    retriever_type="",
    fusion_type=False,
    rag_type="basic",
    # memory=StreamlitChatMessageHistory(),
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

    #     store = Neo4jVector.from_existing_index(
    #     embedding=embeddings,
    #     url=NEO4J_URI,
    #     username=NEO4J_USERNAME,
    #     password=NEO4J_PASSWORD,
    #     index_name=collection_name,
    #     keyword_index_name=collection_name+"keyword",
    #     search_type="hybrid",
    # )

    # filter_value = {"监管要求": "信息技术管理办法"}
    llm = get_chatllm(model_name)
    filter = convert_list_to_dict(items)

    base_retriever = store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k, "filter": filter}
    )
    if retriever_type == "similarity":
        retriever = base_retriever
    elif retriever_type == "mmr":
        retriever = store.as_retriever(
            search_type="mmr"  # , search_kwargs={"k": top_k, "filter": filter}
        )
    elif retriever_type == "multiquery":
        retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    elif retriever_type == "rerank":
        compressor = CohereRerank(model="rerank-multilingual-v2.0")
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    # compressor = LLMChainExtractor.from_llm(llm)

    # _filter = LLMChainFilter.from_llm(llm)

    # metadata_field_info = [
    #     AttributeInfo(
    #         name="结构",
    #         description="the chapter in which the regulatory provisions are located",
    #         type="string",
    #     ),
    #     AttributeInfo(
    #         name="监管要求",
    #         description="the name of the regulatory policy",
    #         type="string",
    #     ),
    # ]
    # document_content_description = "Contents of regulatory provisions"

    # selfquery_retriever = SelfQueryRetriever.from_llm(
    #     llm,
    #     store,
    #     document_content_description,
    #     metadata_field_info,
    #     verbose=True,
    # )

    if fusion_type:
        # initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever])
    else:
        ensemble_retriever = retriever

    prompt = hub.pull("vyang/gpt_answer")

    output_parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # basic rag chain
    rag_chain = (
        # {"context":itemgetter("question")| ensemble_retriever | format_docs,"question": itemgetter("question")}
        {
            # Retrieve context using the normal question
            "context": RunnableLambda(lambda x: x["question"])
            | ensemble_retriever
            | format_docs,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | output_parser
    )

    # rag chain with source
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | output_parser
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": ensemble_retriever, "question": RunnablePassthrough()}
    ) | {
        # "contents": lambda input: [doc.page_content for doc in input["documents"]],
        "documents": lambda input: [doc for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    # result = rag_chain_with_source.invoke(question)
    # answer = result["answer"]
    # # sourcedf=None
    # source = result["documents"]
    # sourcedf = docs_to_df(source)
    # return answer, sourcedf

    # Stepback===================================================
    stepbackprompt = hub.pull("vyang/rag-stepback")

    response_prompt = hub.pull("langchain-ai/stepback-answer")

    generate_queries_step_back = stepbackprompt | llm | output_parser

    stepback_chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"])
            | ensemble_retriever
            | format_docs,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back
            | ensemble_retriever
            | format_docs,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | output_parser
    )

    # hyde chain==============================================
    # HyDE document genration
    template = """Please write regulational requirements to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = prompt_hyde | llm | output_parser

    retrieval_chain = generate_docs_for_retrieval | ensemble_retriever | format_docs

    hyde_rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | prompt
        | llm
        | output_parser
    )

    # Decomposition=======================================================
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # Chain generate_queries_decomposition
    generate_queries_decomposition = (
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # RAG prompt
    prompt_rag = hub.pull("rlm/rag-prompt")

    # chain for decomposition
    answer_chain = (
        {
            "context": itemgetter("question") | ensemble_retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt_rag
        | llm
        | StrOutputParser()
    )

    def retrieve_and_rag(sub_questions):
        """RAG on each sub-question"""

        # Use our decomposition /
        # sub_questions = generate_queries_decomposition.invoke({"question":question})

        # Initialize a list to hold RAG chain results
        rag_results = []

        for sub_question in sub_questions:

            # Retrieve documents for each sub-question
            # retrieved_docs = retriever.get_relevant_documents(sub_question)

            # Use retrieved documents and sub-question in RAG chain
            # answer = (
            #     {"context": itemgetter("question") | ensemble_retriever | format_docs, "question": itemgetter("question")

            #     }|

            #     prompt_rag | llm | StrOutputParser()).invoke({#"context": retrieved_docs,
            #                                                         "question": sub_question})
            answer = answer_chain.invoke({"question": sub_question})

            rag_results.append(answer)

        return rag_results

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    # answers, questions = retrieve_and_rag(question)

    def format_qa_pairs(input):
        """Format Q and A pairs"""

        formatted_string = ""
        for i, (question, answer) in enumerate(
            zip(input["questions"], input["answers"]), start=1
        ):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    # context = format_qa_pairs({"questions": questions, "answers": answers})

    # print(context)
    # Prompt
    decom_template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    decom_prompt = ChatPromptTemplate.from_template(decom_template)

    decom_rag_chain = (
        {
            "context": {
                "questions": {"question": itemgetter("question")}
                | generate_queries_decomposition,
                "answers": {"question": itemgetter("question")}
                | generate_queries_decomposition
                | RunnableLambda(retrieve_and_rag),
            }
            | RunnableLambda(format_qa_pairs),
            "question": itemgetter("question"),
        }
        | decom_prompt
        | llm
        | StrOutputParser()
    )

    # choose chain based on rag type
    if rag_type == "basic":
        final_chain = rag_chain
    elif rag_type == "stepback":
        final_chain = stepback_chain
    elif rag_type == "hyde":
        final_chain = hyde_rag_chain
    elif rag_type == "decomposition":
        final_chain = decom_rag_chain

    print(final_chain.get_graph().print_ascii())
    print(final_chain.get_prompts())

    result = final_chain.stream({"question": question})

    print(result)
    return result, []


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

    # docs = store.similarity_search(query=question, k=topk, filter=filter)
    retriever = store.as_retriever(
        search_type="similarity", search_kwargs={"k": topk, "filter": filter}
    )
    docs = retriever.get_relevant_documents(question)
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
