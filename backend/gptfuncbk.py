import os

import pandas as pd

# import pinecone
from dotenv import load_dotenv
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (
    HuggingFaceHubEmbeddings,
)

# from langchain.indexes import VectorstoreIndexCreator
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import (
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

# embeddings = EmbaasEmbeddings(
#     model="paraphrase-multilingual-mpnet-base-v2",
#     instruction="",
# )

# openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
# openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
# openai.api_base = "https://az.139105.xyz/v1"


# llm = ChatOpenAI(model_name="gpt-3.5-turbo",
#                  openai_api_base="https://op.139105.xyz/v1",
#                  openai_api_key=api_key)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_base="https://az.139105.xyz/v1",
    openai_api_key=AZURE_API_KEY,
)

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
    question,
    chaintype="stuff",
    industry="",
    top_k=4,
    model_name="gpt-3.5-turbo",
    items=[],
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

    filter = convert_list_to_dict(items)

    # chain = VectorDBQA.from_chain_type(
    retriever = store.as_retriever(search_kwargs={"k": top_k, "filter": filter})

    chain = RetrievalQA.from_chain_type(
        llm,
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

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings,
    )

    filter = convert_list_to_dict(items)
    # convert dict to json
    # filter_json = json.dumps(filter)
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
    elif industry_name == "投行":
        return "invbank"
    elif industry_name == "反洗钱":
        return "aml"
    elif industry_name == "医药":
        return "pharma"
    else:
        return "other"


def convert_list_to_dict(lst):
    if len(lst) == 1:
        return {"监管要求": lst[0]}
    else:
        return {}


def get_audit_steps(text, model_name="gpt-3.5-turbo"):
    response_schemas = [
        ResponseSchema(
            name="审计步骤", description="针对监管要求，需要执行的多项具体审计工作步骤"
        ),
        ResponseSchema(
            name="访谈问题",
            description="针对监管要求，需要向被审计方提出的多项访谈问题",
        ),
        ResponseSchema(
            name="资料清单", description="针对监管要求，需要被审计方准备的多项审计资料"
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    template = """
    你是一位具有10年资深经验的内部审计师，你的任务是根据监管要求生成审计工作计划。

    我需要你根据以下监管要求分解成审计目标，并针对这个审计目标编写详细的审计工作计划，并提供相关内容。内容包括：审计工作步骤、访谈问题、资料清单。

    所有的审计步骤、访谈问题和资料清单应当在一个完整的回复中给出。

    {format_instructions}

    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = """    
    监管要求的内容如下:
    {text}
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate(
        messages=[system_message_prompt, human_message_prompt],
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(text=text)

    # print(response)
    json_response = output_parser.parse(response)

    return json_response
