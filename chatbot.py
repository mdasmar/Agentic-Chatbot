from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,SystemMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import sqlite3
import requests
import math
import os

load_dotenv()

WEATHERSTACK_API_KEY=os.getenv("WEATHERSTACK_API_KEY")

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]


embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def rag_docs(path):
    loader=PyPDFLoader(path)
    docs=loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(docs)
    vecstore=FAISS.from_documents(chunks,embedding)

    vecstore.save_local("faissDB")

def get_reteiver():
    vecstore=FAISS.load_local(folder_path='faissDB',embeddings=embedding,allow_dangerous_deserialization=True)
    retreiver=vecstore.as_retriever(search_type='similarity',search_kwargs={'k':4})
    return retreiver

model = ChatOpenAI(
    model=deployment,
    base_url=f"{endpoint}/openai/v1/",
    api_key=api_key
)

search_tool=TavilySearch(max_results=5,topic='general',search_depth='advanced')

@tool
def rag_tool(query:str)->str:
    """
    Retrieve relevant information from the PDF Document.
    use this tool when user asks factual questions which can be answered only using sotred pdf documents
    Args:
        query:the question used to retrive the pdf content.
    """
    retreiver=get_reteiver()
    docs=retreiver.invoke(query)
    if not docs:
        return "No relevant information found in PDF"
    frmt_docs=[]
    for index,doc in enumerate(docs,start=1):
        source=doc.metadata.get('source','unknown')
        page=doc.metadata.get('page','unkonwn')

        frmt_docs.append(
            f"Document {index}\n"
            f"Source:{source}\n"
            f"Page:{page}\n"
            f"Content:{doc.page_content}"
        )

    return "\n\n".join(frmt_docs)
    

@tool
def calculator(expression:str)->str:
    """
    Useful for simple math calculation.
    Input should be a valid math expression.
    Example: 2+2, math.sqrt(16), 10*5
    """
    try:
        allowed={'math':math,'abs':abs,'round':round,
                 'min':min,'max':max,'sum':sum}
        result=eval(expression,{'__builtin__':{}},allowed)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}" 

@tool
def get_stock_price(symbol:str)->dict:
    """
    Fetch the latest stock price for given symbol
    Using Alpha Vintage with API key in URL.(eg 'AAPL','TSLA') 
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=18FFWK2X85IE6A4C"
    req=requests.get(url)
    return req.json()


@tool
def fetch_weather(city):
    """
    Fetch current weather information for a city.
    """
    url=(
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )
    response=requests.get(url)
    data=response.json()
    if "current" not in data:
        return f"could not fetch current weather of {city}"

    return (
        f"City:{city}\n"
        f"Temperature:{data['current']['temperature']}C\n"
        f"Weather:{data['current']['weather_descriptions'][0]}\n"
    )

@tool
def purchase_stock(symbol:str,quantity:int)->dict:
    """
    Simulate a given quantity of stock symbol.
    HUMAN-IN-THE-LOOP:
    before confirming the purchase, this tool will interupt and wait for human decision 
    ("yes"/anything else)
    """
    decision=interrupt("Approve bying {quantity} shares of {symbol}")

    if isinstance(decision,str) and decision.lower()=='yes':
        return {'status':'success','message':f'Purchase order placed for {quantity} shares of {symbol}',
                'symbol':symbol,'quantity':quantity}
    else:
        return {'status':'cancelled','message':f'Purchase order for {quantity} shares of {symbol} was declined',
                'symbol':symbol,'quantity':quantity}

tools=[search_tool,calculator,get_stock_price,fetch_weather,rag_tool,purchase_stock]
llm_with_tool=model.bind_tools(tools)


class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

conn=sqlite3.connect(database="Chat.db",check_same_thread=False)
checkpoint=SqliteSaver(conn)

graph=StateGraph(ChatState)

def chat(state:ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"
            "- Use `purchase_stock` when the user asks for purchasing the stock of any symbol with approval.\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    msg = [system_message,*state["messages"]]
    res=llm_with_tool.invoke(msg)
    return {'messages':res}

toolnode=ToolNode(tools)

graph.add_node('chatbot',chat)
graph.add_node('tools',toolnode)

graph.add_edge(START,'chatbot')
graph.add_conditional_edges('chatbot',tools_condition)
graph.add_edge('tools','chatbot')

workflow=graph.compile(checkpointer=checkpoint)

def get_threads():
    thrds=set()
    for ckpt in checkpoint.list(None):
        thrds.add(ckpt.config['configurable']['thread_id'])
    return list(thrds)




# config={'configurable':{'thread_id':"1"}}

# input={"messages":[HumanMessage(content="what is capital of india")]}

# res=workflow.invoke(input,config=config)

# print(res['messages'][-1].content)