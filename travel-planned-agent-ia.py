import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import bs4

# Camada: Request Client

llm = ChatOpenAI(model="gpt-3.5-turbo")

# query="""
#   Vou viajar para Londres em agosto de 2024. 
#   Quero que faça um roteiro de viagem para mim com os eventos que irão ocorrer na data da viagem.
#   E com os preços de passagem de São Paulo para Londres.
# """

# Camada: Agent LangChain

def researchAgent(query, llm):
  
  tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
  prompt = hub.pull("hwchase17/react")
  agent = create_react_agent(llm, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt) # em desenvolvimento usava o verbose=True
  webContext =  agent_executor.invoke({"input": query})
  return webContext['output']

# print(researchAgent(query, llm))

# Camada: RAG

def loadData():
  loader = WebBaseLoader(
  web_paths= ("https://www.dicasdeviagem.com/inglaterra/",),
  bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark"))),)
  docs = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
  retriever = vectorstore.as_retriever()
  return retriever

def getRelevantDocs(query):
  retriever = loadData()
  relevant_documents = retriever.invoke(query)
  # print(relevant_documents)
  return relevant_documents

# Camada: Supervisor LangChain

def supervisorAgent(query, llm, webContext, relevant_documents):
  prompt_template = """
    Você é um gerente de aum agencia de viagens. 
    Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
    Utilize o contexto de eventos e preços de passagens, o input do usuario e tambem os documentos relevantes
    Contexto: {webContext}
    Documento relevante: {relevant_documents}
    Usurario: {query}
    Assistente:
  """

  prompt = PromptTemplate(input_variables= ['webContext', 'relevant_documents', 'query'], template = prompt_template)

  sequence = RunnableSequence(prompt | llm)

  response = sequence.invoke({"webContext": webContext, "relevant_documents": relevant_documents, "query": query})

  return response

def getResponse(query, llm):
  webContext = researchAgent(query, llm)
  relevant_documents = getRelevantDocs(query)
  response = supervisorAgent(query, llm, webContext, relevant_documents)
  return response

# Camada: Lambda Function

def handler_lambda(event, context):
  query = event.get("question")
  response = getResponse(query, llm).content
  return {"body": response, "status": 200}

# print(getResponse(query, llm).content)

# print(tools[0].name, tools[0].description)

# agent = initialize_agent(
#  tools,
#  llm,
#  agent= 'zero-shot-react-description',
#  verbose = True
#)