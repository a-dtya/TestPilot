from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

load_dotenv()

llm = Ollama(model="mistral",request_timeout=3600.0) #potato pcs require more time to churn the data chunks

parser=LlamaParse(result_type="markdown")

file_extractor= {".pdf":parser}

documents = SimpleDirectoryReader("./data",file_extractor=file_extractor).load_data()
embed_model=resolve_embed_model("local:BAAI/bge-m3")
vector_index=VectorStoreIndex.from_documents(documents,embed_model=embed_model) #to create vector embeddings and 
#store in a vector database
#vector store => vector database
query_engine=vector_index.as_query_engine(llm=llm)

#res=query_engine.query("What are the routes in the api?")
#print(res)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives us documentation about the code for an API. Use this for reading docs for the API."
        )
    )
]

code_llm=Ollama(model="codellama") #the model used by agent to generate code

agent=ReActAgent.from_tools(tools,llm=code_llm,verbose=True, context="") #verbose gives the thought process of the agent