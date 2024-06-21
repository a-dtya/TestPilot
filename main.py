from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

from prompts import context
from code_reader import code_reader

from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import code_parser_template

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
#to check the code_reader and agent functionality
"""while(prompt:=input("Enter prompt (press q to quit)"))!="q":
    result=agent.query(prompt) #read the contents of test.py and explain what it does 
    print(result)"""


class CodeOutput(BaseModel): #creating a pydantic object
    code:str
    description:str
    filename:str

parser=PydanticOutputParser(CodeOutput)
json_prompt_str=parser.format(code_parser_template)#adds the pydantic object to end of prompt, which means required o/p format
json_prompt_tmplate=PromptTemplate(json_prompt_str)# thsis would be the {response} in code_parser_template
output_pipeline=QueryPipeline(chain=[json_prompt_tmplate,llm])

while(prompt:=input("Enter prompt (press q to quit)"))!="q":
    retries=0
    while retries<50:
        try:

            result=agent.query(prompt)
            next_result=output_pipeline.run(response=result)
            print(result)
            print(next_result)
            cleaned_json=ast.literal_eval(str(next_result).replace("assistant:",""))# to remove the assistant term in the dictionary o/p
        #cleaned_json gives a dicitonary
            break
        except Exception as e:
            retries+=1
            print(f"Error occured: retry#{retries}:",e) 

    if retries>=50:
        print("Try Again with a better prompt")
        continue

    print("Code Generated")
    print(cleaned_json["code"])
    print("\n\nDescription: ",cleaned_json["description"])
    filename=cleaned_json["filename"]

    #saving the generated code to a file
    try:
        with open(os.path.join("output",filename),"w") as f:
            f.write(cleaned_json["code"])
        print("File saved", filename)
    except:
        print("Error in saving the file")
