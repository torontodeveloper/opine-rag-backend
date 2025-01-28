from typing import Union
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
import os
from langchain.chains.qa_generation.prompt import templ
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
import getpass
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import  SequentialChain
from fastapi import File,UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


app = FastAPI()

os.environ['OPENAI_API_KEY']= ''

@app.get('/')
def get_home():
    model = ChatOpenAI(model='gpt-4o-mini')
    llm = OpenAI()
    code_prompt = PromptTemplate(template='write a very {language} function that will complete {task}',input_variables=["language","task"])
    test_prompt = PromptTemplate(template='write a test using Pytest {language} for {code}',input_variables=["language","code"])

    code_chain = LLMChain(llm=llm,prompt = code_prompt,output_key="code")
    test_chain = LLMChain(llm=llm,prompt=test_prompt,output_key = "test")
    lang = input('what\'s ur language')
    task = input('whats ur task')

    chain = SequentialChain(
        chains=[code_chain,test_chain],
        input_variables = ["language","task"],
        output_variables = ["test","code"]
    )
    result = chain({
        "language":lang,
        "task":task
    })
    print(result['test'])
    return result['test']

@app.post('/document')
async def post_document(doc:UploadFile = File(...)):
    print('This REST API POST**********',doc)
    
    file_path = f'temp_{doc.filename}'
    with open(file_path,'wb') as file:
        file.write(await doc.read())

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    pages[0]
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    docs = faiss_index.similarity_search("summarize the document?", k=2)
    for doc in docs:
        print('doc**********')
        print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
    return {
        "file":f"{docs}"
    }