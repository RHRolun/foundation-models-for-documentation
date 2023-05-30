import requests
import json
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field

from typing import Optional, List, Mapping, Any

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma

MODEL_URL = "https://text-generation-webui-llm.apps.et-gpu.zfq7.p1.openshiftapps.com/run/textgen"

PARAMS = {
    'max_new_tokens': 500,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.73,
    'typical_p': 1,
    'repetition_penalty': 1.0,
    'encoder_repetition_penalty': 1.0,
    'top_k': 0,
    'min_length': 10,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 0,
    'early_stopping': False,
    'seed': -1,
    'add_bos_token': True,
    'custom_stopping_strings': [],
    'truncation_length': 2048,
    'ban_eos_token': False,
    'skip_special_tokens': True,
}

app = FastAPI()

class Message(BaseModel):
    question: str

class ApiLLM(LLM):
    
    api_url: str

    api_params: dict

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM."""
        payload = json.dumps([prompt, self.api_params]) 
        print(prompt)
        response = requests.post(self.api_url, json={"data":[payload]}).json()
        return response["data"][0]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_url": self.api_url, "api_params": self.api_params}

def inference_finetuned_model(text: str):
    # payload = json.dumps([text, params]) 
    payload = {
        "inputs": [
            {
                "name": "dense_input", 
                "shape": [1, 7], 
                "datatype": "FP32",
                "data": [text]
            },
            ]
        }
    headers = {
        'content-type': 'application/json'
    }

    response = requests.post(finetuned_URL, json=payload, headers=headers)
    prediction = response.json()['outputs'][0]['data'][0]
    return prediction

def index_documents():
    loader = DirectoryLoader('../data/external', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding":"utf8"})
    documents = loader.load()  # FYI with the current dataset, documents[42] is the FAQ

    text_splitter = MarkdownTextSplitter(chunk_overlap=0, chunk_size=500)  # Consider setting chunk_size=1000
    texts = text_splitter.split_documents(documents)
    print(f"{len(documents)} documents were loaded in {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings()

    # https://langchain.readthedocs.io/en/latest/modules/indexes/vectorstore_examples/chroma.html#persist-the-database
    db_dir = "../data/interim"
    docsearch = None
    docsearch = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    return docsearch

def create_qa_chain(llm, docsearch: Chroma):
    template = """You are a talkative AI model who loves to explain how things work. You are smart and constantly learning.
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Detailed answer:"""
    qa_prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    chain_type_kwargs = {"prompt": qa_prompt}

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    return qa_chain

llm = ApiLLM(api_url=MODEL_URL, api_params=PARAMS)
docsearch = index_documents()
qa_chain = create_qa_chain(llm, docsearch)

def _ask_model(question: str):
    return qa_chain(question)

@app.post("/question")
def ask_model(message: Message):
    return _ask_model(message.question)

@app.post("/")
def root():
    return {"status":"working"}