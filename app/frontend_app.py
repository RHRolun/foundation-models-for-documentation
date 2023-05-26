import streamlit as st
import requests
import json
import os
from typing import Optional, List, Mapping, Any

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma


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
MODEL_URL = "https://text-generation-api-llm.apps.et-cluster.6mwp.p1.openshiftapps.com/api"

class ApiLLM(LLM):
    
    api_url: str

    api_params: dict

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM."""
        payload = json.dumps([prompt, self.api_params]) 
        import pdb; pdb.set_trace()
        response = requests.post(self.api_url, json={"data":[payload]}).json()
        return response["data"][0]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_url": self.api_url, "api_params": self.api_params}

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
    if os.path.isdir(os.path.join(db_dir, "index")):
        # Load the existing vector store
        docsearch = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        # Create a new vector store
        docsearch = Chroma.from_documents(texts[:1000], embeddings, persist_directory=db_dir)
        docsearch.persist()
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

def get_reply(search_text: str, qa_chain) -> str:
    return "Working"
    return qa_chain(search_text)

def main():
    llm = ApiLLM(api_url=MODEL_URL, api_params=PARAMS)
    docsearch = index_documents()
    qa_chain = create_qa_chain(llm, docsearch)
    

    st.set_page_config(
        page_title="RedHat Chat Bot",
    )

    st.title("RedHat Chat Bot")
    search_text = st.text_input("", value="What is RHODS?")
    ask_button = st.button("Ask")

    if ask_button:
        reply = get_reply(search_text, qa_chain)
        st.markdown(reply)

if __name__ == "__main__":
    main()

