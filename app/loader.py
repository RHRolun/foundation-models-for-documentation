import os

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma

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

def main():
    index_documents()

if __name__ == "__main__":
    main()