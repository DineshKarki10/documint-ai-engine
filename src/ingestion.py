import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def process_pdf(file_path: str, db: Chroma) -> int:
    """
    Loads a PDF file, splits it into semantic chunks, and adds them to the Chroma vector store.
    Returns the total number of chunks added.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    # 1. Load PDF pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Split text into clean, overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(pages)

    # Add metadata for source tracking
    for doc in docs:
        doc.metadata["source"] = os.path.basename(file_path)

    # 3. Insert chunks into ChromaDB vector store
    db.add_documents(docs)
    db.persist()

    return len(docs)