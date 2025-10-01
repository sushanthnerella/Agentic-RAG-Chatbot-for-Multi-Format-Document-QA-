# In agentic_rag_backend/agents/ingestion_agent.py

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

# --- Configuration ---
# This directory will be created in your project folder to store the vector database.
CHROMA_PERSIST_DIRECTORY = "chroma_db"

# A dictionary mapping file extensions to their respective LangChain loader classes.
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader
}

def process_documents(file_paths: list[str], session_id: str):
    """
    Loads, splits, and embeds documents, then stores them in a ChromaDB collection
    specific to the session ID.

    Args:
        file_paths: A list of paths to the documents to be processed.
        session_id: A unique identifier for the user's session.
    """
    all_chunks = []
    for file_path in file_paths:
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        if ext in LOADER_MAPPING:
            try:
                print(f"Loading document: {file_path}")
                loader = LOADER_MAPPING[ext](file_path)
                documents = loader.load()

                print("Splitting documents into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                print(f"Created {len(chunks)} chunks from {file_path}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"Unsupported file type: {ext}")

    if not all_chunks:
        print("No chunks were created from the documents.")
        return

    # Initialize embeddings model
    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize ChromaDB client and create/get a collection for the session
    print(f"Storing chunks in ChromaDB for session: {session_id}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    
    # Using the session_id as the collection name ensures data isolation per session.
    collection = client.get_or_create_collection(name=session_id)

    # Embed and store the chunks
    # ChromaDB's add method handles embedding and storage automatically.
    # We create unique IDs for each chunk to avoid duplicates.
    for i, chunk in enumerate(all_chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata]
        )

    print(f"Successfully processed and stored {len(all_chunks)} chunks for session {session_id}.")