# Ingest script to load documents, chunk them, 
# and create a vector store using Chroma and Google Generative AI Embeddings.

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_files(file_path = "docs"):
    """Load the files from the specified directory and return a list of documents."""
    print(f"Loading files from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    
    loader = DirectoryLoader(path=file_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def chunk_files(documents, chunk_size=1000, chunk_overlap=200):
    """Chunking the documents into smaller pieces."""
    print(f"Chunking the documents into chunks of size {chunk_size} with an overlap of {chunk_overlap}...")
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, collection_name="my_collection"):
    """Create a Chroma vector store from the chunks."""
    print(f"Creating a vector store with collection name '{collection_name}'...")
    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"Valid chunks: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-2")

    vector_store = Chroma.from_documents(
        documents = chunks, 
        embedding = embeddings, 
        collection_name = collection_name
 )
    
    print(f"Vector store created with {vector_store._collection.count()} documents.")
    return vector_store


if __name__ == "__main__":
    # Loading the files
    documents = load_files()
    # Chunking the files
    chunks = chunk_files(documents)
    # Creating the vector store
    vector_store = create_vector_store(chunks)
    print("Ingestion complete.")