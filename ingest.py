# Ingest script to load documents, chunk them,
# and create a vector store using Chroma and HuggingFace Embeddings.

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()


def load_files(file_path="docs"):
    """Load text files from directory."""

    print(f"Loading files from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")

    loader = DirectoryLoader(
        path=file_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    return documents


def chunk_files(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""

    print(
        f"Chunking documents into chunks of size "
        f"{chunk_size} with overlap {chunk_overlap}..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # Remove empty chunks
    chunks = [c for c in chunks if c.page_content.strip()]

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vector_store(chunks, collection_name="my_collection"):
    """Create Chroma vector store."""

    print(f"Creating vector store '{collection_name}'...")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # HuggingFace embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Test embeddings
    for i, text in enumerate(texts[:3]):
        try:
            emb = embedding_model.embed_query(text)

            print(
                f"Embedded chunk {i+1} "
                f"(vector size: {len(emb)})"
            )

        except Exception as e:
            print(f"Embedding failed for chunk {i}: {e}")

    # Create Chroma DB
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )

    # Add texts
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )

    print("Vector store created successfully.")

    return vector_store


if __name__ == "__main__":

    # Load documents
    documents = load_files()

    # Chunk documents
    chunks = chunk_files(documents)

    # Create vector DB
    vector_store = create_vector_store(chunks)

    print("Ingestion complete.")