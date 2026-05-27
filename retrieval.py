from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

persistent_directory = "chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="my_collection", 
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

query = "What research areas does OpenAI work in?"

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("Relevant documents retrieved:")
print(relevant_docs)

# passing the retrieved documents to a language model for answer generation would go here
model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)
response = model.invoke(
    f"Based on the following documents, answer the question: {query}\n\nDocuments:\n{relevant_docs}"
)
print(f"Generated response: {response.content}")
