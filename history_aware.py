# history_aware_retrieval.py
# A history-aware conversational RAG pipeline built on top of
# existing ingestion (Chroma + HuggingFace) and retrieval (Gemini) setup.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR      = "chroma_db"
COLLECTION_NAME  = "my_collection"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory = PERSIST_DIR,
    embedding_function = embedding_model,
)

base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might "
    "reference context in the chat history, formulate a standalone "
    "question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed; "
    "otherwise return it as is."
)

contextualizer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)

QA_SYSTEM_PROMPT = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, say so clearly. "
    "Keep the answer concise (3-5 sentences max).\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])

def chat(user_input : str, chat_history : list) -> tuple[str, list]:
    """
    Send one user message through the history-aware RAG chain
    Args:
        user_input: The latest user message
        chat_history: List of previous messages in the conversation (alternating HumanMessage and AIMessage)
    Returns:
        answer: The generated response from the model
        chat_history: The updated chat history including the new user input and model response
    """
    contextualized_chain = contextualizer_prompt | llm
    standalone_question = contextualized_chain.invoke({
        "user_input": user_input,
        "chat_history": chat_history
    })
    standalone_question = standalone_question.content
    print(f"\n[Standalone Question]: {standalone_question}")

    retrieved_docs = base_retriever.invoke(standalone_question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    qa_chain = qa_prompt | llm
    answer = qa_chain.invoke({
        "user_input": user_input,
        "chat_history": chat_history,
        "context": context
    }).content

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))

    return answer, chat_history

if __name__ == "__main__":
    print("History-Aware RAG Chatbot")
    print("Type 'history' to view the conversation, 'quit' to exit.\n")
 
    chat_history : list = []
    while True:
        user_input = input("You : ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break
        answer, chat_history = chat(user_input, chat_history)
        print(f"\nBot : {answer}")