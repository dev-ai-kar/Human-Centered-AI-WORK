import streamlit as st
import sys
from pathlib import Path

# ChromaDB requires a newer SQLite version on Streamlit Community Cloud.
import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import tiktoken
from openai import OpenAI
from pypdf import PdfReader


EMBEDDING_MODEL = "text-embedding-3-small"
PDF_FOLDER = Path("./Lab-04-Data")


def create_collection():
    """Create or retrieve the persistent ChromaDB collection for Lab 4."""
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
    return chroma_client.get_or_create_collection("Lab4Collection")


def add_to_collection(collection, text, file_name):
    """Embed PDF text with OpenAI and add it to the ChromaDB collection."""
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
    )


def extract_text_from_pdf(pdf_path):
    """Extract and combine text from every page in a PDF."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_pdfs_to_collection(folder_path, collection):
    """Extract, embed, and store every PDF in a folder."""
    folder = Path(folder_path)
    loaded_files = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        text = extract_text_from_pdf(pdf_path)
        if text:
            add_to_collection(collection, text, pdf_path.name)
            loaded_files.append(pdf_path.name)

    return loaded_files


def retrieve_course_context(client, question, collection, result_count=3):
    """Retrieve the most relevant course documents for a user question."""
    if collection.count() == 0:
        return "No course documents are available.", []

    response = client.embeddings.create(
        input=question,
        model=EMBEDDING_MODEL,
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(result_count, collection.count()),
    )

    documents = results.get("documents", [[]])[0]
    document_ids = results.get("ids", [[]])[0]
    context_parts = []
    sources = []

    for document_id, document in zip(document_ids, documents):
        sources.append(document_id)
        context_parts.append(
            f"Source: {document_id}\n{document[:4000]}"
        )

    if not context_parts:
        return "No relevant course information was found.", []

    return "\n\n---\n\n".join(context_parts), sources

try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "OpenAI API key not found in secrets. Please configure it in "
        "`.streamlit/secrets.toml`",
        icon="🗝️",
    )
    st.stop()

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

# Persistent ChromaDB collection stored in the project directory, not in session state.
collection = create_collection()

loaded_count = collection.count()
if loaded_count == 0:
    loaded_files = load_pdfs_to_collection(PDF_FOLDER, collection)
    loaded_count = len(loaded_files)

# Show title and description.
st.title("Lab 4: Chatbot using RAG")
st.write(
    "Ask questions and get answers from GPT. This chatbot uses a turn-based "
    "buffer with a maximum context token limit."
)

if loaded_count > 0:
    st.success(f"ChromaDB collection loaded successfully: {loaded_count} document(s) ready.")
else:
    st.warning("ChromaDB collection is empty. No documents were loaded yet.")

model_to_use = st.sidebar.selectbox(
    "Which model?", ("gpt-4o-mini", "gpt-4o"), index=0
)
max_tokens = st.sidebar.number_input(
    "Maximum context tokens",
    min_value=256,
    max_value=128000,
    value=4000,
    step=256,
    help=(
        "Sets the maximum number of conversation tokens sent to the model. "
        "A higher value keeps more recent history, while a lower value uses "
        "less context."
    ),
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "What can I help you with?"}
    ]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful course information assistant speaking to a "
        "10-year-old. Use the supplied course-document context to answer "
        "questions accurately. Clearly say when your answer uses the course "
        "information, for example, 'Based on the course information...' Do "
        "not invent course details. If the context does not answer the "
        "question, say that the course documents do not provide that "
        "information and then give a brief general answer if appropriate. "
        "Use simple words, short sentences, and friendly examples. "
        "First, answer the user's question clearly. After every answer, "
        "ask exactly: Do you want more info? If the user says yes, give "
        "useful additional information in simple language, then ask exactly "
        "again: Do you want more info? If the user says no, reply exactly: "
        "What can I help you with? Then wait for a new question."
    ),
}


def count_tokens(messages, model):
    """Count the tokens in a chat-completions message list."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    return sum(
        4 + len(encoding.encode(message["content"])) for message in messages
    ) + 2


def rag_conversation_buffer(messages, token_limit, model, context):
    """Add retrieved course context to the conversation sent to the LLM."""
    buffered_messages = [
        SYSTEM_MESSAGE,
        {
            "role": "system",
            "content": (
                "Relevant course-document context from the vector database "
                "follows. Use it as the primary source for course questions.\n\n"
                f"{context}"
            ),
        },
    ]
    history = messages[-4:]

    for message in history:
        candidate = buffered_messages + [message]
        if count_tokens(candidate, model) > token_limit:
            continue
        buffered_messages.append(message)

    return buffered_messages

if openai_api_key:
    client = st.session_state.openai_client

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context, sources = retrieve_course_context(
            client, prompt, collection
        )
        messages_for_request = rag_conversation_buffer(
            st.session_state.messages, max_tokens, model_to_use, context
        )
        request_tokens = count_tokens(messages_for_request, model_to_use)
        st.caption(f"Request context: {request_tokens:,} tokens")
        if sources:
            st.caption(f"RAG sources: {', '.join(sources)}")

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_request,
                stream=True,
            )
            response = st.write_stream(stream)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
