import streamlit as st
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# ChromaDB requires a newer SQLite version on Streamlit Community Cloud.
import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
import tiktoken
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
HTML_FOLDER = Path("./su_orgs_hw4_data")
VECTOR_DB_PATH = Path("./ChromaDB_for_HW")
MODEL_NAME = "gpt-5-nano"
MAX_INTERACTIONS = 5


def create_collection():
    """Create or retrieve the persistent ChromaDB collection for Lab 4."""
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    return chroma_client.get_or_create_collection("Lab4Collection")


def add_to_collection(collection, text, file_name):
    """Embed a chunk of HTML text and add it to the ChromaDB collection."""
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


def extract_text_from_html(html_path):
    """Convert an HTML org page into clean, readable text."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def chunk_html_text(text):
    """Split each HTML page into two mini-documents.

    This uses a paragraph-aware, two-part split: we turn the page into a list of
    natural text blocks (paragraphs/sections), then divide those blocks into two
    halves. That keeps related ideas together, reduces the chance of splitting a
    sentence or concept in the middle, and gives each chunk enough context to be
    useful for retrieval.
    """
    cleaned_text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not cleaned_text:
        return []

    # Fallback for pages that are very short or have little structure.
    paragraphs = [part.strip() for part in cleaned_text.split("\n") if part.strip()]
    if len(paragraphs) < 2:
        midpoint = max(1, len(cleaned_text) // 2)
        return [cleaned_text[:midpoint].strip(), cleaned_text[midpoint:].strip()]

    midpoint = max(1, len(paragraphs) // 2)
    first_chunk = " ".join(paragraphs[:midpoint]).strip()
    second_chunk = " ".join(paragraphs[midpoint:]).strip()

    return [first_chunk, second_chunk]


def load_html_files_to_collection(folder_path, collection):
    """Extract, chunk, embed, and store every HTML org file in a folder."""
    folder = Path(folder_path)
    loaded_files = []

    for html_path in sorted(folder.glob("*.html")):
        text = extract_text_from_html(html_path)
        if not text:
            continue

        chunks = chunk_html_text(text)
        for chunk_index, chunk in enumerate(chunks, start=1):
            if not chunk:
                continue
            chunk_id = f"{html_path.name}__chunk_{chunk_index}"
            add_to_collection(collection, chunk, chunk_id)
            loaded_files.append(chunk_id)

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
# Only create and populate the vector DB when it doesn't already exist so the app can
# be rerun without re-embedding all organization pages every time.
collection = create_collection()

loaded_count = collection.count()
if not VECTOR_DB_PATH.exists() or loaded_count == 0:
    loaded_files = load_html_files_to_collection(HTML_FOLDER, collection)
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

model_to_use = MODEL_NAME
st.sidebar.caption(f"Using model: {model_to_use}")
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
        "You are a helpful campus organizations assistant. Use the retrieved "
        "organization information from the vector database as your main source of "
        "truth. If the available context does not answer the question, say so "
        "clearly and provide only a brief general answer. Do not invent details "
        "about organizations, events, leadership, requirements, or contact info. "
        "Answer in simple, friendly language that is easy for a college student to "
        "understand. Keep answers concise and grounded in the retrieved context. "
        "Remember the user’s recent conversation history, but do not rely on memory "
        "for facts that are not in the current context. If the question asks for "
        "a follow-up, connect it to the same organization or topic."
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


def trim_conversation_history(messages, max_interactions=MAX_INTERACTIONS):
    """Keep only the last five user/assistant exchanges in memory."""
    return messages[-(max_interactions * 2):]


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
    history = trim_conversation_history(messages)

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
        st.session_state.messages = trim_conversation_history(st.session_state.messages)
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
        st.session_state.messages = trim_conversation_history(st.session_state.messages)
