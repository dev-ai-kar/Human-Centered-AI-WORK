import json
import sys
from pathlib import Path

import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from bs4 import BeautifulSoup
import chromadb
import tiktoken
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
HTML_FOLDER = Path("./su_orgs_hw4_data")
VECTOR_DB_PATH = Path("./ChromaDB_for_HW")
MODEL_NAME = "gpt-5-nano"
COLLECTION_NAME = "Lab4Collection"
MAX_INTERACTIONS = 5


def create_collection():
    """Create or retrieve the persistent ChromaDB collection for the club database."""
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    return chroma_client.get_or_create_collection(COLLECTION_NAME)


def add_to_collection(collection, text, file_name):
    """Embed a chunk of HTML text and add it to the ChromaDB collection."""
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    embedding = response.data[0].embedding
    collection.add(documents=[text], ids=[file_name], embeddings=[embedding])


def extract_text_from_html(html_path):
    """Remove boilerplate from an HTML org page and return plain text."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def chunk_html_text(text):
    """Split a longer HTML page into a small number of meaningful chunks."""
    cleaned_text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not cleaned_text:
        return []

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


def relevant_club_info(query: str, collection=None, client=None, result_count=3) -> str:
    """Use ChromaDB to retrieve the most relevant organization information for a query."""
    if collection is None:
        collection = st.session_state.get("club_collection")
    if client is None:
        client = st.session_state.openai_client

    if collection is None or collection.count() == 0:
        return json.dumps({"context": "No student organization information is available in the local database.", "sources": []})

    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(result_count, collection.count()),
    )

    documents = results.get("documents", [[]])[0]
    document_ids = results.get("ids", [[]])[0]
    if not documents:
        return json.dumps({"context": "No relevant student organization information was found.", "sources": []})

    context_parts = []
    for document_id, document in zip(document_ids, documents):
        context_parts.append(f"Source: {document_id}\n{document[:4000]}")

    return json.dumps({
        "context": "\n\n---\n\n".join(context_parts),
        "sources": list(document_ids),
    })


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

club_collection = create_collection()
st.session_state.club_collection = club_collection

loaded_count = club_collection.count()
if not VECTOR_DB_PATH.exists() or loaded_count == 0:
    loaded_files = load_html_files_to_collection(HTML_FOLDER, club_collection)
    loaded_count = len(loaded_files)

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful student organization assistant. Use the relevant_club_info tool to find facts "
        "about clubs, organizations, events, and activities from the local campus database. Only answer with "
        "details supported by the retrieved documents. If the available information does not answer the question, "
        "say so clearly and do not invent facts. Remember the recent chat history but keep your answers concise and grounded."
    ),
}


def count_tokens(messages, model):
    """Count the approximate token usage for a message list."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    return sum(4 + len(encoding.encode(str(message.get("content", "")))) for message in messages) + 2


def trim_conversation_history(messages, max_interactions=MAX_INTERACTIONS):
    """Keep only the most recent user/assistant interactions in memory."""
    return messages[-(max_interactions * 2):]


CLUB_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": "Search the ChromaDB database for information related to a student organization question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A question about a club, organization, event, or campus activity.",
                }
            },
            "required": ["query"],
        },
    },
}

st.title("Homework 5: Short-Term Memory Club Assistant")
st.write(
    "Ask about student organizations using a tool-based retrieval flow. "
    "The model decides when to call the organization search and then answers "
    "using the retrieved evidence."
)

if loaded_count > 0:
    st.success(f"ChromaDB collection loaded successfully: {loaded_count} document(s).")
else:
    st.warning("ChromaDB collection is empty. No organization documents were loaded yet.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about a club, event, or student organization."}]

for message in st.session_state.messages:
    if message["role"] == "tool":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a student organization..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages = trim_conversation_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_request = [SYSTEM_MESSAGE] + trim_conversation_history(st.session_state.messages)
    completion = st.session_state.openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages_for_request,
        tools=[CLUB_INFO_TOOL],
        tool_choice="auto",
    )
    assistant_message = completion.choices[0].message

    conversation_turn = [SYSTEM_MESSAGE] + trim_conversation_history(st.session_state.messages)
    final_answer = "I couldn't find enough relevant information to answer that confidently."
    source_ids = []

    if assistant_message.tool_calls:
        conversation_turn.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in assistant_message.tool_calls
            ],
        })

        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "relevant_club_info":
                arguments = json.loads(tool_call.function.arguments)
                tool_result = relevant_club_info(arguments["query"], club_collection, st.session_state.openai_client)
                parsed_result = json.loads(tool_result)
                source_ids = parsed_result.get("sources", [])
                conversation_turn.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "relevant_club_info",
                    "content": tool_result,
                })

        if source_ids:
            st.caption(f"Retrieved from: {', '.join(source_ids)}")

        follow_up = st.session_state.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_turn,
            tools=[CLUB_INFO_TOOL],
            tool_choice="none",
        )
        final_answer = follow_up.choices[0].message.content or final_answer
    else:
        final_answer = assistant_message.content or final_answer

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.session_state.messages = trim_conversation_history(st.session_state.messages)
