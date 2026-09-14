import streamlit as st
import tiktoken
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from google import genai


def read_url_content(url):
    """Reuse the same URL extraction logic from HW2."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException as error:
        print(f"Error reading {url}: {error}")
        return None


# Show title and description.
st.title("Conversational AI")
st.write(
    "Ask questions and get answers from GPT. This chatbot uses a turn-based "
    "buffer with a maximum context token limit, and it can use up to two web "
    "sources as extra context."
)

st.sidebar.header("⚙️ Options")

st.sidebar.subheader("📄 Web context")
url_1 = st.sidebar.text_input(
    "URL 1",
    placeholder="https://example.com/article-1",
)
url_2 = st.sidebar.text_input(
    "URL 2",
    placeholder="https://example.com/article-2",
)

st.sidebar.subheader("🤖 Model Selection")
llm_provider = st.sidebar.selectbox(
    "Choose an LLM:",
    options=["OpenAI", "Google Gemini"],
    index=0,
)

use_advanced_model = st.sidebar.checkbox(
    "Use advanced model",
    value=False,
    help="Use the more capable model for the selected LLM.",
)

if llm_provider == "OpenAI":
    model_to_use = "gpt-5-mini" if use_advanced_model else "gpt-5-nano"
else:
    model_to_use = "gemini-3.8-flash" if use_advanced_model else "gemini-3.5-flash-lite"

st.sidebar.markdown(f"**Selected Model:** `{model_to_use}`")

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
        "You are a helpful assistant speaking to a 10-year-old. "
        "Use simple words, short sentences, and friendly examples. "
        "Answer clearly and use the provided web context when it is available. "
        "After every answer, ask exactly: Do you want more info? If the user says yes, give "
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


def conversation_buffer(messages, token_limit, model):
    """Return the system prompt and newest complete two-turn history."""
    buffered_messages = [SYSTEM_MESSAGE]
    history = messages[-4:]

    for message in history:
        candidate = buffered_messages + [message]
        if count_tokens(candidate, model) > token_limit:
            continue
        buffered_messages.append(message)

    return buffered_messages


def build_url_context(urls):
    """Collect up to two web sources into one readable context block."""
    contexts = []
    for index, current_url in enumerate(urls, start=1):
        if not current_url or not current_url.strip():
            continue
        content = read_url_content(current_url)
        if content and content.strip():
            contexts.append(f"Source {index} ({current_url}):\n{content[:4000]}")
    return "\n\n".join(contexts)


if llm_provider == "OpenAI":
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error(
            "OpenAI API key not found in secrets. Please configure it in "
            "`.streamlit/secrets.toml`",
            icon="🗝️",
        )
        st.stop()
    client = OpenAI(api_key=openai_api_key)
else:
    google_api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get(
        "GEMINI_API_KEY"
    )
    if not google_api_key:
        st.error(
            "Google API key not found in secrets. Please configure `GOOGLE_API_KEY` "
            "or `GEMINI_API_KEY` in `.streamlit/secrets.toml`",
            icon="🗝️",
        )
        st.stop()
    client = genai.Client(api_key=google_api_key)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to ask?"):
    source_urls = [url_1, url_2]
    source_context = build_url_context(source_urls)
    user_prompt = prompt
    if source_context:
        user_prompt = (
            "Use the following web sources as additional context:\n\n"
            f"{source_context}\n\nUser question: {prompt}"
        )

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_request = conversation_buffer(
        st.session_state.messages, max_tokens, model_to_use
    )
    request_tokens = count_tokens(messages_for_request, model_to_use)
    st.caption(f"Request context: {request_tokens:,} tokens")

    with st.chat_message("assistant"):
        if llm_provider == "OpenAI":
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_request,
                stream=True,
            )
            response = st.write_stream(stream)
        else:
            prompt_text = "\n\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages_for_request
            )
            response_obj = client.models.generate_content(
                model=model_to_use,
                contents=prompt_text,
            )
            response = response_obj.text
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
