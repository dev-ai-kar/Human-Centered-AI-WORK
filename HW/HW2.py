import streamlit as st
from openai import OpenAI
from google import genai
import requests
from bs4 import BeautifulSoup


def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as error:
        print(f"Error reading {url}: {error}")
        return None

# Show title and description.
st.title("📄 Document summarization")
st.write(
    "Enter a URL below and choose a summary format. Your selected LLM will summarize it! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

url = st.text_input("Website URL", placeholder="https://example.com/article")

# Sidebar: Summary options and model selection
st.sidebar.header("⚙️ Options")

# Summary options
st.sidebar.subheader("📝 Summary Format")
summary_option = st.sidebar.radio(
    "Choose a summary format:",
    options=[
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points"
    ],
    index=0
)

output_language = st.sidebar.selectbox(
    "Choose an output language:",
    options=["English", "French", "Spanish"],
    index=0,
)

# Model selection
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
    model_name = "gpt-5-mini" if use_advanced_model else "gpt-5-nano"
else:
    model_name = "gemini-3.8-flash" if use_advanced_model else "gemini-3.5-flash-lite"
st.sidebar.markdown(f"**Selected Model:** `{model_name}`")

if llm_provider == "OpenAI":
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OpenAI API key not found in secrets. Please configure it in `.streamlit/secrets.toml`", icon="🗝️")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
else:
    google_api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not google_api_key:
        st.error("Google Gemini API key not found in secrets. Please configure `GOOGLE_API_KEY` in `.streamlit/secrets.toml`", icon="🗝️")
        st.stop()

    client = genai.Client(api_key=google_api_key)

if url:
    document = read_url_content(url)

    if not document or not document.strip():
        st.error("Could not read that URL or it did not contain any readable text.")
        st.stop()

    # Build the prompt based on the selected summary format.
    if summary_option == "100 words":
        prompt = "Please summarize the following document in exactly 100 words."
    elif summary_option == "2 connecting paragraphs":
        prompt = "Please summarize the following document in 2 well-connected paragraphs."
    else:
        prompt = "Please summarize the following document in 5 bullet points."

    prompt += f" Output the summary entirely in {output_language}."
    input_text = f"Here's a document:\n\n{document}\n\n---\n\n{prompt}"

    if llm_provider == "OpenAI":
        messages = [{"role": "user", "content": input_text}]
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)
    else:
        interaction = client.interactions.create(
            model=model_name,
            input=input_text,
        )
        st.write(interaction.output_text)
