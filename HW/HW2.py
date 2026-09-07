import streamlit as st
from openai import OpenAI
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
    "Enter a URL below and choose a summary format. GPT will summarize it! "
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
use_advanced_model = st.sidebar.checkbox(
    "Use advanced model",
    value=False,
    help="Check to use gpt-5-mini (advanced). Uncheck to use gpt-5-nano (basic)."
)

model_name = "gpt-5-mini" if use_advanced_model else "gpt-5-nano"
st.sidebar.markdown(f"**Selected Model:** `{model_name}`")

# Get OpenAI API key from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found in secrets. Please configure it in `.streamlit/secrets.toml`", icon="🗝️")
    st.stop()

if openai_api_key:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

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
        
        messages = [
            {
                "role": "user",
                "content": f"Here's a document:\n\n{document}\n\n---\n\n{prompt}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
