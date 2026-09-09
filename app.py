import os
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hugging Face Chatbot")

st.title("💬 Hugging Face Chatbot")

# Read token from local environment or Streamlit Cloud Secrets
token = os.getenv("HF_TOKEN")

if not token:
    token = st.secrets.get("HF_TOKEN")

if not token:
    st.error("HF_TOKEN is missing. Add it in Streamlit Cloud Secrets.")
    st.stop()

client = InferenceClient(
    provider="auto",
    api_key=token
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask me anything...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=st.session_state.messages,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as error:
        st.error(f"Request failed: {error}")
