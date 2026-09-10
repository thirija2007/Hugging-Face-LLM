import streamlit as st
from huggingface_hub import InferenceClient

st.title("My AI Chatbot")

token = st.secrets.get("HF_TOKEN")

if not token:
    st.error("HF_TOKEN is missing.")
    st.stop()

client = InferenceClient(
    provider="auto",
    api_key=token
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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