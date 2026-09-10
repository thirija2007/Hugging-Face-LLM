import streamlit as st
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖"
)

# Page title
st.title("🤖 My AI Chatbot")

# Get Hugging Face token from Streamlit Secrets
token = st.secrets.get("HF_TOKEN")

if not token:
    st.error("HF_TOKEN is missing. Please add it to Streamlit Secrets.")
    st.stop()

# Create Hugging Face client
client = InferenceClient(
    provider="auto",
    api_key=token
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask me anything...")

if question:

    # Add and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=st.session_state.messages,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            st.markdown(answer)

        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })