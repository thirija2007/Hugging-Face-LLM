import os
import streamlit as st
from huggingface_hub import InferenceClient

st.title("Hugging Face LLM")

question = st.text_input("Enter your question:")

token = os.getenv("HF_TOKEN")

if not token:
    st.error(
        "HF_TOKEN was not found. Set it in the same terminal before running Streamlit."
    )
    st.stop()

if st.button("Ask AI"):
    if question.strip():
        try:
            client = InferenceClient(
                provider="auto",
                api_key=token
            )

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                max_tokens=1000
            )

            st.write("### AI Response")
            st.write(response.choices[0].message.content)

        except Exception as error:
            st.error(f"Request failed: {error}")
    else:
        st.warning("Please enter a question.")