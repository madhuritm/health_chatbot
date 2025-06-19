import streamlit as st
import requests

# Title
st.title("Diabetes Health Chatbot")

# User input
user_query = st.text_input("Ask a question:")

# On button click, send to FastAPI (or your Llama backend)
if st.button("Get Answer"):
    if user_query:
        # Example: your FastAPI endpoint
        response = requests.post(
            "http://52.200.142.157:8000/ask",
            json={"query": user_query}
        )
        answer = response.json()["answer"]
        st.text_area("Answer:", value=answer, height=400)

    else:
        st.warning("Please enter a question.")
print("end")