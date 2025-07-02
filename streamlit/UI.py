import streamlit as st
import requests
from uuid import uuid4


# Title
st.title("Diabetes Health Chatbot")

# User input
user_query = st.text_input("Ask a question:")

if "session_id" not in st.session_state:
    st.session_state["session_id"]=str(uuid4())

# On button click, send to FastAPI (or your Llama backend)
if st.button("Get Answer"):
    if user_query:
        # Example: your FastAPI endpoint
        response = requests.post(
            "http://52.200.142.157:8000/ask",
            json={"query": user_query, "session_id": st.session_state["session_id"]}
        
        )
        output = response.json()        
        answer = output.get("answer", "No sources found")
        sources = output.get("sources", "No sources found.")
        chunks=output.get("chunks", "None")

        st.text_area("Answer:", value=answer, height=200)
        st.text_area("sources:", value=sources, height=100)
        st.text_area("chunks:", value=chunks, height=500 )
    else:
        st.warning("Please enter a question.")
print("end")