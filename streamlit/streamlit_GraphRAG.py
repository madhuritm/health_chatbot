import os
import streamlit as st
import requests
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()
base_url = (os.getenv("HEALTH_CHATBOT_BASE_API_URL") or "").strip().rstrip("/")

if not base_url:
    st.error("Missing HEALTH_CHATBOT_BASE_API_URL in .env")
    st.stop()

# Title
st.title("Diabetes Health Chatbot")

# User input
user_query = st.text_input("Ask a question:")

if "session_id" not in st.session_state:
    st.session_state["session_id"]=str(uuid4())

# On button click, send to FastAPI (or your Llama backend)
if st.button("Get Answer"):
    query = user_query.strip()
    if not query:
        st.warning("Please enter a question.")
        st.stop()
    try:
        # Example: your FastAPI endpoint
        response = requests.post(
            f"{base_url}/ask",
            json={"query": query, "session_id": st.session_state["session_id"]},
            timeout=30,        
        )
        response.raise_for_status()        
        output = response.json()        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        st.stop()
    except ValueError:
        st.error("API returned non JSON response.")
        st.stop()  

    st.text_area("Answer:", value=str(output.get("answer", "")), height=200)
    st.text_area("Sources:", value=str(output.get("sources", "")), height=100)