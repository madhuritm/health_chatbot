from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import re


app=FastAPI()

#FAISS index
index=faiss.read_index("../index/diabetes_index.index")

#chunks metadata
chunks_df=pd.read_json("../data/chunks_with_embeddings.json", lines=True)

#embedding model
embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#local LLama
llm=Llama(model_path="../llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=4096)


class Question(BaseModel):
    query:str

#API endpoint
@app.post("/ask")
def ask(question:Question):
    query_embedding=embedder.encode([question.query]).astype('float32')
    D,I = index.search(query_embedding, k=3)

    top_chunks="\n\n".join(chunks_df.iloc[idx]['chunk_text'] for idx in I[0])

   # output=llm("Context: " + top_chunks + "\n\nQuestion: " + question.query , max_tokens=300,  stop=["\n\n", "\nQuestion:", "Question:", "</s>"] )

    output = llm.create_completion(
    prompt="Context: " + top_chunks + "\n\nQuestion: " + question.query,
    max_tokens=300,
    temperature=0,
    top_p=1,
    stop=["\n\n", "\nQuestion:", "Question:", "</s>"]
)

    answer = output["choices"][0]["text"]
    answer = re.split(r"\n[A-Z][^:]*:", answer)[0]  # split at headings
    answer = re.split(r"[A-Z][^\.?!]*\?", answer)[0]  # split at follow-up questions

    if not answer:
        answer = "We don't have an answer to that question yet."
    return {"answer": answer.strip()}