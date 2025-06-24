##python script to use openai api to get the answer from RAG chunks
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

#load .env file
load_dotenv()

#set up the openai client
client=OpenAI()

#set up fast api
app=FastAPI()

#load the index
index=faiss.read_index("../index/diabetes_index.index")

#load the chunks
chunks_df=pd.read_json("../data/chunks_with_embeddings.json", lines=True)


#load the transformer model
embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class Question(BaseModel):
    query:str

@app.post("/ask")
def ask(question:Question):
    ##embed the query
    query_embedding=embedder.encode([question.query]).astype('float32')
    ##find the indices closest in the metadata
    D,I=index.search(query_embedding, k=3)

    ##extract these chunks and make a str
    top_k_chunks="\n\n".join(chunks_df.iloc[idx]['chunk_text'] for idx in I[0])

    ##write the openai api
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role":"system", "content":"You are a safe and helpful chatbot. Use only the context to answer in not more than 100 words"},
        {"role":"system", "content":f"Context:\n{top_k_chunks}"},
        {"role":"user", "content":question.query}
    ]
    )

    answer=response.choices[0].message.content

    if not answer:
        answer="Answer yet to be found"
    
    return{"answer":answer.strip()}

    ##if answer is not found in RAG chunks return answer not found
