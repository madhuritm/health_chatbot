##python script to use openai api to get the answer from RAG chunks
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
from uuid import uuid4
from typing import Optional

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
metadatas=chunks_df[['chunk_id', 'chunk_url', 'chunk_title','chunk_text']]

#load the transformer model
embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sessions={}

class Question(BaseModel):
    query:str
    session_id:Optional[str]=None

@app.post("/ask")
def ask(question:Question):
    if not question.session_id:
        question.session_id=str(uuid4())

    
    history=sessions.get(question.session_id, [])
    history.append({"role":"user", "content":question.query})    
    system_prompt={"role":"system", "content":"You are a safe and helpful chatbot. If the answer is not in the context, say 'I don’t know based on the provided information'. Use only the context to answer in not more than 100 words"}
      


    ##embed the query
    query_embedding=embedder.encode([question.query]).astype('float32')
    ##find the indices closest in the metadata
    D,I=index.search(query_embedding, k=3)

    source_url=[
        {
            'url' : metadatas.iloc[idx]['chunk_url']
        }
        for idx in I[0]
    ]

    ##extract these chunks and make a str
    top_k_chunks="\n\n".join(chunks_df.iloc[idx]['chunk_text'] for idx in I[0])   

    context_prompt={"role":"system", "content":f"Context:\n{top_k_chunks}\n\nQuestion: {question.query}"}


    ##write the openai api
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_prompt, context_prompt] + history,
        temperature=0,
        top_p=1
    )

    
    answer=response.choices[0].message.content

    history.append({"role":"assistant", "content":answer})

    
    print("Done")
    
    return{"answer":answer, "sources":source_url, "chunks":top_k_chunks}

    
