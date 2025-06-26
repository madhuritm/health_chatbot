from sentence_transformers import SentenceTransformer
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import faiss
import os

api_key = os.getenv("OPENAI_API_KEY")

#to load.env
#load_dotenv()



client=OpenAI(api_key=api_key)

#read the diabetes index that has been created (FAISS index helps to do a very quick semantic search for the query embedding by using optimized DS)
index=faiss.read_index("../index/diabetes_index.index")

#load the embeddings and stored related metadata 
df=pd.read_json("../data/chunks_with_embeddings.json", lines=True)
metadatas=df[['chunk_id', 'chunk_url', 'chunk_title','chunk_text']]

#create embedding from query
query="How can I prevent high blood sugar?"
query_embedding=model.encode([query], convert_to_numpy=True).astype('float32')

#get the closest 3 embeddings
D,I=index.search(query_embedding, k=3)

print(I)
print(D)

source_url=[
    {
        "url":metadatas.iloc[idx]['chunk_url']
    }
    for idx in I[0]
]

for idx in I[0]:
    print(metadatas[metadatas['chunk_id'] == int(idx)]['chunk_text'].values[0])
    print("-" * 40)


#create a chunk of the closest emebddings
top_k_indices=I[0]
top_k_chunks = "\n\n".join(metadatas.iloc[idx]['chunk_text'] for idx in top_k_indices)

#send query to openAI
response=client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"system", "content":"You are a safe and helpful chatbot. Use only the context to answer in not more than 100 words"},
        {"role":"system", "content":f"Context:\n{top_k_chunks}"},
        {"role":"user", "content":query}
    ]
)


answer = response.choices[0].message.content

if not answer:
    answer = "We don't have an answer to that question yet."

answer=answer + "\n" + "sources"
output=answer+"\n"+"\n".join(idx["url"] for idx in source_url)

print(f"answer:{output}")

print("Done!")
