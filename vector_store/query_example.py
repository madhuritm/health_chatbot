from sentence_transformers import SentenceTransformer
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from llama_cpp import Llama
import pandas as pd
import faiss

index=faiss.read_index("../index/diabetes_index.index")
df=pd.read_json("../data/chunks_with_embeddings.json", lines=True)
metadatas=df[['chunk_id', 'chunk_url', 'chunk_title','chunk_text']]

query="What are the symptoms of type2 diabetes?"
query_embedding=model.encode([query], convert_to_numpy=True).astype('float32')

##k=1 for right now becuase context window is small. 
D,I=index.search(query_embedding, k=1)

print(I)
print(D)

for idx in I[0]:
    print(metadatas[metadatas['chunk_id'] == int(idx)]['chunk_text'].values[0])
    print("-" * 40)


top_k_indices=I[0]
top_k_chunks = "\n\n".join(metadatas.iloc[idx]['chunk_text'] for idx in top_k_indices)
print("Top K Chunks:\n", top_k_chunks)

llm=Llama(model_path="../llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")

output=llm("Context: " + top_k_chunks + "\n\nQuestion: " + query , max_tokens=300)
print(output["choices"][0]["text"])
