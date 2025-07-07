import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from config import client, index, chunks_df, embedder


api_key = os.getenv("OPENAI_API_KEY")

def extract_k_chunks(query:str, ) -> tuple[str, list[dict]]:
    query_embedding=embedder.encode([query], convert_to_numpy=True).astype('float32')
    #get the closest 3 embeddings
    D,I=index.search(query_embedding, k=3)
    source_url=[{"url":chunks_df.iloc[idx]['chunk_url']}for idx in I[0]]
    #create a chunk of the closest emebddings
    top_k_indices=I[0]
    top_k_chunks = "\n\n".join(chunks_df.iloc[idx]['chunk_text'] for idx in top_k_indices)
    return [top_k_chunks, source_url]

def get_llm_response(query:str, top_k_chunks:str)->str:
    #send query to openAI
    response=client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"system", "content":"You are a safe and helpful chatbot. If the answer is not in the context, say 'I don’t know based on the provided information. Use only the context to answer in not more than 100 words"},
        {"role":"system", "content":f"Context:\n{top_k_chunks}"},
        {"role":"user", "content":query}
    ])
    return response.choices[0].message.content

def cal_cosine_similarity(top_k_chunks:str, answer:str)->float:
    chunk_embedding=embedder.encode(top_k_chunks)
    answer_embedding=embedder.encode(answer)
    sims=cosine_similarity([chunk_embedding], [answer_embedding])
    return sims


#create embedding from query
query="what is migraine?"
[top_k_chunks,source_url] = extract_k_chunks(query)
answer=get_llm_response(query, top_k_chunks)

sims=cal_cosine_similarity(top_k_chunks, answer)

print(sims[0][0])


if answer == "I don’t know based on the provided information.":    
    output=answer
else:
    answer=answer + "\n" + "sources"
    output=answer+"\n"+"\n".join(idx["url"] for idx in source_url)

print(f"answer:{output}")

print("Done!")