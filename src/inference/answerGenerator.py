#!/usr/bin/env python3
import pandas as pd
import os, argparse, sys
from sklearn.metrics.pairwise import cosine_similarity
from src.config.config import client, index, chunks_df, embedder

# Make Python aware of the Graphweaver project
GRAPHWEAVER_ROOT = "/home/ec2-user/Graphweaver"
if GRAPHWEAVER_ROOT not in sys.path:
    sys.path.append(GRAPHWEAVER_ROOT)

from graphRetriever.Text2CypherQuery import text2cypher



api_key = os.getenv("OPENAI_API_KEY")

def extract_k_chunks(query:str) -> list[str, str]:
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

def extract_answer_with_rag(query: str) -> list[str, str]:
    [top_k_chunks,source_url] = extract_k_chunks(query)
    answer=get_llm_response(query, top_k_chunks)
    sims=cal_cosine_similarity(top_k_chunks, answer)
    print(f"Similarity is {sims[0][0]}")
    return [answer, source_url]

def get_answer_with_graph_and_rag(query: str):
    #1) Ty graph retrieval first
    graph_answer, cypher = text2cypher(query)

    if graph_answer:
        print("using graph answer path")
        #use graph answer as context to llm
        graph_context = f"Answer derieved from a medical knowledge graph: {graph_answer}"
        final_answer = get_llm_response(query, graph_context)
        return final_answer, []
    
    print("Falling back to RAG path")
    # 2) Fall back to RAG if graph has no answer
    return extract_answer_with_rag(query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    args = ap.parse_args()
    query = args.question

    answer, source_url = get_answer_with_graph_and_rag(query)
     # If LLM says it does not know, just return that
    if answer == "I don’t know based on the provided information.":
        output = answer
    else:
        if source_url:
            answer = answer + "\n" + "sources"
            output = answer + "\n" + "\n".join(idx["url"] for idx in source_url)
        else:
            # Graph path: no sources list
            output = answer

    print(f"answer:{output}")
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
