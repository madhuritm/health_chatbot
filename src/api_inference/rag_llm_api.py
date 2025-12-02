##python script to use openai api to get the answer from RAG chunks
from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from config import client,index,chunks_df,embedder
import spacy

#set up fast api
app=FastAPI()
sessions={}
nlp=spacy.load("en_core_web_sm")

def cal_cosine_similarity(context:str, answer:str)->float:
    context_embedding=embedder.encode(context)
    answer_embedding=embedder.encode(answer)
    sim=cosine_similarity([context_embedding], [answer_embedding])
    return sim[0][0]

def extract_entities(text):
    doc=nlp(text)
    return set(ent.text.lower() for ent in doc.ents)

def detect_hallucinated_entities(answer, context):
    answer_entities=extract_entities(answer)
    context_entities=extract_entities(context)

    hallucinated=answer_entities-context_entities
    return hallucinated


#this function extracts k closest chunks to the query from the embedding
def extract_k_chunks(query:str)-> tuple[str, list[dict]]:
    query_embedding=embedder.encode([query]).astype('float32')
    D,I=index.search(query_embedding, k=3)
    source_url=[{'url':chunks_df.iloc[idx]['chunk_url']} for idx in I[0]]
    top_k_chunks="\n\n".join(chunks_df.iloc[idx]['chunk_text'] for idx in I[0])
    return [top_k_chunks, source_url]

#gets the llm response
def get_llm_response(history:list[dict], query:str, top_k_chunks:str) -> str:
    history.append({"role":"user", "content":query})
    system_prompt={"role":"system", "content":"You are a safe and helpful chatbot. If answer is not in the context, say 'I don't know based on the provided information'. Use current context to answer in not more than 100 words"}
    context_prompt={"role":"system", "content":f"Context:\n{top_k_chunks}\n\nQuestion:{query}"}
    messages=[system_prompt, context_prompt] + history
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        top_p=1
    )
    return response.choices[0].message.content


class Question(BaseModel):
    query:str
    session_id:Optional[str]=None

@app.post("/ask")
def ask(question: Question):
    if not question.session_id:
        question.session_id=str(uuid4())

    [top_k_chunks, source_url]=extract_k_chunks(question.query)

    history=sessions.get(question.session_id, [])   
    answer=get_llm_response(history, question.query, top_k_chunks)
    history.append({"role":"assistant", "content":answer})    
    sessions[question.session_id]=history
    sim=cal_cosine_similarity(top_k_chunks, answer)

    if sim < 0.5:
        return{"answer":"At this point we don't have an answer", "source":""}
    else:
        return{"answer":answer, "sources":source_url}

    
