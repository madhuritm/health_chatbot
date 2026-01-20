from dotenv import load_dotenv
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

load_dotenv()
client=OpenAI()
index=faiss.read_index("/home/ec2-user/health_chatbot/index/diabetes_index.index")
chunks_df=pd.read_json("/home/ec2-user/health_chatbot/data/chunks_with_embeddings.json", lines=True)
embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")