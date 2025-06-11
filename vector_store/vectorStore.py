import faiss
import numpy as np
import pandas as pd

df=pd.read_json("../data/chunks_with_embeddings.json", lines=True)
print(df.columns)

embedding_dim=384
index=faiss.IndexFlatL2(embedding_dim)

embeddings=df['embedding'].to_list()


embeddings_array=np.array(embeddings, dtype=np.float32)
metadatas=df[['chunk_id', 'chunk_url', 'chunk_title','chunk_text']]

index.add(embeddings_array)
metadata_store=metadatas

faiss.write_index(index,"../index/diabetes_index.index")


