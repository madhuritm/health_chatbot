from sentence_transformers import SentenceTransformer
import pandas as pd

# This script generates embeddings for text chunks using a pre-trained SentenceTransformer model and saves the results to a JSON file.
# load the csv into pandas DataFrame
df=pd.read_csv("../chunking/chunks.csv")
print(df.columns)

# Load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#generate embeddings for the 'chunk_text' column and put it into a new column called 'embedding' in the DataFrame df
embeddings=model.encode(df['chunk_text'].tolist(), show_progress_bar=True)
df['embedding'] =embeddings.tolist()

#save the DataFrame and store it in a json file
df_to_save=df[['chunk_id', 'chunk_text','embedding' ]]
df_to_save.to_json("chunks_with_embeddings.json", orient='records', lines=True)