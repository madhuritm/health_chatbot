#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import pandas as pd, argparse, sys

def main():
    # This script generates embeddings for text chunks using a pre-trained SentenceTransformer model and saves the results to a JSON file.
    # load the csv into pandas DataFrame
    ap =argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    df=pd.read_csv(input_file)
    print(df.columns)

    # Load the model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    #generate embeddings for the 'chunk_text' column and put it into a new column called 'embedding' in the DataFrame df
    embeddings=model.encode(df['chunk_text'].tolist(), show_progress_bar=True)
    df['embedding'] =embeddings.tolist()
    #save the DataFrame and store it in a json file
    df.to_json(output_file, orient='records', lines=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())