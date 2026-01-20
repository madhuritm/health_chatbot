#!/usr/bin/env python3

import faiss, argparse, sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    df=pd.read_json(input_file, lines=True)
    print(df.columns)

    embedding_dim=384
    index=faiss.IndexFlatL2(embedding_dim)

    embeddings=df['embedding'].to_list()
    embeddings_array=np.array(embeddings, dtype=np.float32)
    metadatas=df[['chunk_id', 'chunk_url', 'chunk_title','chunk_text']]

    index.add(embeddings_array)
    metadata_store=metadatas

    faiss.write_index(index, output_file)
    return 0

if __name__ == "__main__":
    sys.exit(main())


