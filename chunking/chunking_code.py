import csv
import nltk
import os
import math
nltk.download('punkt')
from nltk.tokenize import sent_tokenize



nltk_data_path = "/home/ec2-user/nltk_data"
nltk.data.path.insert(0, nltk_data_path)  # <--- this is the key line

# Just for safety, ensure it's downloaded
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)

inputCSV="../scraping/scraped_contents.csv"


print("NLTK search paths:", nltk.data.path)

def getChunks(text: str, chunk_size: int = 200, overlap: float=0.4)->list[str]:
    sentences=sent_tokenize(text)
    chunks=[]
    current_chunk=[]

    print("Entered getchunks")

    i=0
    while i < len(sentences):
        current_chunk=[]
        total_words=0

        while i < len(sentences) and total_words < chunk_size:
            sentence=sentences[i]
            word_count = len(sentence.split())
            if word_count > chunk_size and total_words==0:
                current_chunk.append(sentence)
                total_words += word_count
                i += 1
                break
            if total_words + word_count > chunk_size:
                break
            current_chunk.append(sentence)
            total_words += word_count
            i+=1

        chunks.append(" ".join(current_chunk))
        if(i==len(sentences)):
            return chunks

        i=i-round(overlap*len(current_chunk))
    return chunks

def cleanData(inputCSV: str)->list[list[str]]:
    result=[]
    with open(inputCSV, newline='') as f:
        reader=csv.reader(f)
        for line in reader:
            if line and line[0] != 'URL':
                chunks=getChunks(line[2])
                for temp in chunks:
                    result.append([line[0], line[1], temp])
            
    return result

def save_chunks_to_csv(chunks, out_file="chunks.csv"):
    with open(out_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx, chunk in enumerate(chunks):
            writer.writerow([chunk[0], chunk[1],idx, chunk[2]])


if __name__ == "__main__":
    resultChunks=cleanData(inputCSV)
    save_chunks_to_csv(resultChunks, out_file="chunks.csv")    

    

    print(f"Completed successfully")
