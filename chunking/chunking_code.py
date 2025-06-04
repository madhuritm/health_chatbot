import csv
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

inputCSV="../scraping/scraped_contents.csv"

def getChunks(text: str, chunk_size: int = 100, overlap: float=0.2)->list[str]:
    sentences=sent_tokenize(text)
    chunks=[]
    current_chunk=[]

    i=0
    while i < len(sentences):
        current_chunk=[]
        total_words=0

        while i < len(sentences) and total_words < chunk_size:
            sentence=sentences[i]
            word_count = len(sentence.split())
            if total_words + word_count > chunk_size:
                break
            current_chunk.append(sentence)
            total_words += word_count
            i+=1

        chunks.append(" ".join(current_chunk))

        i=i-int(overlap*len(current_chunk))
    return chunks

def cleanData(inputCSV: str)->list[list[str]]:
    result=[]
    with open(inputCSV, newline='') as f:
        reader=csv.reader(f)
        for line in reader:
            if line[0] != 'URL':
                chunks=getChunks(line[2])
                result.append([line[0], line[1], chunks])
            
    return result

def save_chunks_to_csv(chunks, out_file="chunks.csv"):
    with open(out_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx, chunk in enumerate(chunks):
            writer.writerow([chunk[0], chunk[1],idx, chunk])


if __name__ == "__main__":
    resultChunks=cleanData(inputCSV)
    save_chunks_to_csv(resultChunks, out_file="chunks.csv")    

    

    print(f"Completed successfully")
