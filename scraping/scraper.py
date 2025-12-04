from bs4 import BeautifulSoup
import requests, argparse
from urllib.parse import urlparse, urljoin
from collections import deque
import csv


def scraper(inputURL:str)->list[str]:
    #do HTTP request and get the whole page
    eachLine=[]
    try:
        response=requests.get(inputURL, timeout=10)  
        response.raise_for_status()  
        soup=BeautifulSoup(response.text, "html.parser")

        #title
        title=soup.title.string.strip() if soup.title else "No title"

        #Try to extract main content
        #First try <article>, then fallback to <div id="main"> or <main>
        content_tag=soup.find("article")
        if not content_tag:
            content_tag=soup.find("div", id="main") or soup.find("main")
        
        if content_tag:
            for junk in content_tag.find_all(["nav", "footer", "aside", "script", "style"]):
                junk.decompose()
            content=content_tag.get_text(separator="\n", strip=True)
        else:
            content="No main content found"
        
        return [inputURL, title, content]
    
    except Exception as e:
        print(f"error fetching {inputURL}:{e}")
        return []
    
    

    #Write conditions to look for the relevant content. 
    #put the URL, title name, content into a List
    #return this List
    



def scraping(inputFile:str)->list[list[str]]:
    #open the input file
    result=[]
    with open(inputFile, "r", encoding="cp1251") as f:
        for line in f:
            output_value = scraper(line.strip())
            result.append(output_value)
    

    #open the output file
    #loop over each html link
        #call function scraper - input url, output - Title, content
        #Enter this into the output file
    return result



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    result=scraping(input_file)

    with open(output_file, "w", newline='',encoding="utf-8") as f:
        writer=csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["URL", "Title", "Content"])
        writer.writerows(result)


    print(f"Completed scraping")
