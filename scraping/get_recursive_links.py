#!/usr/bin/env python3
import requests, argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

class Crawler:
    def __init__(self, domain: str, base_url: str, subject: str):
        self.domain = domain
        self.base_url =base_url
        self.subject = subject

    def is_valid_link(self, url, href):
        full_url = urljoin(url, href)
        parsed = urlparse(full_url)

        return (
            self.domain in parsed.netloc and
            "/"+ self.subject in parsed.path and
            not parsed.fragment  # ignore #anchors
        )

    def get_links_from_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            links = set()

            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin(url, href)
                if self.is_valid_link(url, href):
                    links.add(full_url)

            return links
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return set()

    def crawl_links(self, max_pages=100):
        visited = set()
        to_visit = deque([self.base_url])
        all_links = set()

        while to_visit and len(visited) < max_pages:
            current = to_visit.popleft()
            if current in visited:
                continue

            visited.add(current)
            print(f"Visiting: {current}")
            found_links = self.get_links_from_page(current)

            for link in found_links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

            all_links.update(found_links)

        return sorted(all_links)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--BASE_URL", required = True)
    ap.add_argument("--DOMAIN", required = True)   
    ap.add_argument("--output_file", required = True) 
    ap.add_argument("--subject", required = True)
    ap.add_argument("--max_links", default=100)    
    args = ap.parse_args()

    BASE_URL = args.BASE_URL
    DOMAIN = args.DOMAIN
    output_file = args.output_file
    subject = args.subject
    max_links = args.max_links

    crawler = Crawler(DOMAIN, BASE_URL, subject)
    links = crawler.crawl_links(max_links)

    print(f"\nTotal unique {subject} related links found: {len(links)}")
    with open(output_file, "w") as f:
        for link in links:
            f.write(link + "\n")
