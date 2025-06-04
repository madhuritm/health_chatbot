import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque

BASE_URL = "https://medlineplus.gov/diabetes.html"
DOMAIN = "medlineplus.gov"

def is_valid_link(base_url, href):
    full_url = urljoin(base_url, href)
    parsed = urlparse(full_url)

    return (
        DOMAIN in parsed.netloc and
        "/diabetes" in parsed.path and
        not parsed.fragment  # ignore #anchors
    )

def get_links_from_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(url, href)
            if is_valid_link(url, href):
                links.add(full_url)

        return links
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return set()

def crawl_diabetes_links(seed_url, max_pages=100):
    visited = set()
    to_visit = deque([seed_url])
    all_links = set()

    while to_visit and len(visited) < max_pages:
        current = to_visit.popleft()
        if current in visited:
            continue

        visited.add(current)
        print(f"Visiting: {current}")
        found_links = get_links_from_page(current)

        for link in found_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        all_links.update(found_links)

    return sorted(all_links)

if __name__ == "__main__":
    all_diabetes_links = crawl_diabetes_links(BASE_URL, max_pages=100)

    print(f"\nTotal unique diabetes-related links found: {len(all_diabetes_links)}")
    with open("diabetes_recursive_links.txt", "w") as f:
        for link in all_diabetes_links:
            f.write(link + "\n")
