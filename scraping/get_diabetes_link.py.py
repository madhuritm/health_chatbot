import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://medlineplus.gov/diabetes.html"
DOMAIN = "medlineplus.gov"

def get_internal_links(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request failed

    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(url, href)  # Convert relative links to full URLs

        # Keep only links within medlineplus.gov and under the /diabetes path
        parsed = urlparse(full_url)
        if DOMAIN in parsed.netloc and "/diabetes" in parsed.path:
            links.add(full_url)

    return sorted(links)

if __name__ == "__main__":
    diabetes_links = get_internal_links(BASE_URL)
    
    print(f"Found {len(diabetes_links)} links:")
    for link in diabetes_links:
        print(link)

    # Save to file
    with open("diabetes_links.txt", "w") as f:
        for link in diabetes_links:
            f.write(link + "\n")
