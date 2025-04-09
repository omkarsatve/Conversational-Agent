import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os

# One folder up from the current directory
SAVE_DIR = os.path.abspath(os.path.join(os.getcwd(), "../data"))
os.makedirs(SAVE_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(SAVE_DIR, "dxfactor_full_scrape.txt")

ROOT_URL = "https://dxfactor.com/"
WAIT_TIME = 5  # seconds

# === Setup Chrome in stealth mode ===
driver = uc.Chrome(version_main=134, headless=True)
driver.get(ROOT_URL)
time.sleep(WAIT_TIME)

def get_text_from_current_page():
    soup = BeautifulSoup(driver.page_source, "html.parser")

    content = []

    # Only extract in-page tags in natural DOM order
    for tag in soup.find_all(["title", "h1", "h2", "h3", "h4", "h5", "h6","p"]):
        text = tag.get_text(strip=True)
        if text and len(text) > 3:
            content.append(text)

    # Remove duplicates while keeping order
    seen = set()
    unique_content = []
    for line in content:
        if line not in seen:
            seen.add(line)
            unique_content.append(line)

    return "\n".join(unique_content)

# === Function: Extract first-level internal links ===
def get_first_level_links():
    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or "mailto:" in href or "tel:" in href:
            continue
        full_url = urljoin(ROOT_URL, href)
        if urlparse(full_url).netloc == urlparse(ROOT_URL).netloc:
            links.add(full_url)
    return list(links)

# === Scrape root page ===
print(f"[+] Scraping root page: {ROOT_URL}")
visited = set()
all_data = []

root_text = get_text_from_current_page()
all_data.append(("Home", ROOT_URL, root_text))
visited.add(ROOT_URL)

# === Find and scrape first-level links ===
print("[+] Collecting and scraping first-level links...")
first_level_links = get_first_level_links()

for link in first_level_links:
    if link in visited:
        continue
    try:
        driver.get(link)
        time.sleep(WAIT_TIME)
        page_text = get_text_from_current_page()
        title = link.rstrip("/").split("/")[-1].replace("-", " ").title()
        if not title:
            title = "No Title"
        all_data.append((title, link, page_text))
        visited.add(link)
        print(f"  ✓ Scraped: {link}")
    except Exception as e:
        print(f"  ✗ Failed: {link} — {e}")

# === Save results ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for title, url, text in all_data:
        f.write(f"\n\n========== {title} ==========\n")
        f.write(f"URL: {url}\n\n")
        f.write(text)

print(f"\n Scraping complete. Saved to '{OUTPUT_FILE}'")
driver.quit()
