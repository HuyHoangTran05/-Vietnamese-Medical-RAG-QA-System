import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm


URLS_PATH = Path("data/sources/medical_urls.csv")
OUTPUT_PATH = Path("data/processed/web_article_corpus.jsonl")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def make_doc_id(url: str) -> str:
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"web_{url_hash}"


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.title and soup.title.text:
        return clean_text(soup.title.text)

    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" "))

    return ""


def fetch_and_extract(url: str):
    headers = {
        "User-Agent": "VietnameseMedicalRAGBot/0.1 educational project"
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    html = response.text

    title = extract_title(html)

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )

    content = clean_text(extracted or "")

    return title, content


def main():
    if not URLS_PATH.exists():
        raise FileNotFoundError(f"Cannot find {URLS_PATH}")

    df = pd.read_csv(URLS_PATH)

    required_cols = {"source_name", "source_type", "trust_level", "url"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns in {URLS_PATH}: {missing_cols}")

    docs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting web articles"):
        source_type = str(row["source_type"]).strip()

        if source_type != "web_article":
            continue

        url = str(row["url"]).strip()
        source_name = str(row["source_name"]).strip()
        trust_level = str(row["trust_level"]).strip()

        try:
            title, content = fetch_and_extract(url)

            if len(content) < 300:
                print(f"Skip short content: {url}")
                continue

            doc = {
                "doc_id": make_doc_id(url),
                "source": source_name,
                "source_type": "web_article",
                "trust_level": trust_level,
                "title": title,
                "url": url,
                "content": content,
            }

            docs.append(doc)

        except Exception as e:
            print(f"Failed to ingest {url}: {e}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Saved {len(docs)} web article docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()