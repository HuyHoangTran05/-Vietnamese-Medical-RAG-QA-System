import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm


PDF_DIR = Path("data/raw/pdfs")
OUTPUT_PATH = Path("data/processed/pdf_corpus.jsonl")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_doc_id(file_path: Path) -> str:
    raw = str(file_path).encode("utf-8")
    file_hash = hashlib.md5(raw).hexdigest()[:12]
    return f"pdf_{file_hash}"


def extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))

    pages = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)

        if text:
            pages.append(f"[Trang {page_idx}]\n{text}")

    return "\n\n".join(pages)


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    docs = []

    for pdf_file in tqdm(pdf_files, desc="Ingesting PDFs"):
        try:
            content = extract_pdf_text(pdf_file)

            if len(content) < 300:
                print(f"Skip short PDF: {pdf_file}")
                continue

            doc = {
                "doc_id": make_doc_id(pdf_file),
                "source": "manual_pdf",
                "source_type": "pdf",
                "trust_level": "high",
                "title": pdf_file.stem,
                "file_path": str(pdf_file),
                "content": content,
            }

            docs.append(doc)

        except Exception as e:
            print(f"Failed to ingest {pdf_file}: {e}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Saved {len(docs)} PDF docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()