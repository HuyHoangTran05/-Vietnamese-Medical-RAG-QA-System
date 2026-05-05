import json
import re
from pathlib import Path
from typing import Dict, List


INPUT_PATH = Path("data/processed/combined_medical_corpus.jsonl")
OUTPUT_PATH = Path("data/processed/medical_chunks.jsonl")
import re
from typing import Dict, List


def looks_like_boilerplate(text: str) -> bool:
    lower = text.lower()

    boilerplate_patterns = [
        "để đặt lịch khám",
        "quý khách vui lòng",
        "hotline",
        "tải và đặt lịch",
        "ứng dụng myvinmec",
        "đặt lịch trực tiếp",
        "liên hệ tổng đài",
        "bấm số hotline",
        "theo dõi lịch và đặt hẹn",
    ]

    return any(pattern in lower for pattern in boilerplate_patterns)
def looks_like_administrative_text(text: str) -> bool:
    lower = text.lower()

    admin_terms = [
        "quyết định này",
        "thi hành quyết định",
        "chánh văn phòng",
        "chánh thanh tra",
        "cục trưởng",
        "vụ trưởng",
        "giám đốc sở y tế",
        "thủ trưởng",
        "bộ trưởng",
        "thứ trưởng",
        "căn cứ",
        "ban hành kèm theo",
    ]

    count = sum(term in lower for term in admin_terms)

    return count >= 2

def looks_like_table_of_contents(text: str) -> bool:
    lower = text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    toc_keywords = [
        "mục lục",
        "danh mục bảng",
        "danh mục hình",
        "danh mục biểu đồ",
        "danh mục sơ đồ",
        "danh mục chữ viết tắt",
    ]

    if any(keyword in lower for keyword in toc_keywords):
        return True

    if len(lines) < 3:
        return False

    toc_like_lines = 0

    for line in lines:
        line_lower = line.lower()

        if re.search(r"\.{3,}\s*\d+$", line):
            toc_like_lines += 1

        if re.match(r"^(bảng|hình|sơ đồ|biểu đồ)\s+\d+(\.\d+)*", line_lower):
            toc_like_lines += 1

        if re.search(r"\b(chương|mục|bảng|hình|biểu đồ|sơ đồ)\b", line_lower) and re.search(r"\d+$", line):
            toc_like_lines += 1

    return toc_like_lines / max(len(lines), 1) > 0.35
def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_noisy_lines(text: str) -> str:
    """
    Lọc rác ở cấp dòng, tổng quát cho PDF/article:
    - dòng quá ngắn
    - dòng toàn số/ký hiệu
    - watermark có timestamp/email
    - dòng giống header/footer/mã tài liệu
    """
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        # Bỏ dòng quá ngắn kiểu: "*", "7", "LAMA"
        if len(line) < 8:
            continue

        # Bỏ dòng toàn số hoặc gần như toàn số/ký hiệu
        letters = re.findall(r"[A-Za-zÀ-ỹ]", line)
        if len(letters) < 3:
            continue

        # Bỏ dòng có email/timestamp/watermark kỹ thuật
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", line):
            continue

        if re.search(r"\b\d{2}:\d{2}:\d{2}\b", line):
            continue

        if re.search(r"[\w\.-]+@[\w\.-]+", line):
            continue

        # Bỏ dòng có nhiều dấu gạch/chấm kiểu mục lục
        if re.search(r"\.{5,}", line):
            continue

        # Bỏ dòng có tỷ lệ ký tự lạ quá cao
        total_chars = len(line)
        alpha_chars = len(re.findall(r"[A-Za-zÀ-ỹ]", line))
        if total_chars > 0 and alpha_chars / total_chars < 0.35:
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def get_text_stats(text: str) -> Dict[str, float]:
    chars = len(text)
    letters = len(re.findall(r"[A-Za-zÀ-ỹ]", text))
    digits = len(re.findall(r"\d", text))
    spaces = len(re.findall(r"\s", text))
    words = re.findall(r"[A-Za-zÀ-ỹ]{2,}", text.lower())
    unique_words = set(words)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = [line for line in lines if len(line) < 30]

    return {
        "chars": chars,
        "letters": letters,
        "digits": digits,
        "spaces": spaces,
        "num_words": len(words),
        "num_unique_words": len(unique_words),
        "num_lines": len(lines),
        "num_short_lines": len(short_lines),
        "letter_ratio": letters / max(chars, 1),
        "digit_ratio": digits / max(chars, 1),
        "unique_word_ratio": len(unique_words) / max(len(words), 1),
        "short_line_ratio": len(short_lines) / max(len(lines), 1),
    }

def looks_like_repeated_header_footer(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        return False

    # Nếu nhiều dòng rất giống nhau hoặc trùng title tài liệu
    normalized_lines = [
        re.sub(r"\d+", "", line.lower()).strip()
        for line in lines
        if len(line.strip()) > 20
    ]

    if len(normalized_lines) < 2:
        return False

    repeated_count = len(normalized_lines) - len(set(normalized_lines))

    return repeated_count >= 1 and len(lines) <= 6

def looks_like_name_list(text: str) -> bool:
    """
    Nhận diện chunk kiểu danh sách tác giả/hội đồng:
    nhiều dòng ngắn, nhiều học hàm/học vị, ít nội dung y khoa.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 4:
        return False

    degree_patterns = [
        r"\bGS\b",
        r"\bPGS\b",
        r"\bTS\b",
        r"\bThS\b",
        r"\bBS\b",
        r"\bBSCKI\b",
        r"\bBSCKII\b",
        r"\bDS\b",
        r"\bCN\b",
    ]

    degree_count = 0

    for line in lines:
        for pattern in degree_patterns:
            if re.search(pattern, line):
                degree_count += 1
                break

    short_line_ratio = sum(len(line) < 60 for line in lines) / max(len(lines), 1)

    return degree_count >= 3 and short_line_ratio > 0.6


def is_low_information_chunk(text: str, source_type: str = "") -> bool:
    text = normalize_spaces(text)

    if not text:
        return True

    stats = get_text_stats(text)

    if looks_like_table_of_contents(text):
        return True

    if looks_like_boilerplate(text):
        return True

    if looks_like_administrative_text(text):
        return True

    if looks_like_repeated_header_footer(text):
        return True

    # Ngưỡng chung
    min_chars = 180
    min_words = 35

    # PDF cần ngưỡng cao hơn vì dễ có header/footer/caption/mảnh vỡ
    if source_type == "pdf":
        min_chars = 240
        min_words = 45

    if stats["chars"] < min_chars:
        return True

    if stats["num_words"] < min_words:
        return True

    if stats["letter_ratio"] < 0.45:
        return True

    if stats["digit_ratio"] > 0.25:
        return True

    if stats["num_lines"] >= 4 and stats["short_line_ratio"] > 0.75:
        return True

    if looks_like_table_of_contents(text):
        return True

    if looks_like_name_list(text):
        return True

    if source_type == "pdf" and starts_like_broken_fragment(text):
        return True

    if stats["num_words"] >= 50 and stats["unique_word_ratio"] < 0.25:
        return True
        
    
    return False
def starts_like_broken_fragment(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    # Bỏ tag trang nếu có
    text = re.sub(r"^\[Trang\s+\d+\]\s*", "", text).strip()

    if not text:
        return True

    first_token = text.split()[0]

    # Nếu từ đầu quá ngắn và viết thường -> có thể là mảnh vỡ
    if len(first_token) <= 3 and first_token[0].islower():
        return True

    # Nếu bắt đầu bằng dấu câu/ký tự lạ
    if re.match(r"^[,.;:)\]\-–]", text):
        return True

    return False
def load_jsonl(path: Path) -> List[Dict]:
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return docs


def split_by_paragraphs(text: str, max_chars: int = 900, overlap_chars: int = 150) -> List[str]:
    text = normalize_spaces(text)
    text = remove_noisy_lines(text)

    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)

            if len(para) <= max_chars:
                current = para
            else:
                # fallback nếu 1 paragraph quá dài
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = end - overlap_chars
                current = ""

    if current:
        chunks.append(current)

    return chunks


def make_qa_chunk(doc: Dict, chunk_id: int) -> Dict:
    question = doc.get("question", "").strip()
    answer = doc.get("answer", "").strip()

    text = f"Câu hỏi: {question}\nTrả lời: {answer}".strip()

    return {
        "chunk_id": f"chunk_{chunk_id:06d}",
        "doc_id": doc.get("doc_id"),
        "source": doc.get("source"),
        "source_type": doc.get("source_type"),
        "trust_level": doc.get("trust_level"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "file_path": doc.get("file_path"),
        "chunk_index": 0,
        "chunk_type": "qa_pair",
        "question": question,
        "answer": answer,
        "text": text,
    }


def make_text_chunks(doc: Dict, start_chunk_id: int) -> List[Dict]:
    content = doc.get("content", "")
    text_chunks = split_by_paragraphs(content)

    chunks = []
    valid_idx = 0

    for text in text_chunks:
        if is_low_information_chunk(text, source_type=doc.get("source_type", "")):
            continue

        title = doc.get("title", "")
        source = doc.get("source", "")
        source_type = doc.get("source_type", "")

        enriched_text = f"""
Tiêu đề tài liệu: {title}
Nguồn: {source}
Loại nguồn: {source_type}

Nội dung:
{text}
""".strip()

        chunk = {
            "chunk_id": f"chunk_{start_chunk_id + valid_idx:06d}",
            "doc_id": doc.get("doc_id"),
            "source": source,
            "source_type": source_type,
            "trust_level": doc.get("trust_level"),
            "title": title,
            "url": doc.get("url"),
            "file_path": doc.get("file_path"),
            "chunk_index": valid_idx,
            "chunk_type": "text_chunk",
            "question": "",
            "answer": "",
            "text": enriched_text,
            "raw_text": text,
        }

        chunks.append(chunk)
        valid_idx += 1

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. Please run scripts/merge_corpus_sources.py first."
        )

    docs = load_jsonl(INPUT_PATH)

    all_chunks = []
    chunk_id = 0

    for doc in docs:
        source_type = doc.get("source_type", "")

        if source_type == "qa_pair" and doc.get("question") and doc.get("answer"):
            chunk = make_qa_chunk(doc, chunk_id)

            if chunk["text"]:
                all_chunks.append(chunk)
                chunk_id += 1

        else:
            chunks = make_text_chunks(doc, chunk_id)

            all_chunks.extend(chunks)
            chunk_id += len(chunks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Loaded {len(docs)} documents")
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()