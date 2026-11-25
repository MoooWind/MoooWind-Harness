import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from collections import Counter
from functools import lru_cache

from docx import Document


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "book_library.json"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 160


def _read_book(path):
    extension = path.suffix.lower()
    if extension == ".txt":
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gb18030")
        return [("全文", text)]

    if extension == ".docx":
        document = Document(path)
        sections = [
            (f"第 {index} 段", paragraph.text)
            for index, paragraph in enumerate(document.paragraphs, start=1)
            if paragraph.text.strip()
        ]
        for table_index, table in enumerate(document.tables, start=1):
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in table.rows
            ]
            table_text = "\n".join(row for row in rows if row.strip())
            if table_text:
                sections.append((f"第 {table_index} 个表格", table_text))
        return sections

    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                "读取 PDF 需要安装 pypdf，请运行："
                "python -m pip install -r requirements-book-library.txt"
            ) from error

        reader = PdfReader(str(path))
        sections = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                sections.append((f"第 {page_number} 页", text))
        if not sections:
            raise ValueError(
                f"{path.name} 没有可提取的文本，可能是扫描版 PDF；当前版本尚不支持 OCR。"
            )
        return sections

    raise ValueError("目前支持 .pdf、.docx 和 .txt 文件")


def _split_sections(source, sections):
    chunks = []
    for location, text in sections:
        normalized = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(normalized):
            end = min(start + CHUNK_SIZE, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + CHUNK_SIZE // 2, end)
                if boundary > start:
                    end = boundary
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "source": source,
                        "location": location,
                        "text": chunk_text,
                    }
                )
            if end >= len(normalized):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def _load_index():
    if not INDEX_PATH.exists():
        return {"books": [], "chunks": []}
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as file:
            index = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取本地书籍索引失败：{error}") from error
    if (
        not isinstance(index, dict)
        or not isinstance(index.get("books"), list)
        or not isinstance(index.get("chunks"), list)
    ):
        raise ValueError(f"本地书籍索引格式错误：{INDEX_PATH}")
    return index


def _save_index(index):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=str(INDEX_PATH.parent),
        prefix=f"{INDEX_PATH.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(index, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, INDEX_PATH)
    except Exception as error:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"保存书籍索引失败：{error}；清理临时文件也失败：{cleanup_error}"
                ) from error
        if isinstance(error, OSError):
            raise RuntimeError(f"保存书籍索引失败：{error}") from error
        raise


def _file_id(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_book(file_path):
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到书籍文件：{path}")

    book_id = _file_id(path)
    index = _load_index()
    existing = next(
        (book for book in index["books"] if book.get("id") == book_id),
        None,
    )
    if existing:
        return {
            "title": existing["title"],
            "chunks": existing["chunks"],
            "already_indexed": True,
        }

    chunks = _split_sections(path.name, _read_book(path))
    if not chunks:
        raise ValueError(f"{path.name} 没有可索引的文本")

    index["books"].append(
        {
            "id": book_id,
            "title": path.name,
            "path": str(path),
            "chunks": len(chunks),
        }
    )
    index["chunks"].extend(
        {"book_id": book_id, **chunk} for chunk in chunks
    )
    _save_index(index)
    return {
        "title": path.name,
        "chunks": len(chunks),
        "already_indexed": False,
    }


def list_books():
    return _load_index()["books"]


@lru_cache(maxsize=8192)
def _terms(text):
    terms = []
    for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) == 1:
                terms.append(token)
            else:
                terms.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            terms.append(token)
    return terms


def search(query, top_k=4):
    if top_k <= 0:
        return []
    index = _load_index()
    chunks = index["chunks"]
    query_terms = set(_terms(query))
    if not chunks or not query_terms:
        return []

    document_terms = [_terms(chunk["text"]) for chunk in chunks]
    document_frequencies = Counter(
        term
        for terms in document_terms
        for term in set(terms)
    )
    average_length = sum(map(len, document_terms)) / len(document_terms)
    scored_chunks = []

    for chunk, terms in zip(chunks, document_terms):
        term_frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_frequency = math.log(
                1 + (len(chunks) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(terms) / max(average_length, 1)
            )
            score += inverse_frequency * frequency * 2.5 / denominator
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda result: result[0], reverse=True)
    return [dict(chunk, score=score) for score, chunk in scored_chunks[:top_k]]


def format_context(results):
    return "\n\n".join(
        f"【来源：{result['source']}，{result['location']}】\n{result['text']}"
        for result in results
    )
