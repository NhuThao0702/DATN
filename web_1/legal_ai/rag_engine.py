from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnxruntime as ort
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .rag_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    DB_PATH,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_ONNX_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    ENABLE_RERANKER,
    FINAL_TOP_K,
    LLAMA_BATCH_SIZE,
    LLAMA_CONTEXT_SIZE,
    LLAMA_GPU_LAYERS,
    LLAMA_PARALLEL,
    LLAMA_SERVER_HOST,
    LLAMA_SERVER_PATH,
    LLAMA_SERVER_PORT,
    LLAMA_START_TIMEOUT,
    LLAMA_THREADS,
    LLAMA_UBATCH_SIZE,
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    QUERY_REWRITE_PROMPT,
    RERANKER_BATCH_SIZE,
    RERANKER_MAX_LENGTH,
    RERANKER_ONNX_PATH,
    RERANK_THRESHOLD,
    SYSTEM_PROMPT,
    VECTOR_SEARCH_K,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@dataclass
class AIState:
    vector_store: Chroma | None = None
    reranker: "ONNXReranker | None" = None
    llm: ChatOpenAI | None = None
    answer_chain: Any = None
    server_process: subprocess.Popen | None = None
    owns_server_process: bool = False
    is_initializing: bool = False
    is_ready: bool = False
    init_error: str | None = None
    stage: str = "Chưa khởi động"
    lock: threading.RLock = field(default_factory=threading.RLock)
    generation_lock: threading.Lock = field(default_factory=threading.Lock)


ai_state = AIState()


class ONNXEmbedding(Embeddings):
    """Embedding tương thích LangChain, chạy bằng ONNX Runtime trên CPU."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        max_length: int = EMBEDDING_MAX_LENGTH,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        model_file = model_path / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError(f"Không tìm thấy embedding ONNX: {model_file}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.max_length = max_length
        self.batch_size = batch_size

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in self.input_names
        }
        outputs = self.session.run(None, feed)
        hidden = outputs[0]
        attention = encoded["attention_mask"]
        expanded = attention[:, :, np.newaxis].astype(np.float32)
        pooled = (hidden * expanded).sum(axis=1) / expanded.sum(axis=1).clip(min=1e-9)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return pooled.astype(np.float32)

    def _encode_all(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        blocks = []
        for index in range(0, len(texts), self.batch_size):
            blocks.append(self._encode_batch(texts[index : index + self.batch_size]))
        return np.vstack(blocks).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode_all(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode_all([text])[0]


class ONNXReranker:
    """Cross-encoder reranker chạy thuần ONNX Runtime."""

    def __init__(self, model_path: Path) -> None:
        model_file = model_path / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError(f"Không tìm thấy reranker ONNX: {model_file}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def score(
        self,
        pairs: Sequence[Sequence[str]],
        batch_size: int = RERANKER_BATCH_SIZE,
    ) -> list[float]:
        scores: list[float] = []
        for index in range(0, len(pairs), batch_size):
            batch = pairs[index : index + batch_size]
            queries = [item[0] for item in batch]
            passages = [item[1] for item in batch]
            encoded = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=RERANKER_MAX_LENGTH,
                return_tensors="np",
            )
            feed = {
                key: value.astype(np.int64)
                for key, value in encoded.items()
                if key in self.input_names
            }
            if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"], dtype=np.int64)
            logits = self.session.run(None, feed)[0].reshape(-1).astype(np.float32)
            scores.extend((1.0 / (1.0 + np.exp(-logits))).tolist())
        return scores


def _server_url(path: str = "/v1/models") -> str:
    return f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}{path}"


def _http_ready(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(_server_url(), timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def _validate_required_files() -> None:
    required = [
        EMBEDDING_ONNX_PATH / "model.onnx",
        EMBEDDING_MODEL_PATH / "tokenizer.json",
        LLM_MODEL_PATH,
        LLAMA_SERVER_PATH,
    ]
    if ENABLE_RERANKER:
        required.extend([
            RERANKER_ONNX_PATH / "model.onnx",
            RERANKER_ONNX_PATH / "tokenizer.json",
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n- ".join(missing)
        raise FileNotFoundError(
            "Thiếu file mô hình RAG:\n- " + joined +
            "\nHãy sao chép thư mục models và llama-b8644-bin-win-vulkan-x64 "
            "vào legal_ai hoặc cấu hình LEGAL_AI_SOURCE_DIR."
        )


def start_llama_server() -> None:
    if _http_ready():
        ai_state.stage = f"Đã kết nối llama-server đang chạy ở cổng {LLAMA_SERVER_PORT}"
        ai_state.owns_server_process = False
        return

    log_path = Path(__file__).resolve().parent / "llama_server.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    command = [
        str(LLAMA_SERVER_PATH),
        "-m",
        str(LLM_MODEL_PATH),
        "-c",
        str(LLAMA_CONTEXT_SIZE),
        "-b",
        str(LLAMA_BATCH_SIZE),
        "--ubatch-size",
        str(LLAMA_UBATCH_SIZE),
        "--parallel",
        str(LLAMA_PARALLEL),
        "-t",
        str(LLAMA_THREADS),
        "--n-gpu-layers",
        str(LLAMA_GPU_LAYERS),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
    ]

    creationflags = 0x08000000 if os.name == "nt" else 0
    ai_state.server_process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    ai_state.owns_server_process = True
    ai_state.stage = "Đang tải Qwen vào RAM bằng CPU"

    deadline = time.time() + LLAMA_START_TIMEOUT
    while time.time() < deadline:
        process = ai_state.server_process
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"llama-server đã thoát. Xem log tại: {log_path}"
            )
        if _http_ready():
            return
        time.sleep(1)

    raise TimeoutError(
        f"Quá thời gian chờ llama-server. Xem log tại: {log_path}"
    )


def shutdown_ai_system() -> None:
    with ai_state.lock:
        process = ai_state.server_process
        if ai_state.owns_server_process and process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=8)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        ai_state.server_process = None
        ai_state.owns_server_process = False
        ai_state.is_ready = False
        if ai_state.stage != "Lỗi khởi tạo":
            ai_state.stage = "Đã dừng"


atexit.register(shutdown_ai_system)


def _iter_data_files() -> Iterable[Path]:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    extensions = {".pdf", ".txt", ".md", ".json"}
    return sorted(
        path for path in DATA_PATH.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _data_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in _iter_data_files():
        digest.update(str(path.relative_to(DATA_PATH)).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _load_documents() -> list[Any]:
    documents: list[Any] = []
    for path in _iter_data_files():
        try:
            if path.suffix.lower() == ".pdf":
                loaded = PyMuPDFLoader(str(path)).load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            for document in loaded:
                text = re.sub(
                    r"(?i)\d*\s*CÔNG BÁO/Số.*?\d{4}\s*\d*",
                    "",
                    document.page_content,
                )
                document.page_content = re.sub(r"[ \t]+", " ", text).strip()
                document.metadata["source"] = str(path)
            documents.extend(item for item in loaded if item.page_content.strip())
        except Exception as exc:
            print(f"[RAG] Bỏ qua tài liệu lỗi {path.name}: {exc}")
    return documents


def _split_documents(documents: list[Any]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nĐiều ", "\nChương ", "\nPhần ", "\n\n", "\n", ". ", " "],
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _build_embedding() -> ONNXEmbedding:
    return ONNXEmbedding(
        EMBEDDING_ONNX_PATH,
        tokenizer_path=EMBEDDING_MODEL_PATH,
        max_length=EMBEDDING_MAX_LENGTH,
        batch_size=EMBEDDING_BATCH_SIZE,
    )


def _load_or_build_vector_store() -> Chroma:
    embedding = _build_embedding()
    fingerprint = _data_fingerprint()
    fingerprint_file = DB_PATH / "data_fingerprint.txt"
    cached_fingerprint = ""
    if fingerprint_file.exists():
        cached_fingerprint = fingerprint_file.read_text(encoding="utf-8").strip()

    db_files_exist = DB_PATH.exists() and any(DB_PATH.iterdir())
    if db_files_exist and cached_fingerprint == fingerprint:
        ai_state.stage = "Đang mở cơ sở dữ liệu pháp luật"
        return Chroma(
            collection_name="traffic_law",
            persist_directory=str(DB_PATH),
            embedding_function=embedding,
            collection_metadata={"hnsw:space": "ip"},
        )

    if DB_PATH.exists():
        import shutil
        shutil.rmtree(DB_PATH, ignore_errors=True)
    DB_PATH.mkdir(parents=True, exist_ok=True)

    ai_state.stage = "Đang đọc và chia nhỏ văn bản pháp luật"
    documents = _load_documents()
    if not documents:
        raise RuntimeError(f"Không có tài liệu PDF/TXT trong {DATA_PATH}")
    chunks = _split_documents(documents)
    if not chunks:
        raise RuntimeError("Không tạo được đoạn văn bản để xây ChromaDB")

    ai_state.stage = f"Đang tạo ChromaDB từ {len(chunks)} đoạn"
    store = Chroma(
        collection_name="traffic_law",
        persist_directory=str(DB_PATH),
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "ip"},
    )
    for index in range(0, len(chunks), 50):
        store.add_documents(chunks[index : index + 50])
    fingerprint_file.write_text(fingerprint, encoding="utf-8")
    return store


def _clean_llm_text(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", value or "", flags=re.DOTALL)
    value = value.replace("Output:", "").strip().strip('"')
    return value


def _format_history(messages: Sequence[dict[str, str]], max_turns: int = 3) -> str:
    if not messages:
        return "(Chưa có lịch sử trò chuyện)"
    selected = list(messages)[-(max_turns * 2):]
    lines = []
    for message in selected:
        role = message.get("role", "user")
        text = str(message.get("text") or message.get("content") or "").strip()
        if not text:
            continue
        label = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{label}: {text[:1200]}")
    return "\n".join(lines) or "(Chưa có lịch sử trò chuyện)"


def _needs_rewrite(question: str) -> bool:
    normalized = question.lower().strip()
    patterns = [
        "biển này",
        "biển đó",
        "thế còn",
        "vậy còn",
        "thì sao",
        "có bị",
        "như vậy",
    ]
    return len(question.split()) <= 12 and any(item in normalized for item in patterns)


def _rewrite_query(question: str, messages: Sequence[dict[str, str]]) -> str:
    if not _needs_rewrite(question) or not messages or ai_state.llm is None:
        return question
    prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
    chain = prompt | ai_state.llm | StrOutputParser()
    try:
        result = chain.invoke(
            {"chat_history": _format_history(messages, max_turns=2), "question": question}
        )
        return _clean_llm_text(result).splitlines()[0] or question
    except Exception:
        return question


def _legal_query(question: str) -> str:
    """Chuẩn hóa từ khóa bằng Python để không phải gọi LLM thêm một lần."""
    value = question.strip()
    replacements = [
        (r"\b(vượt|chạy) đèn đỏ\b", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"),
        (r"\bvượt đèn vàng\b", "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"),
        (r"\bđi ngược chiều\b", "đi ngược chiều của đường một chiều hoặc đường có biển cấm đi ngược chiều"),
        (r"\b(đè vạch|lấn làn|sai làn)\b", "không chấp hành vạch kẻ đường hoặc đi không đúng phần đường làn đường"),
        (r"\b(cấm dừng đỗ|cấm dừng và đỗ)\b", "dừng xe đỗ xe tại vị trí có biển cấm"),
        (r"\b(giam bằng|thu bằng)\b", "tước quyền sử dụng hoặc trừ điểm giấy phép lái xe"),
        (r"\b(xe hơi|xe con|xe tải|xe khách)\b", "xe ô tô"),
        (r"\b(xe máy|xe tay ga)\b", "xe mô tô xe gắn máy"),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def _deduplicate_documents(documents: Sequence[Any]) -> list[Any]:
    output = []
    seen: set[str] = set()
    for document in documents:
        key = hashlib.sha1(document.page_content.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        output.append(document)
    return output



def _compact_sign_context_for_search(sign_context: str) -> str:
    """Rút gọn ngữ cảnh biển báo để đưa vào câu truy vấn vector."""
    if not sign_context or not sign_context.strip():
        return ""

    wanted_prefixes = (
        "Mã biển:",
        "Tên biển:",
        "Ý nghĩa:",
        "Phương tiện đang chọn:",
        "Phương tiện áp dụng:",
        "Hành vi bị cấm hoặc cần chú ý:",
    )
    lines = []
    for raw_line in sign_context.splitlines():
        line = raw_line.strip()
        if line.startswith(wanted_prefixes):
            lines.append(line)

    return " | ".join(lines)[:1400]


def _is_penalty_question(question: str) -> bool:
    normalized = question.lower()
    keywords = (
        "phạt",
        "mức phạt",
        "xử phạt",
        "vi phạm",
        "bao nhiêu tiền",
        "trừ điểm",
        "tước bằng",
        "tước giấy phép",
        "điều ",
        "khoản ",
        "nghị định",
    )
    return any(keyword in normalized for keyword in keywords)


def _prefer_legal_sources(documents: Sequence[Any], question: str) -> list[Any]:
    """
    Khi người dùng hỏi xử phạt, ưu tiên các đoạn từ Nghị định 168
    nếu chúng đã xuất hiện trong tập kết quả tìm kiếm.
    """
    if not _is_penalty_question(question):
        return list(documents)

    indexed = list(enumerate(documents))

    def source_rank(item: tuple[int, Any]) -> tuple[int, int]:
        index, document = item
        source_name = Path(
            str(document.metadata.get("source", ""))
        ).name.lower()
        priority = 0 if "168" in source_name else 1
        return priority, index

    return [document for _, document in sorted(indexed, key=source_rank)]


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value or ""))
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _build_search_queries(question: str, rewritten: str, legal_hint: str, sign_context: str) -> list[str]:
    queries = [rewritten, legal_hint]
    if _is_penalty_question(question):
        sign_hint = _compact_sign_context_for_search(sign_context)
        queries.append(
            "Nghị định 168/2024/NĐ-CP phạt tiền trừ điểm giấy phép lái xe "
            f"điểm khoản điều {question} {sign_hint}"
        )
    output = []
    seen = set()
    for query in queries:
        query = str(query or "").strip()
        key = _normalize_search_text(query)
        if query and key not in seen:
            seen.add(key)
            output.append(query)
    return output


def _lexical_document_score(document: Any, query: str, penalty_question: bool) -> float:
    text = _normalize_search_text(document.page_content)
    query_norm = _normalize_search_text(query)
    stopwords = {
        "la", "va", "cua", "cho", "trong", "nay", "do", "bi", "co", "the",
        "nhu", "nao", "bao", "nhieu", "mot", "cac", "khi", "neu", "ve"
    }
    tokens = {
        token for token in re.findall(r"[a-z0-9\.]+", query_norm)
        if len(token) >= 3 and token not in stopwords
    }
    score = sum(1.0 for token in tokens if token in text)

    source_name = _normalize_search_text(Path(str(document.metadata.get("source", ""))).name)
    if penalty_question and "168" in source_name:
        score += 8.0
    if penalty_question and "phat tien" in text:
        score += 4.0
    if penalty_question and any(term in text for term in ["dieu ", "khoan ", "diem "]):
        score += 3.0
    if any(term in query_norm for term in ["o to", "xe tai", "xe khach"]):
        if any(term in text for term in ["xe o to", "o to"]):
            score += 3.0
    if any(term in query_norm for term in ["xe may", "xe mo to", "xe gan may"]):
        if any(term in text for term in ["xe mo to", "xe gan may"]):
            score += 3.0
    return score


def _rank_without_reranker(documents: Sequence[Any], query: str) -> list[Any]:
    penalty_question = _is_penalty_question(query)
    ranked = sorted(
        enumerate(documents),
        key=lambda item: (
            -_lexical_document_score(item[1], query, penalty_question),
            item[0],
        ),
    )
    return [document for _, document in ranked]


def retrieve_context(
    question: str,
    messages: Sequence[dict[str, str]] | None = None,
    sign_context: str = "",
    top_k: int = FINAL_TOP_K,
) -> tuple[str, str, list[dict[str, Any]], str]:
    if ai_state.vector_store is None:
        return "", question, [], question

    messages = messages or []

    # Với câu hỏi kiểu "biển này", dùng trực tiếp biển đang chọn thay vì
    # gọi LLM thêm một lần để viết lại câu hỏi. Cách này nhanh hơn đáng kể
    # trên máy RAM 8 GB và giúp tìm đúng mã/tên biển.
    sign_hint = _compact_sign_context_for_search(sign_context)
    if sign_hint:
        rewritten = f"{question}\nNgữ cảnh biển báo đang chọn: {sign_hint}"
    else:
        rewritten = _rewrite_query(question, messages)

    legal_hint = _legal_query(rewritten)

    search_queries = _build_search_queries(
        question=question,
        rewritten=rewritten,
        legal_hint=legal_hint,
        sign_context=sign_context,
    )
    collected_docs = []
    for search_query in search_queries:
        collected_docs.extend(
            ai_state.vector_store.similarity_search(search_query, k=VECTOR_SEARCH_K)
        )

    candidates = _deduplicate_documents(collected_docs)
    candidates = _prefer_legal_sources(candidates, rewritten)
    candidates = _rank_without_reranker(candidates, rewritten)
    if not candidates:
        return "", legal_hint, [], rewritten

    if ai_state.reranker is not None:
        use_two_queries = bool(legal_hint and legal_hint != rewritten)
        if use_two_queries:
            size = len(candidates)
            pairs = (
                [[rewritten, item.page_content] for item in candidates]
                + [[legal_hint, item.page_content] for item in candidates]
            )
            all_scores = ai_state.reranker.score(pairs)
            scores = [max(a, b) for a, b in zip(all_scores[:size], all_scores[size:])]
        else:
            scores = ai_state.reranker.score(
                [[rewritten, item.page_content] for item in candidates]
            )
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        filtered = [item for item in ranked if item[1] >= RERANK_THRESHOLD]
        selected_pairs = (filtered or ranked)[:top_k]
    else:
        selected_pairs = [(item, None) for item in candidates[:top_k]]

    selected_docs = [item[0] for item in selected_pairs]
    context = "\n\n".join(
        f"[Nguồn: {Path(str(document.metadata.get('source', 'Không rõ'))).name}"
        f" - Trang {int(document.metadata.get('page', 0)) + 1}]\n{document.page_content}"
        for document in selected_docs
    )
    sources = []
    for document, score in selected_pairs:
        sources.append(
            {
                "file": Path(str(document.metadata.get("source", "Không rõ"))).name,
                "page": int(document.metadata.get("page", 0)) + 1,
                "score": round(float(score), 4) if score is not None else None,
            }
        )
    return context, legal_hint, sources, rewritten


def initialize_ai_system() -> None:
    with ai_state.lock:
        if ai_state.is_ready or ai_state.is_initializing:
            return
        ai_state.is_initializing = True
        ai_state.init_error = None
        ai_state.stage = "Đang kiểm tra mô hình"

    try:
        _validate_required_files()
        ai_state.vector_store = _load_or_build_vector_store()
        if ENABLE_RERANKER:
            ai_state.stage = "Đang tải bộ xếp hạng văn bản"
            ai_state.reranker = ONNXReranker(RERANKER_ONNX_PATH)
        else:
            ai_state.stage = "Đã bỏ qua reranker để tiết kiệm RAM"
            ai_state.reranker = None
        start_llama_server()

        ai_state.stage = "Đang kết nối Qwen với LangChain"
        no_think_params = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "min_p": 0.0,
        }
        ai_state.llm = ChatOpenAI(
            base_url=f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
            api_key="sk-local-no-key",
            model="qwen",
            temperature=0.2,
            top_p=0.8,
            max_tokens=LLM_MAX_TOKENS,
            timeout=240,
            extra_body=no_think_params,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        ai_state.answer_chain = prompt | ai_state.llm | StrOutputParser()
        ai_state.is_ready = True
        ai_state.stage = "Sẵn sàng"
    except Exception as exc:
        ai_state.init_error = str(exc)
        ai_state.stage = "Lỗi khởi tạo"
        print(f"[RAG] Khởi tạo thất bại: {exc}")
    finally:
        ai_state.is_initializing = False


def initialize_ai_background() -> bool:
    with ai_state.lock:
        if ai_state.is_ready or ai_state.is_initializing:
            return False
        thread = threading.Thread(
            target=initialize_ai_system,
            daemon=True,
            name="traffic-legal-rag-init",
        )
        thread.start()
        return True


def get_status() -> dict[str, Any]:
    return {
        "available": ai_state.init_error is None or ai_state.is_ready or ai_state.is_initializing,
        "ready": ai_state.is_ready,
        "initializing": ai_state.is_initializing,
        "stage": ai_state.stage,
        "error": ai_state.init_error,
        "llama_port": LLAMA_SERVER_PORT,
        "data_path": str(DATA_PATH),
        "db_path": str(DB_PATH),
    }


def answer_question(
    question: str,
    messages: Sequence[dict[str, str]] | None = None,
    sign_context: str = "",
) -> dict[str, Any]:
    question = str(question or "").strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống")
    if not ai_state.is_ready or ai_state.answer_chain is None:
        raise RuntimeError(ai_state.init_error or "Hệ thống RAG chưa sẵn sàng")

    messages = messages or []
    with ai_state.generation_lock:
        context, legal_hint, sources, rewritten = retrieve_context(
            question,
            messages=messages,
            sign_context=sign_context,
            top_k=FINAL_TOP_K,
        )

        if not context and not sign_context.strip():
            return {
                "answer": "Xin lỗi, tôi không tìm thấy quy định cụ thể trong cơ sở dữ liệu hiện tại.",
                "found_context": False,
                "sources": [],
                "rewritten_query": rewritten,
                "legal_hint": legal_hint,
            }

        response = ai_state.answer_chain.invoke(
            {
                "question": question,
                "context": context or "(Không có đoạn pháp luật liên quan)",
                "sign_context": sign_context or "(Không có biển báo đang được chọn)",
                "legal_hint": legal_hint,
                "chat_history": _format_history(messages),
            }
        )
        answer = _clean_llm_text(response)

        # Với câu hỏi xử phạt, không để model kết luận chung chung như
        # “có thể bị phạt” nếu chưa trích được số tiền và căn cứ cụ thể.
        if _is_penalty_question(question):
            has_amount = bool(
                re.search(r"\d[\d\.\s]*(?:đồng|triệu)", answer, flags=re.IGNORECASE)
            )
            has_basis = bool(
                re.search(r"\b(?:Điều|Khoản|Điểm)\s+\w+", answer, flags=re.IGNORECASE)
            )
            if not (has_amount and has_basis):
                answer = (
                    "🔎 **Kết quả tra cứu:** Hệ thống chưa trích được đồng thời mức tiền "
                    "và căn cứ điểm/khoản/điều đủ rõ từ các đoạn tài liệu vừa tìm thấy.\n\n"
                    "👉 Hãy nêu rõ loại phương tiện và hành vi cụ thể; hệ thống sẽ không "
                    "tự suy đoán mức phạt khi nguồn chưa đủ căn cứ."
                )

        if sign_context.strip():
            sources = [
                {"file": "Ngữ cảnh biển báo đang chọn", "page": None, "score": None},
                *sources,
            ]
        return {
            "answer": answer,
            "found_context": bool(context or sign_context.strip()),
            "sources": sources,
            "rewritten_query": rewritten,
            "legal_hint": legal_hint,
        }