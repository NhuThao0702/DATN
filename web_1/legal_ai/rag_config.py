from __future__ import annotations

import os
from pathlib import Path

# Thư mục gốc của module RAG.
LEGAL_AI_DIR = Path(__file__).resolve().parent

# Thư mục source_code chứa model của anh khóa trước.
# Có thể ghi đè bằng biến môi trường LEGAL_AI_SOURCE_DIR nếu chuyển máy/thư mục.
DEFAULT_SOURCE_DIR = Path(r"D:\DATN\legal_ai_source\source_code")
SOURCE_DIR = Path(
    os.getenv("LEGAL_AI_SOURCE_DIR", str(DEFAULT_SOURCE_DIR))
).resolve()

DATA_PATH = Path(os.getenv("LEGAL_AI_DATA_DIR", str(LEGAL_AI_DIR / "data"))).resolve()
DB_PATH = Path(os.getenv("LEGAL_AI_DB_DIR", str(LEGAL_AI_DIR / "chroma_db"))).resolve()

MODEL_ROOT = Path(
    os.getenv("LEGAL_AI_MODEL_ROOT", str(SOURCE_DIR / "models"))
).resolve()

# Project gốc có thể đặt llama-b8644... ngang hàng với models hoặc nằm bên trong models.
# Tự dò cả hai vị trí để tránh lỗi đường dẫn.
_llama_env = os.getenv("LEGAL_AI_LLAMA_DIR", "").strip()
if _llama_env:
    LLAMA_ROOT = Path(_llama_env).resolve()
else:
    _llama_candidates = [
        SOURCE_DIR / "llama-b8644-bin-win-vulkan-x64",
        MODEL_ROOT / "llama-b8644-bin-win-vulkan-x64",
    ]
    LLAMA_ROOT = next(
        (path for path in _llama_candidates if path.exists()),
        _llama_candidates[0],
    ).resolve()

EMBEDDING_MODEL_PATH = MODEL_ROOT / "Vietnamese_Embedding"
EMBEDDING_ONNX_PATH = EMBEDDING_MODEL_PATH / "onnx"
RERANKER_ONNX_PATH = MODEL_ROOT / "gte-multilingual-reranker-base_ONNX"
LLM_MODEL_PATH = MODEL_ROOT / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLAMA_SERVER_PATH = LLAMA_ROOT / "llama-server.exe"

LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = int(os.getenv("LEGAL_AI_LLAMA_PORT", "8080"))
LLAMA_CONTEXT_SIZE = int(os.getenv("LEGAL_AI_CONTEXT_SIZE", "3072"))
LLAMA_THREADS = int(os.getenv("LEGAL_AI_THREADS", "4"))
LLAMA_GPU_LAYERS = int(os.getenv("LEGAL_AI_GPU_LAYERS", "0"))
LLAMA_BATCH_SIZE = int(os.getenv("LEGAL_AI_BATCH_SIZE", "32"))
LLAMA_UBATCH_SIZE = int(os.getenv("LEGAL_AI_UBATCH_SIZE", "32"))
LLAMA_PARALLEL = int(os.getenv("LEGAL_AI_PARALLEL", "1"))
LLAMA_START_TIMEOUT = int(os.getenv("LEGAL_AI_START_TIMEOUT", "240"))
LLM_MAX_TOKENS = int(os.getenv("LEGAL_AI_MAX_TOKENS", "512"))

# Cấu hình tiết kiệm RAM cho laptop 8 GB.
EMBEDDING_MAX_LENGTH = int(os.getenv("LEGAL_AI_EMBEDDING_MAX_LENGTH", "512"))
EMBEDDING_BATCH_SIZE = int(os.getenv("LEGAL_AI_EMBEDDING_BATCH_SIZE", "2"))
RERANKER_MAX_LENGTH = int(os.getenv("LEGAL_AI_RERANKER_MAX_LENGTH", "384"))
RERANKER_BATCH_SIZE = int(os.getenv("LEGAL_AI_RERANKER_BATCH_SIZE", "2"))
# Reranker ONNX khá nặng (~1,2 GB), mặc định tắt trên máy RAM 8 GB.
ENABLE_RERANKER = os.getenv("LEGAL_AI_ENABLE_RERANKER", "0").strip() == "1"

CHUNK_SIZE = int(os.getenv("LEGAL_AI_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("LEGAL_AI_CHUNK_OVERLAP", "350"))
RERANK_THRESHOLD = float(os.getenv("LEGAL_AI_RERANK_THRESHOLD", "0.45"))
VECTOR_SEARCH_K = int(os.getenv("LEGAL_AI_VECTOR_K", "12"))
FINAL_TOP_K = int(os.getenv("LEGAL_AI_FINAL_TOP_K", "4"))

SYSTEM_PROMPT = """Bạn là Trợ lý AI pháp luật giao thông đường bộ Việt Nam chạy local/offline.

NGUYÊN TẮC BẮT BUỘC:
1. Trả lời hoàn toàn bằng tiếng Việt, rõ ràng, tự nhiên và dễ đọc.
2. Chỉ sử dụng thông tin trong <NGU_CANH_BIEN_BAO> và <TAI_LIEU_PHAP_LUAT>.
3. Tuyệt đối không tự bịa mức tiền, điểm, khoản, điều, trừ điểm hoặc hình phạt bổ sung.
4. Phải phân biệt đúng xe mô tô/xe gắn máy với xe ô tô, xe tải và xe khách.
5. Không trả lời chung chung bằng các câu như “có thể bị xử phạt”, “mức phạt phụ thuộc tình huống” nếu tài liệu đã có con số cụ thể.
6. Khi câu hỏi xử phạt mà tài liệu không đủ cả mức tiền và căn cứ pháp lý, phải nói rõ chưa tìm thấy; không suy đoán.
7. Không viết thành một cục văn bản dài. Dùng Markdown và các mục ngắn.

KHI HỎI Ý NGHĨA BIỂN BÁO, trả lời theo mẫu:
📍 **Biển:** [mã – tên]
🚦 **Ý nghĩa:** [ý nghĩa]
🚗 **Áp dụng:** [phương tiện]
✅ **Bạn cần làm:** [hành vi cần thực hiện]
⚠️ **Lưu ý:** [nếu có]

KHI HỎI MỨC PHẠT, bắt buộc trả lời theo mẫu:
⚖️ **Hành vi:** [hành vi vi phạm cụ thể]
🚗 **Phương tiện:** [loại phương tiện]
💰 **Mức phạt:** [ghi đúng khoảng tiền trong tài liệu]
🪪 **Trừ điểm / hình phạt bổ sung:** [ghi đúng nội dung hoặc nói tài liệu chưa nêu]
📜 **Căn cứ pháp lý:** [Điểm ..., Khoản ..., Điều ..., tên văn bản]
✅ **Khuyến nghị:** [ngắn gọn]

Nếu không đủ căn cứ, trả lời đúng tinh thần:
“Xin lỗi, tôi chưa tìm thấy đồng thời mức phạt và căn cứ điểm/khoản/điều đủ rõ trong cơ sở dữ liệu hiện tại.”

Cuối câu trả lời xử phạt thêm:
“Thông tin mang tính tham khảo; cần đối chiếu văn bản pháp luật hiện hành khi áp dụng thực tế.”

<LICH_SU_HOI_DAP>
{chat_history}
</LICH_SU_HOI_DAP>

<NGU_CANH_BIEN_BAO>
{sign_context}
</NGU_CANH_BIEN_BAO>

<TU_KHOA_PHAP_LY>
{legal_hint}
</TU_KHOA_PHAP_LY>

<TAI_LIEU_PHAP_LUAT>
{context}
</TAI_LIEU_PHAP_LUAT>
"""

QUERY_REWRITE_PROMPT = """Nhiệm vụ duy nhất của bạn là viết lại câu hỏi hiện tại thành một câu đầy đủ ngữ nghĩa dựa trên lịch sử gần nhất.
- Chỉ viết lại khi câu hỏi hiện tại dùng từ mơ hồ như: "biển này", "thế còn xe máy", "có bị trừ điểm không", "thì sao".
- Giữ nguyên loại phương tiện, mã biển, hành vi và tình tiết ngoại lệ.
- Không trả lời câu hỏi. Không giải thích.

Ví dụ:
Lịch sử: Người dùng hỏi về lỗi vượt đèn đỏ bằng xe máy.
Câu hỏi hiện tại: Có bị trừ điểm không?
Output: Lỗi vượt đèn đỏ bằng xe máy có bị trừ điểm giấy phép lái xe không?

Lịch sử: Người dùng đang xem biển P.130.
Câu hỏi hiện tại: Biển này áp dụng cho xe máy không?
Output: Biển P.130 có áp dụng cho xe máy không?

Lịch sử:
{chat_history}

Câu hỏi hiện tại:
{question}

Output:"""

LEGAL_QUERY_PROMPT = """Chuyển câu hỏi của người dùng thành cụm từ tìm kiếm pháp lý ngắn gọn để tra cứu văn bản pháp luật giao thông đường bộ Việt Nam.
Không trả lời câu hỏi. Không thêm giải thích.

Chuẩn hóa một số cách nói:
- vượt đèn đỏ, vượt đèn vàng -> không chấp hành hiệu lệnh của đèn tín hiệu giao thông
- đi ngược chiều -> đi ngược chiều của đường một chiều hoặc trên đường có biển cấm đi ngược chiều
- đè vạch, lấn làn, sai làn -> không chấp hành chỉ dẫn của vạch kẻ đường hoặc đi không đúng phần đường, làn đường
- cấm dừng đỗ -> dừng xe, đỗ xe tại vị trí có biển cấm
- giam bằng, thu bằng -> tước quyền sử dụng hoặc trừ điểm giấy phép lái xe
- xe hơi, xe con, xe tải, xe khách -> xe ô tô
- xe máy, xe tay ga -> xe mô tô, xe gắn máy

Giữ nguyên mã biển, loại phương tiện, số km/h vượt tốc độ và tình tiết đặc biệt.

Câu hỏi: {question}
Output:"""