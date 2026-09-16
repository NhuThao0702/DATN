from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory, g
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import uuid
import re
import unicodedata
import time
from pathlib import Path

# Lõi chatbot RAG pháp luật giao thông chạy local/offline.
try:
    from legal_ai import (
        answer_question as rag_answer_question,
        get_status as get_rag_status,
        initialize_ai_background,
        shutdown_ai_system,
    )
    RAG_MODULE_AVAILABLE = True
    RAG_IMPORT_ERROR = ""
except Exception as rag_import_exception:
    RAG_MODULE_AVAILABLE = False
    RAG_IMPORT_ERROR = str(rag_import_exception)


# Bộ tra cứu mức phạt có cấu trúc.
# Nếu hai file legal_ai/penalty_engine.py và legal_ai/penalties.json chưa sẵn sàng,
# Web vẫn chạy và tự chuyển sang RAG/rule-based như trước.
try:
    from legal_ai.penalty_engine import lookup_penalty, format_penalty_answer
    PENALTY_ENGINE_AVAILABLE = True
    PENALTY_IMPORT_ERROR = ""
except Exception as penalty_import_exception:
    lookup_penalty = None
    format_penalty_answer = None
    PENALTY_ENGINE_AVAILABLE = False
    PENALTY_IMPORT_ERROR = str(penalty_import_exception)


# Trang toàn văn chính thức đã được dùng để tạo file dữ liệu RAG local.
NGHI_DINH_168_FULLTEXT_URL = (
    "https://xaydungchinhsach.chinhphu.vn/"
    "toan-van-nghi-dinh-168-2024-nd-cp-quy-dinh-xu-phat-"
    "vi-pham-hanh-chinh-ve-trat-tu-atgt-duong-bo-119241231164556785.htm"
)

# Quy chuẩn dùng để mô tả đúng tên biển, đối tượng và điều kiện áp dụng.
QCVN_41_2024_URL = (
    "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
)


# ==================================================
# MỨC PHẠT ĐÃ ĐỐI CHIẾU - FALLBACK CỤC BỘ
# ==================================================
# Các bản ghi dưới đây chỉ dùng khi penalties.json chưa điền đủ hoặc còn
# chuỗi mẫu kiểu "Điền mức phạt...". Mục tiêu là không để chatbot trả
# placeholder cho những lỗi demo quan trọng nhất.
VERIFIED_PENALTY_RULES = {
    "di_nguoc_chieu": {
        "aliases": [
            "di nguoc chieu", "cam di nguoc chieu", "duong mot chieu",
            "p.102", "p102",
        ],
        "sign_codes": ["P.102"],
        "behavior": "Đi ngược chiều của đường một chiều hoặc đường có biển Cấm đi ngược chiều.",
        "by_vehicle": {
            "oto": {
                "fine": "18.000.000 - 20.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm d khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_tai": {
                "fine": "18.000.000 - 20.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm d khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_khach": {
                "fine": "18.000.000 - 20.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm d khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_may": {
                "fine": "4.000.000 - 6.000.000 đồng",
                "points": "Trừ 2 điểm GPLX",
                "legal_basis": "Điểm a khoản 7 và điểm a khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP",
            },
        },
    },
    "vuot_den_do": {
        "aliases": [
            "vuot den do", "den do", "khong chap hanh hieu lenh den",
            "khong chap hanh den tin hieu",
        ],
        "sign_codes": ["den_do"],
        "behavior": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
        "by_vehicle": {
            "oto": {
                "fine": "18.000.000 - 20.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm b khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_tai": {
                "fine": "18.000.000 - 20.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm b khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_khach": {
                "fine": "18.000.000 - 20.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm b khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_may": {
                "fine": "4.000.000 - 6.000.000 đồng",
                "points": "Trừ 4 điểm GPLX",
                "legal_basis": "Điểm c khoản 7 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP",
            },
        },
    },
    "cam_re_trai_phai": {
        "aliases": [
            "cam re trai", "cam re phai", "re trai noi cam", "re phai noi cam",
            "p.123a", "p123a", "p.123b", "p123b", "p.103b", "p103b", "p.103c", "p103c",
        ],
        "sign_codes": ["P.123a", "P.123b", "P.103b", "P.103c"],
        "behavior": "Rẽ trái hoặc rẽ phải tại nơi có biển báo cấm rẽ tương ứng đối với loại phương tiện đang điều khiển.",
        "by_vehicle": {
            "oto": {
                "fine": "2.000.000 - 3.000.000 đồng",
                "points": "Trừ 2 điểm GPLX",
                "legal_basis": "Điểm k khoản 4 và điểm a khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_tai": {
                "fine": "2.000.000 - 3.000.000 đồng",
                "points": "Trừ 2 điểm GPLX",
                "legal_basis": "Điểm k khoản 4 và điểm a khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_khach": {
                "fine": "2.000.000 - 3.000.000 đồng",
                "points": "Trừ 2 điểm GPLX",
                "legal_basis": "Điểm k khoản 4 và điểm a khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP",
            },
            "xe_may": {
                "fine": "600.000 - 800.000 đồng",
                "points": "Không thuộc nhóm hành vi bị trừ điểm được liệt kê tại khoản 13 Điều 7",
                "legal_basis": "Điểm a khoản 3 Điều 7 Nghị định 168/2024/NĐ-CP",
            },
        },
    },
}


PLACEHOLDER_MARKERS = (
    "dien muc phat", "dien so diem", "dien hinh phat", "diem ...",
    "khoan ...", "chua xac dinh", "chua co du lieu",
)


def _contains_placeholder(value):
    norm = normalize_vietnamese_text(value) if 'normalize_vietnamese_text' in globals() else str(value or '').lower()
    return any(marker in norm for marker in PLACEHOLDER_MARKERS)


def structured_penalty_record_is_usable(record, vehicle_type="all"):
    if not isinstance(record, dict):
        return False

    # Bản ghi gộp xuất hiện khi một câu hỏi pháp lý khớp nhiều phương tiện
    # hoặc khi người dùng tra ngược theo Điều/Khoản mà có nhiều hành vi.
    combined = record.get("_combined_records")
    if isinstance(combined, list) and combined:
        return all(
            structured_penalty_record_is_usable(item, vehicle_type="all")
            for item in combined
            if isinstance(item, dict)
        )

    fine = record.get("fine", "")
    basis = record.get("legal_basis", "")
    if isinstance(fine, dict):
        fine = fine.get(vehicle_type) or fine.get("all") or ""
    if _contains_placeholder(fine) or _contains_placeholder(basis):
        return False
    return bool(re.search(r"\d[\d.\s]*(?:đồng|dong|triệu|trieu)", str(fine), flags=re.IGNORECASE))


def lookup_verified_builtin_penalty(question, vehicle_type="all", sign_code=""):
    q_norm = normalize_vietnamese_text(question) if 'normalize_vietnamese_text' in globals() else str(question or '').lower()
    code = str(sign_code or '').strip()

    matched = None
    for rule_id, rule in VERIFIED_PENALTY_RULES.items():
        if code and code in rule.get("sign_codes", []):
            matched = (rule_id, rule)
            break
        if any(alias in q_norm for alias in rule.get("aliases", [])):
            matched = (rule_id, rule)
            break

    if not matched:
        return None

    rule_id, rule = matched
    by_vehicle = rule.get("by_vehicle", {})

    if vehicle_type in by_vehicle:
        data = by_vehicle[vehicle_type]
        return {
            "id": f"builtin_{rule_id}_{vehicle_type}",
            "behavior": rule["behavior"],
            "vehicle_types": [vehicle_type],
            "fine": data["fine"],
            "points": data["points"],
            "additional_penalty": "Không có hình phạt bổ sung riêng trong trường hợp thông thường nêu trên.",
            "legal_basis": data["legal_basis"],
            "source_title": "Nghị định 168/2024/NĐ-CP",
            "source_url_key": "nghi_dinh_168_fulltext",
        }

    # Nếu người dùng chưa chọn loại phương tiện, trả song song 2 nhóm phổ biến
    # thay vì bắt người dùng hỏi lại.
    auto = by_vehicle.get("oto")
    moto = by_vehicle.get("xe_may")
    if auto and moto:
        return {
            "id": f"builtin_{rule_id}_all",
            "behavior": rule["behavior"],
            "vehicle_types": ["oto", "xe_may"],
            "fine": f"Ô tô: {auto['fine']}; Xe máy: {moto['fine']}",
            "points": f"Ô tô: {auto['points']}; Xe máy: {moto['points']}",
            "additional_penalty": "Tình tiết gây tai nạn hoặc trường hợp đặc biệt có thể áp dụng mức xử lý khác.",
            "legal_basis": f"Ô tô: {auto['legal_basis']}; Xe máy: {moto['legal_basis']}",
            "source_title": "Nghị định 168/2024/NĐ-CP",
            "source_url_key": "nghi_dinh_168_fulltext",
        }
    return None


try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ==================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# Có thể giữ nguyên đường dẫn mặc định hoặc cấu hình bằng biến môi trường.
# Ví dụ trên Windows PowerShell:
#   $env:DATN_MODEL_PATH="D:\DATN\runs\detect\traffic_sign_yolov8s_150_v2\weights\best.pt"
#   $env:DATN_DATA_YAML_PATH="D:\DATN\data\data.yaml"
MODEL_PATH = Path(os.getenv(
    "DATN_MODEL_PATH",
    r"D:\DATN\runs\detect\traffic_sign_yolov8s_150_v2\weights\best.pt",
))
DATA_YAML_PATH = Path(os.getenv(
    "DATN_DATA_YAML_PATH",
    r"D:\DATN\data\data.yaml",
))
SKIP_MODEL_LOAD = os.getenv("DATN_SKIP_MODEL_LOAD", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==================================================
# FLASK + DATABASE ĐĂNG NHẬP
# ==================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "datn_traffic_sign_assistant_2026"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(BASE_DIR / "users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Vui lòng đăng nhập để sử dụng hệ thống."


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


# ==================================================
# XÁC THỰC DÀNH CHO MOBILE (BEARER TOKEN)
# ==================================================
MOBILE_TOKEN_MAX_AGE = 7 * 24 * 60 * 60
mobile_token_serializer = URLSafeTimedSerializer(
    app.config["SECRET_KEY"],
    salt="traffic-sign-mobile-auth-v1",
)


def create_mobile_token(user):
    return mobile_token_serializer.dumps({
        "user_id": int(user.id),
        "username": str(user.username),
    })


def get_mobile_user_from_request():
    authorization = str(request.headers.get("Authorization", "")).strip()
    if not authorization.lower().startswith("bearer "):
        return None, "Thiếu Bearer token."

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None, "Bearer token không hợp lệ."

    try:
        payload = mobile_token_serializer.loads(
            token,
            max_age=MOBILE_TOKEN_MAX_AGE,
        )
        user_id = int(payload.get("user_id"))
    except SignatureExpired:
        return None, "Phiên đăng nhập đã hết hạn."
    except (BadSignature, TypeError, ValueError, AttributeError):
        return None, "Token đăng nhập không hợp lệ."

    # Giữ cách truy vấn tương thích với phiên bản Flask-SQLAlchemy của đồ án.
    user = User.query.get(user_id)
    if user is None:
        return None, "Tài khoản không còn tồn tại."
    return user, ""


def mobile_auth_required(view_function):
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        user, error = get_mobile_user_from_request()
        if user is None:
            return jsonify({"error": error or "Vui lòng đăng nhập."}), 401
        g.mobile_user = user
        return view_function(*args, **kwargs)

    return wrapped


def web_or_mobile_auth_required(view_function):
    """Cho phép Web dùng cookie Flask-Login, Mobile dùng Bearer token."""
    @wraps(view_function)
    def wrapped(*args, **kwargs):
        if current_user.is_authenticated:
            return view_function(*args, **kwargs)

        user, error = get_mobile_user_from_request()
        if user is None:
            return jsonify({"error": error or "Vui lòng đăng nhập."}), 401
        g.mobile_user = user
        return view_function(*args, **kwargs)

    return wrapped


# ==================================================
# LOAD DATA.YAML VÀ MODEL MỚI
# ==================================================
def load_names_from_data_yaml(path: Path):
    """Đọc names trong data.yaml, không cần cài PyYAML."""
    if not path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy data.yaml: {path}")
        return {}

    names = {}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue

        if line.strip() == "names:":
            in_names = True
            continue

        if in_names:
            # Kết thúc block names khi gặp key cấp root khác
            if not raw_line.startswith((" ", "\t")) and ":" in line:
                break

            match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if match:
                idx = int(match.group(1))
                name = match.group(2).strip().strip('"').strip("'")
                names[idx] = name

    return names


YOLO_NAMES = load_names_from_data_yaml(DATA_YAML_PATH)

if not MODEL_PATH.exists() and not SKIP_MODEL_LOAD:
    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_PATH}\n"
        "Bạn hãy kiểm tra lại đường dẫn MODEL_PATH trong server.py."
    )

model = None if SKIP_MODEL_LOAD else YOLO(str(MODEL_PATH))

# Nếu model có names thì ưu tiên names từ model; nếu không có thì dùng data.yaml.
try:
    if model is not None and hasattr(model, "names") and model.names:
        MODEL_NAMES = {int(k): v for k, v in model.names.items()}
    else:
        MODEL_NAMES = YOLO_NAMES
except Exception:
    MODEL_NAMES = YOLO_NAMES

print("Đã load model:" if model is not None else "Đang chạy chế độ kiểm thử, chưa load model:", MODEL_PATH)
print("Số class:", len(MODEL_NAMES))
print("Thiết bị chạy:", DEVICE)


# ==================================================
# OCR TÙY CHỌN CHO BIỂN PHỤ CHỮ
# ==================================================
reader = None
try:
    import easyocr
    reader = easyocr.Reader(["vi", "en"], gpu=False)
    print("EasyOCR đã sẵn sàng.")
except Exception as e:
    print("Không bật EasyOCR, hệ thống vẫn chạy YOLO bình thường.")
    print("Lý do:", e)


# ==================================================
# TỪ ĐIỂN Ý NGHĨA NHÃN MỚI
# ==================================================
SIGN_INFO = {
    "P.102": ("Biển cấm", "Cấm đi ngược chiều", "Cấm đi vào theo chiều này."),
    "P.103a": ("Biển cấm", "Cấm xe ô tô", "Xe ô tô không được đi vào đoạn đường này."),
    "P.103b": ("Biển cấm", "Cấm ô tô rẽ phải", "Xe ô tô không được rẽ phải."),
    "P.103c": ("Biển cấm", "Cấm ô tô rẽ trái", "Xe ô tô không được rẽ trái."),
    "P.106a": ("Biển cấm", "Cấm xe ô tô tải", "Xe tải không được đi vào đoạn đường này."),
    "P.106b": (
        "Biển cấm",
        "Cấm xe ô tô tải có khối lượng chuyên chở lớn hơn trị số ghi trên biển",
        "Cấm xe ô tô tải có khối lượng chuyên chở theo giấy kiểm định lớn hơn trị số ghi trên biển; quy chuẩn còn áp dụng đối với máy kéo và xe máy chuyên dùng.",
    ),
    "P.108a": (
        "Biển cấm",
        "Cấm xe sơ-mi rơ-moóc",
        "Cấm xe sơ-mi rơ-moóc và xe kéo rơ-moóc đi vào, trừ xe được ưu tiên theo quy định.",
    ),
    "P.111a": ("Biển cấm", "Cấm xe gắn máy", "Xe gắn máy không được đi vào; biển không cấm xe đạp."),
    "P.112": ("Biển cấm", "Cấm người đi bộ", "Người đi bộ không được đi vào khu vực này."),
    "P.115": (
        "Biển cấm",
        "Hạn chế trọng tải toàn bộ xe",
        "Cấm xe có tổng trọng lượng thực tế của xe, người, hành lý và hàng hóa vượt trị số ghi trên biển.",
    ),
    "P.116": (
        "Biển cấm",
        "Hạn chế tải trọng trên trục xe",
        "Cấm xe có tải trọng phân bổ trên một trục bất kỳ vượt trị số ghi trên biển.",
    ),
    "P.117": (
        "Biển cấm",
        "Hạn chế chiều cao",
        "Cấm xe có chiều cao tính từ mặt đường đến điểm cao nhất của xe hoặc hàng vượt trị số ghi trên biển.",
    ),
    "P.123a": ("Biển cấm", "Cấm rẽ trái", "Không được rẽ trái."),
    "P.123b": ("Biển cấm", "Cấm rẽ phải", "Không được rẽ phải."),
    "P.124a1": ("Biển cấm", "Cấm quay đầu xe", "Không được quay đầu xe."),
    "P.124a2": ("Biển cấm", "Cấm quay đầu xe", "Không được quay đầu xe."),
    "P.124b1": ("Biển cấm", "Cấm ô tô quay đầu", "Xe ô tô không được quay đầu."),
    "P.124b2": ("Biển cấm", "Cấm ô tô quay đầu", "Xe ô tô không được quay đầu."),
    "P.124c": ("Biển cấm", "Cấm rẽ trái và quay đầu", "Không được rẽ trái hoặc quay đầu."),
    "P.124d": ("Biển cấm", "Cấm rẽ phải và quay đầu", "Không được rẽ phải hoặc quay đầu."),
    "P.124e": ("Biển cấm", "Cấm ô tô rẽ trái và quay đầu", "Xe ô tô không được rẽ trái hoặc quay đầu."),
    "P.127c": ("Biển cấm", "Tốc độ tối đa theo loại phương tiện", "Chú ý tốc độ tối đa áp dụng cho từng loại phương tiện."),
    "P.128": ("Biển cấm", "Cấm sử dụng còi", "Không được sử dụng còi tại khu vực này."),
    "P.130": ("Biển cấm", "Cấm dừng và đỗ xe", "Không được dừng và đỗ xe tại khu vực có hiệu lực."),
    "P.131a": ("Biển cấm", "Cấm đỗ xe", "Không được đỗ xe tại khu vực có hiệu lực."),
    "P.131b": ("Biển cấm", "Cấm đỗ xe ngày lẻ", "Không được đỗ xe vào ngày lẻ."),
    "P.131c": ("Biển cấm", "Cấm đỗ xe ngày chẵn", "Không được đỗ xe vào ngày chẵn."),
    "P.137": ("Biển cấm", "Cấm rẽ trái, rẽ phải", "Không được rẽ trái hoặc rẽ phải."),
    "P.138": ("Biển cấm", "Cấm đi thẳng, rẽ trái", "Không được đi thẳng hoặc rẽ trái."),
    "P.139": ("Biển cấm", "Cấm đi thẳng, rẽ phải", "Không được đi thẳng hoặc rẽ phải."),

    "R.302a": ("Biển hiệu lệnh", "Hướng phải đi vòng chướng ngại vật bên phải", "Đi vòng qua chướng ngại vật theo hướng bên phải."),
    "R.302b": ("Biển hiệu lệnh", "Hướng phải đi vòng chướng ngại vật bên trái", "Đi vòng qua chướng ngại vật theo hướng bên trái."),
    "R.303": ("Biển hiệu lệnh", "Nơi giao nhau chạy theo vòng xuyến", "Đi theo chiều vòng xuyến."),
    "R.411": ("Biển hiệu lệnh", "Hướng đi trên mỗi làn đường", "Đi đúng hướng quy định trên từng làn đường."),
    "R.415a": ("Biển hiệu lệnh", "Phân làn đường theo phương tiện", "Đi đúng làn đường dành cho loại phương tiện của bạn."),
    "R.420": ("Biển hiệu lệnh", "Bắt đầu khu đông dân cư", "Chú ý quy định khi đi vào khu đông dân cư."),

    "I.414a": ("Biển chỉ dẫn", "Chỉ hướng đường", "Chú ý hướng đường được chỉ dẫn."),
    "I.414b": ("Biển chỉ dẫn", "Chỉ hướng đường", "Chú ý hướng đường được chỉ dẫn."),
    "I.416": ("Biển chỉ dẫn", "Đường tránh", "Chú ý thông tin đường tránh phía trước."),
    "I.423a": ("Biển chỉ dẫn", "Vị trí người đi bộ sang ngang", "Chú ý người đi bộ sang đường."),
    "I.423b": ("Biển chỉ dẫn", "Vị trí người đi bộ sang ngang", "Chú ý người đi bộ sang đường."),
    "I.425": ("Biển chỉ dẫn", "Bệnh viện", "Chú ý khu vực bệnh viện."),
    "I.434a": ("Biển chỉ dẫn", "Bến xe buýt", "Chú ý khu vực bến xe buýt."),
    "I.441a": ("Biển chỉ dẫn", "Báo hiệu phía trước có công trường", "Chú ý công trường phía trước."),
    "I.441b": ("Biển chỉ dẫn", "Báo hiệu phía trước có công trường", "Chú ý công trường phía trước."),
    "I.441c": ("Biển chỉ dẫn", "Báo hiệu phía trước có công trường", "Chú ý công trường phía trước."),

    "S.501": ("Biển phụ", "Phạm vi tác dụng của biển", "Biển phụ quy định phạm vi tác dụng."),
    "S.502": ("Biển phụ", "Khoảng cách đến đối tượng báo hiệu", "Biển phụ cho biết khoảng cách đến đối tượng báo hiệu."),
    "S.505a": ("Biển phụ", "Loại xe áp dụng", "Biển phụ quy định loại phương tiện áp dụng."),
    "S.508a": ("Biển phụ", "Hướng tác dụng của biển", "Biển phụ chỉ hướng tác dụng."),
    "S.508b": ("Biển phụ", "Hướng tác dụng của biển", "Biển phụ chỉ hướng tác dụng."),
    "S.509a": ("Biển phụ", "Thuyết minh bằng chữ", "Biển phụ thuyết minh thêm nội dung áp dụng."),
    "bien_phu_chu": ("Biển phụ", "Cho phép xe hai bánh rẽ phải khi đèn đỏ", "Xe hai bánh được phép rẽ phải khi đèn đỏ theo chỉ dẫn của biển."),
    "bp_cam_dung_do_ngay_chan": ("Biển phụ", "Cấm dừng, đỗ xe ngày chẵn", "Cấm dừng và đỗ xe vào ngày chẵn."),
    "bp_cam_dung_do_ngay_le": ("Biển phụ", "Cấm dừng, đỗ xe ngày lẻ", "Cấm dừng và đỗ xe vào ngày lẻ."),
    "cam_xe_hai_banh_re_trai": ("Biển phụ/biển cấm", "Cấm xe hai bánh rẽ trái", "Xe hai bánh không được rẽ trái."),

    "den_do": ("Đèn giao thông", "Đèn đỏ", "Đèn đỏ, vui lòng dừng lại."),
    "den_xanh": ("Đèn giao thông", "Đèn xanh", "Đèn xanh, được phép đi nếu bảo đảm an toàn."),
    "den_vang": ("Đèn giao thông", "Đèn vàng", "Đèn vàng, giảm tốc độ và chú ý quan sát."),
}

WARNING_NAMES = {
    "W.201a": "Chỗ ngoặt nguy hiểm bên trái",
    "W.201b": "Chỗ ngoặt nguy hiểm bên phải",
    "W.202a": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.202b": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    "W.203c": "Đường bị thu hẹp",
    "W.205a": "Đường giao nhau",
    "W.205b": "Đường giao nhau",
    "W.205c": "Đường giao nhau",
    "W.205d": "Đường giao nhau",
    "W.205e": "Đường giao nhau",
    "W.206": "Giao nhau chạy theo vòng xuyến",
    "W.207a": "Giao nhau với đường không ưu tiên",
    "W.207b": "Giao nhau với đường không ưu tiên",
    "W.207c": "Giao nhau với đường không ưu tiên",
    "W.207d": "Giao nhau với đường không ưu tiên",
    "W.207e": "Giao nhau với đường không ưu tiên",
    "W.207l": "Giao nhau với đường không ưu tiên",
    "W.208": "Giao nhau với đường ưu tiên",
    "W.209": "Giao nhau có tín hiệu đèn",
    "W.219": "Dốc xuống nguy hiểm",
    "W.224": "Đường người đi bộ cắt ngang",
    "W.225": "Trẻ em",
    "W.227": "Công trường",
    "W.239a": "Đường cáp điện phía trên",
    "W.245a": "Đi chậm",
    "W.247": "Chú ý xe đỗ",
}
for code, name in WARNING_NAMES.items():
    SIGN_INFO[code] = ("Biển cảnh báo", name, f"Chú ý {name.lower()} phía trước.")

# Kiểm tra nhanh để phát hiện class model chưa có mô tả nghiệp vụ.
_missing_sign_info = [
    str(name) for _, name in MODEL_NAMES.items()
    if str(name) not in SIGN_INFO
]
if _missing_sign_info:
    print("[CẢNH BÁO] Class model chưa có SIGN_INFO:", _missing_sign_info)
else:
    print("SIGN_INFO đã khớp đủ", len(MODEL_NAMES), "class của model.")


def get_sign_info(code: str):
    group, name, warning = SIGN_INFO.get(code, ("Biển báo", code.replace("_", " "), f"Phát hiện {code.replace('_', ' ')}."))
    return {
        "code": code,
        "group": group,
        "name": name,
        "warning": warning,
    }


def read_text_from_roi(roi_img):
    if reader is None or roi_img is None or roi_img.size == 0:
        return ""

    try:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        raw = reader.readtext(gray, detail=0)
        text = " ".join(raw).strip()
        text = re.sub(r"\s+", " ", text)
        return text
    except Exception as e:
        print("Lỗi OCR:", e)
        return ""


def vehicle_name(vehicle_type):
    return {
        "all": "chưa lọc phương tiện",
        "xe_may": "xe máy",
        "oto": "ô tô",
        "xe_tai": "xe tải",
        "xe_khach": "xe khách",
        "xe_dap": "xe đạp",
    }.get(vehicle_type, "phương tiện")


def build_audio_text(detections, vehicle_type="all"):
    if not detections:
        return ""

    # Ưu tiên cảnh báo đèn đỏ/vàng trước, sau đó biển cấm, rồi các biển còn lại.
    priority = {
        "den_do": 0,
        "den_vang": 1,
        "P": 2,
        "bp": 3,
        "cam_xe_hai_banh_re_trai": 3,
        "R": 4,
        "W": 5,
        "I": 6,
        "S": 7,
        "bien_phu_chu": 8,
        "den_xanh": 9,
    }

    def get_pri(d):
        c = d["code"]
        if c in priority:
            return priority[c]
        if c.startswith("P."):
            return priority["P"]
        if c.startswith("R."):
            return priority["R"]
        if c.startswith("W."):
            return priority["W"]
        if c.startswith("I."):
            return priority["I"]
        if c.startswith("S."):
            return priority["S"]
        if c.startswith("bp_"):
            return priority["bp"]
        return 99

    dets = sorted(detections, key=get_pri)
    parts = []

    codes = {d["code"] for d in dets}

    # Ghép tình huống biển cấm dừng/đỗ + biển phụ ngày chẵn/ngày lẻ.
    if "P.130" in codes and "bp_cam_dung_do_ngay_chan" in codes:
        parts.append("Cấm dừng và đỗ xe vào ngày chẵn")
    elif "P.130" in codes and "bp_cam_dung_do_ngay_le" in codes:
        parts.append("Cấm dừng và đỗ xe vào ngày lẻ")

    for d in dets:
        code = d["code"]
        warning = d.get("warning", "")

        # Những trường hợp đã ghép ở trên thì không đọc lặp lại.
        if code in {"P.130", "bp_cam_dung_do_ngay_chan", "bp_cam_dung_do_ngay_le"} and any("Cấm dừng và đỗ xe vào" in p for p in parts):
            continue

        if code == "den_xanh":
            # Đèn xanh ít cần cảnh báo gấp, vẫn hiển thị nhưng không nhất thiết đọc nếu đã có biển khác.
            if len(dets) == 1:
                parts.append(warning)
            continue

        if code == "P.127c":
            parts.append(f"Biển giới hạn tốc độ theo loại phương tiện, người lái {vehicle_name(vehicle_type)} cần chú ý tốc độ tương ứng")
            continue

        if code == "R.415a":
            parts.append(f"Biển phân làn theo phương tiện, {vehicle_name(vehicle_type)} cần đi đúng làn quy định")
            continue

        if code == "cam_xe_hai_banh_re_trai":
            if vehicle_type in {"xe_may", "xe_dap"}:
                parts.append("Xe hai bánh không được rẽ trái")
            else:
                parts.append("Có biển cấm xe hai bánh rẽ trái")
            continue

        # Trong bộ dữ liệu 72 lớp chính thức của đề tài, nhãn bien_phu_chu
        # là biển cho phép xe hai bánh rẽ phải khi đèn đỏ. Không dùng OCR
        # cho nhãn này vì nội dung đã được định nghĩa cố định theo dataset.
        if code == "bien_phu_chu":
            if vehicle_type in {"xe_may", "xe_dap"}:
                parts.append("Xe hai bánh được phép rẽ phải khi đèn đỏ theo chỉ dẫn của biển")
            elif vehicle_type == "all":
                parts.append("Biển cho phép xe hai bánh rẽ phải khi đèn đỏ")
            else:
                parts.append("Biển cho phép xe hai bánh rẽ phải khi đèn đỏ, không áp dụng cho phương tiện đang chọn")
            continue

        parts.append(warning or d["name"])

    # Loại trùng, giữ thứ tự
    seen = set()
    final_parts = []
    for p in parts:
        p = p.strip().rstrip(".")
        if p and p not in seen:
            final_parts.append(p)
            seen.add(p)

    if not final_parts:
        return ""

    if len(final_parts) == 1:
        return final_parts[0] + " ở phía trước."
    return ". ".join(final_parts) + "."


def _box_iou(box_a, box_b):
    """IoU giữa 2 bounding box [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _deduplicate_detections(detections, iou_threshold=0.68):
    """
    Loại box trùng gần như cùng một vị trí nhưng model gán nhiều class khác nhau.
    Giữ box có confidence cao hơn. Không gộp các biển nằm cạnh nhau nếu IoU thấp.
    """
    if not detections:
        return []

    ordered = sorted(
        detections,
        key=lambda d: float(d.get("conf", 0.0)),
        reverse=True,
    )

    kept = []
    for det in ordered:
        if any(
            _box_iou(det.get("box", [0, 0, 0, 0]), old.get("box", [0, 0, 0, 0]))
            >= iou_threshold
            for old in kept
        ):
            continue
        kept.append(det)

    # Trả lại thứ tự từ confidence cao xuống thấp để frontend ổn định hơn.
    return kept


def detect_frame(frame, conf_threshold=0.35, imgsz=640, vehicle_type="all"):
    """
    Nhận diện một frame.

    - Camera/video có thể truyền imgsz=512 để giảm thời gian CPU.
    - Ảnh upload vẫn có thể dùng imgsz=640 để ưu tiên độ chính xác.
    - Loại box chồng gần như cùng vị trí trước khi sinh cảnh báo.
    """
    if model is None:
        raise RuntimeError(
            "Model YOLO chưa được tải. Hãy bỏ DATN_SKIP_MODEL_LOAD khi chạy demo nhận diện."
        )

    results = model.predict(
        source=frame,
        imgsz=int(imgsz),
        conf=float(conf_threshold),
        iou=0.45,
        device=DEVICE,
        verbose=False,
        max_det=30,
    )

    raw_detections = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            code = MODEL_NAMES.get(
                cls_id,
                YOLO_NAMES.get(cls_id, str(cls_id))
            )
            info = get_sign_info(code)

            raw_detections.append({
                "cls_id": cls_id,
                "code": code,
                "label": info["name"],
                "group": info["group"],
                "name": info["name"],
                "warning": info["warning"],
                "conf": round(conf, 3),
                "box": [x1, y1, x2, y2],
                "ocr_text": "",
            })

    # Một vùng biển báo đôi khi bị YOLO trả nhiều class chồng lên nhau.
    detections = _deduplicate_detections(raw_detections, iou_threshold=0.68)

    annotated = frame.copy()

    # Chỉ OCR và vẽ trên các detection cuối cùng để giảm công việc thừa.
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        code = detection["code"]
        conf = float(detection["conf"])
        info = get_sign_info(code)

        if code in {"S.509a", "S.502", "S.501"}:
            roi = frame[
                max(0, y1):max(0, y2),
                max(0, x1):max(0, x2)
            ]
            detection["ocr_text"] = read_text_from_roi(roi)

        label_draw = f"{code} - {info['name']} {conf:.2f}"
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 220, 80),
            2
        )
        (tw, th), _ = cv2.getTextSize(
            label_draw,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - th - 10)),
            (x1 + tw + 6, y1),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            annotated,
            label_draw,
            (x1 + 3, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            2
        )

    audio_text = build_audio_text(detections, vehicle_type)
    return annotated, audio_text, detections


# ==================================================
# ROUTES ĐĂNG NHẬP
# ==================================================
@app.route("/api/mobile/register", methods=["POST"])
def api_mobile_register():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    confirm_password = str(payload.get("confirm_password", password))

    if len(username) < 3 or len(username) > 50:
        return jsonify({"error": "Tên đăng nhập phải từ 3 đến 50 ký tự."}), 400
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        return jsonify({
            "error": "Tên đăng nhập chỉ gồm chữ không dấu, số, dấu chấm, gạch dưới hoặc gạch ngang."
        }), 400
    if len(password) < 6:
        return jsonify({"error": "Mật khẩu phải có ít nhất 6 ký tự."}), 400
    if password != confirm_password:
        return jsonify({"error": "Mật khẩu nhập lại chưa khớp."}), 400

    existing_user = User.query.filter(
        db.func.lower(User.username) == username.lower()
    ).first()
    if existing_user:
        return jsonify({"error": "Tên đăng nhập đã tồn tại."}), 409

    try:
        user = User(
            username=username,
            password=generate_password_hash(password, method="pbkdf2:sha256"),
        )
        db.session.add(user)
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({"error": "Không thể tạo tài khoản. Vui lòng thử lại."}), 500

    return jsonify({
        "message": "Đăng ký thành công.",
        "token": create_mobile_token(user),
        "expires_in": MOBILE_TOKEN_MAX_AGE,
        "user": {"id": user.id, "username": user.username},
    }), 201


@app.route("/api/mobile/login", methods=["POST"])
def api_mobile_login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    if not username or not password:
        return jsonify({"error": "Vui lòng nhập tên đăng nhập và mật khẩu."}), 400

    user = User.query.filter(
        db.func.lower(User.username) == username.lower()
    ).first()
    if user is None or not check_password_hash(user.password, password):
        return jsonify({"error": "Sai tên đăng nhập hoặc mật khẩu."}), 401

    return jsonify({
        "message": "Đăng nhập thành công.",
        "token": create_mobile_token(user),
        "expires_in": MOBILE_TOKEN_MAX_AGE,
        "user": {"id": user.id, "username": user.username},
    })


@app.route("/api/mobile/me", methods=["GET"])
@mobile_auth_required
def api_mobile_me():
    user = g.mobile_user
    return jsonify({"user": {"id": user.id, "username": user.username}})


@app.route("/api/mobile/logout", methods=["POST"])
@mobile_auth_required
def api_mobile_logout():
    # Token ký không lưu phía server; mobile xóa token khỏi bộ nhớ khi đăng xuất.
    return jsonify({"message": "Đăng xuất thành công."})


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Tên đăng nhập đã tồn tại.")
            return redirect(url_for("register"))

        user = User(username=username, password=generate_password_hash(password, method="pbkdf2:sha256"))
        db.session.add(user)
        db.session.commit()
        flash("Đăng ký thành công. Bạn có thể đăng nhập ngay.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("home"))

        flash("Sai tên đăng nhập hoặc mật khẩu.")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template("index.html")


# ==================================================
# API NHẬN DIỆN ẢNH/CAMERA/VIDEO FRAME
# ==================================================
@app.route("/detect", methods=["POST"])
@login_required
def detect():
    try:
        request_started = time.perf_counter()

        payload = request.get_json(force=True) or {}
        data_url = payload.get("image")
        if not data_url:
            return jsonify({"error": "Không có ảnh đầu vào"}), 400

        conf = float(payload.get("conf", 0.35))
        vehicle_type = payload.get("vehicle_type", "all")
        source_type = str(payload.get("source_type", "upload")).lower()

        # Camera/video ưu tiên tốc độ; ảnh upload ưu tiên độ chính xác.
        default_imgsz = 640 if source_type == "upload" else 512
        try:
            imgsz = int(payload.get("imgsz", default_imgsz))
        except (TypeError, ValueError):
            imgsz = default_imgsz

        # Chỉ cho một số kích thước an toàn để tránh request bất thường.
        imgsz = min(640, max(384, imgsz))

        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        img_data = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Không đọc được ảnh"}), 400

        infer_started = time.perf_counter()
        _, audio_text, detections = detect_frame(
            frame,
            conf_threshold=conf,
            imgsz=imgsz,
            vehicle_type=vehicle_type,
        )
        inference_ms = round((time.perf_counter() - infer_started) * 1000, 1)
        total_ms = round((time.perf_counter() - request_started) * 1000, 1)

        return jsonify({
            "detections": detections,
            "audio_text": audio_text,
            "count": len(detections),
            # Frontend dùng đúng kích thước ảnh thực mà YOLO đã nhận để quy đổi
            # tọa độ box sang vùng object-fit: contain, kể cả ảnh/camera quay dọc.
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "vehicle_type": vehicle_type,
            "source_type": source_type,
            "imgsz": imgsz,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mobile/detect", methods=["POST"])
@mobile_auth_required
def api_mobile_detect():
    try:
        request_started = time.perf_counter()
        payload = request.get_json(force=True) or {}

        data_url = payload.get("image")
        if not data_url:
            return jsonify({"error": "Không có ảnh đầu vào"}), 400

        conf = float(payload.get("conf", 0.35))
        vehicle_type = payload.get("vehicle_type", "all")

        try:
            imgsz = int(payload.get("imgsz", 640))
        except (TypeError, ValueError):
            imgsz = 640
        imgsz = min(640, max(384, imgsz))

        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        img_data = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Không đọc được ảnh"}), 400

        infer_started = time.perf_counter()
        annotated, audio_text, detections = detect_frame(
            frame,
            conf_threshold=conf,
            imgsz=imgsz,
            vehicle_type=vehicle_type,
        )
        inference_ms = round((time.perf_counter() - infer_started) * 1000, 1)

        # Mobile upload ảnh cần ảnh đã vẽ box để hiển thị trực tiếp.
        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )

        annotated_image = None
        if ok:
            encoded = base64.b64encode(buffer).decode("utf-8")
            annotated_image = "data:image/jpeg;base64," + encoded

        total_ms = round((time.perf_counter() - request_started) * 1000, 1)

        return jsonify({
            "detections": detections,
            "audio_text": audio_text,
            "count": len(detections),
            "vehicle_type": vehicle_type,
            "annotated_image": annotated_image,
            "imgsz": imgsz,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        })

    except Exception as e:
        print("[MOBILE DETECT ERROR]", e)
        return jsonify({"error": str(e)}), 500

# ==================================================
# MOBILE LIVE: NHẬN DIỆN LIÊN TỤC THEO HÀNH TRÌNH
# Dán route này ngay SAU /api/mobile/detect trong server.py.
# Khác /api/mobile/detect ở chỗ KHÔNG encode annotated_image,
# nên response nhẹ hơn đáng kể cho camera chạy liên tục.
# ==================================================
@app.route("/api/mobile/live-detect", methods=["POST"])
@mobile_auth_required
def api_mobile_live_detect():
    try:
        request_started = time.perf_counter()
        payload = request.get_json(force=True) or {}

        data_url = payload.get("image")
        if not data_url:
            return jsonify({"error": "Không có ảnh đầu vào"}), 400

        conf = float(payload.get("conf", 0.45))
        vehicle_type = payload.get("vehicle_type", "all")

        try:
            imgsz = int(payload.get("imgsz", 512))
        except (TypeError, ValueError):
            imgsz = 512

        imgsz = min(640, max(384, imgsz))

        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        img_data = base64.b64decode(data_url)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Không đọc được ảnh"}), 400

        infer_started = time.perf_counter()

        # Live chỉ cần detection + audio, không cần ảnh annotated trả về.
        _, audio_text, detections = detect_frame(
            frame,
            conf_threshold=conf,
            imgsz=imgsz,
            vehicle_type=vehicle_type,
        )

        inference_ms = round(
            (time.perf_counter() - infer_started) * 1000,
            1,
        )

        total_ms = round(
            (time.perf_counter() - request_started) * 1000,
            1,
        )

        return jsonify({
            "detections": detections,
            "audio_text": audio_text,
            "count": len(detections),
            "vehicle_type": vehicle_type,
            "imgsz": imgsz,
            # Chỉ route hành trình Mobile trả thêm kích thước frame để App
            # quy đổi bounding box đúng lên CameraView. Ảnh tĩnh không đổi.
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
            "inference_ms": inference_ms,
            "total_ms": total_ms,
            "live": True,
        })

    except Exception as e:
        print("[MOBILE LIVE DETECT ERROR]", e)
        return jsonify({"error": str(e)}), 500

# ==================================================
# VIDEO: LƯU FILE ĐỂ FRONTEND PHÁT VÀ GỬI FRAME DẦN
# ==================================================
@app.route("/upload_video", methods=["POST"])
@login_required
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Không có file video"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_ext:
        return jsonify({"error": "Định dạng không hỗ trợ. Dùng mp4, avi, mov, mkv hoặc webm."}), 400

    job_id = str(uuid.uuid4())[:8]
    input_filename = f"{job_id}_input{ext}"
    input_path = UPLOAD_FOLDER / input_filename
    file.save(str(input_path))

    return jsonify({
        "job_id": job_id,
        "input_file": input_filename,
        "url": url_for("serve_upload", filename=input_filename),
    })


@app.route("/uploads/<filename>")
@login_required
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)



# ==================================================
# API TRA CỨU BIỂN BÁO
# ==================================================
VEHICLE_LABELS = {
    "all": "Không lọc phương tiện",
    "xe_may": "Xe máy",
    "oto": "Ô tô",
    "xe_tai": "Xe tải",
    "xe_khach": "Xe khách",
    "xe_dap": "Xe đạp",
}


def normalize_vietnamese_text(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9.\s_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_all_sign_codes():
    # Chỉ công khai các lớp thực sự có trong best.pt hiện tại.
    # SIGN_INFO vẫn có thể giữ một số mô tả cũ để tương thích code, nhưng
    # tra cứu/RAG phải bám đúng bộ 72 lớp của model chính thức.
    codes = set()

    try:
        for _, name in MODEL_NAMES.items():
            codes.add(str(name))
    except Exception:
        pass

    if not codes:
        codes = set(SIGN_INFO.keys())

    return sorted(codes)


SELECTABLE_VEHICLES = ["xe_may", "oto", "xe_tai", "xe_khach", "xe_dap"]

# Không dùng fallback “mọi xe” cho những biển có đối tượng hoặc điều kiện riêng.
# mode: universal/restricted/conditional/contextual/non_vehicle.
SIGN_APPLICABILITY_RULES = {
    "P.103a": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại ô tô, gồm ô tô con, xe tải và xe khách; không áp dụng cho xe máy hoặc xe đạp.",
    },
    "P.103b": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại ô tô khi rẽ phải, gồm ô tô con, xe tải và xe khách.",
    },
    "P.103c": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại ô tô khi rẽ trái, gồm ô tô con, xe tải và xe khách.",
    },
    "P.106a": {
        "vehicles": ["xe_tai"],
        "mode": "restricted",
        "text": "Xe ô tô tải; theo quy chuẩn còn áp dụng đối với máy kéo và xe máy chuyên dùng, trừ xe ưu tiên.",
    },
    "P.106b": {
        "vehicles": ["xe_tai"],
        "mode": "conditional",
        "text": "Xe ô tô tải có khối lượng chuyên chở ghi trong giấy kiểm định lớn hơn trị số trên biển; theo quy chuẩn còn áp dụng đối với máy kéo và xe máy chuyên dùng.",
    },
    "P.108a": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "conditional",
        "text": "Xe sơ-mi rơ-moóc và ô tô đang kéo rơ-moóc; không thể kết luận chỉ từ tên loại ô tô nếu chưa biết xe có kéo rơ-moóc hay không.",
    },
    "P.111a": {
        "vehicles": ["xe_may"],
        "mode": "restricted",
        "text": "Xe gắn máy; biển không có giá trị cấm đối với xe đạp.",
    },
    "P.112": {
        "vehicles": [],
        "mode": "non_vehicle",
        "text": "Người đi bộ; đây không phải biển phân loại theo phương tiện trong bộ lọc.",
    },
    "P.115": {
        "vehicles": SELECTABLE_VEHICLES,
        "mode": "conditional",
        "text": "Mọi xe cơ giới và xe thô sơ, kể cả xe ưu tiên, nhưng chỉ khi tổng trọng lượng thực tế của xe, người, hành lý và hàng hóa vượt trị số ghi trên biển.",
    },
    "P.116": {
        "vehicles": SELECTABLE_VEHICLES,
        "mode": "conditional",
        "text": "Mọi xe cơ giới và xe thô sơ, kể cả xe ưu tiên, nhưng chỉ khi tải trọng trên một trục bất kỳ vượt trị số ghi trên biển.",
    },
    "P.117": {
        "vehicles": SELECTABLE_VEHICLES,
        "mode": "conditional",
        "text": "Mọi xe cơ giới và xe thô sơ, kể cả xe ưu tiên, nhưng chỉ khi chiều cao của xe hoặc hàng vượt trị số ghi trên biển.",
    },
    "P.124b1": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại ô tô, gồm ô tô con, xe tải và xe khách.",
    },
    "P.124b2": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại ô tô, gồm ô tô con, xe tải và xe khách.",
    },
    "P.124e": {
        "vehicles": ["oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại ô tô, gồm ô tô con, xe tải và xe khách.",
    },
    "P.127c": {
        "vehicles": SELECTABLE_VEHICLES,
        "mode": "contextual",
        "text": "Chỉ loại phương tiện được thể hiện trên từng phần của biển ghép, theo trị số tốc độ tương ứng.",
    },
    "P.128": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại xe có sử dụng còi; trong các lựa chọn của hệ thống gồm xe máy và các loại ô tô.",
    },
    "P.130": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại xe cơ giới dừng hoặc đỗ ở phía đường có đặt biển, trừ xe ưu tiên theo quy định.",
    },
    "P.131a": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "restricted",
        "text": "Các loại xe cơ giới đỗ ở phía đường có đặt biển.",
    },
    "P.131b": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "conditional",
        "text": "Các loại xe cơ giới đỗ ở phía đường có đặt biển vào ngày lẻ.",
    },
    "P.131c": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "conditional",
        "text": "Các loại xe cơ giới đỗ ở phía đường có đặt biển vào ngày chẵn.",
    },
    "R.415a": {
        "vehicles": SELECTABLE_VEHICLES,
        "mode": "contextual",
        "text": "Từng loại phương tiện theo hình biểu thị trên mỗi làn; phải đọc đúng ký hiệu của làn tương ứng.",
    },
    "S.501": {"vehicles": SELECTABLE_VEHICLES, "mode": "contextual", "text": "Không quy định loại xe độc lập; phải đọc cùng biển chính để biết phạm vi tác dụng."},
    "S.502": {"vehicles": SELECTABLE_VEHICLES, "mode": "contextual", "text": "Không quy định loại xe độc lập; biển chỉ khoảng cách đến đối tượng của biển chính."},
    "S.505a": {"vehicles": SELECTABLE_VEHICLES, "mode": "contextual", "text": "Chỉ loại phương tiện có hình hoặc chữ trên biển phụ; cần đọc nội dung thực tế của biển."},
    "S.508a": {"vehicles": SELECTABLE_VEHICLES, "mode": "contextual", "text": "Không quy định loại xe độc lập; phải đọc cùng biển chính và hướng mũi tên."},
    "S.508b": {"vehicles": SELECTABLE_VEHICLES, "mode": "contextual", "text": "Không quy định loại xe độc lập; phải đọc cùng biển chính và hướng mũi tên."},
    "S.509a": {"vehicles": SELECTABLE_VEHICLES, "mode": "contextual", "text": "Phụ thuộc nội dung chữ và biển chính đi kèm; không được kết luận áp dụng cho mọi xe khi chưa đọc chữ."},
    "bien_phu_chu": {
        "vehicles": ["xe_may", "xe_dap"],
        "mode": "restricted",
        "text": "Xe hai bánh trong phạm vi biển của bộ dữ liệu: xe máy và xe đạp; không áp dụng cho ô tô.",
    },
    "cam_xe_hai_banh_re_trai": {
        "vehicles": ["xe_may", "xe_dap"],
        "mode": "restricted",
        "text": "Xe hai bánh trong phạm vi biển của bộ dữ liệu: xe máy và xe đạp; không áp dụng cho ô tô.",
    },
    "bp_cam_dung_do_ngay_chan": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "contextual",
        "text": "Đọc cùng biển cấm dừng, đỗ chính; áp dụng cho xe cơ giới vào ngày chẵn.",
    },
    "bp_cam_dung_do_ngay_le": {
        "vehicles": ["xe_may", "oto", "xe_tai", "xe_khach"],
        "mode": "contextual",
        "text": "Đọc cùng biển cấm dừng, đỗ chính; áp dụng cho xe cơ giới vào ngày lẻ.",
    },
}


def get_sign_applicability(code):
    rule = SIGN_APPLICABILITY_RULES.get(code)
    if rule:
        return {
            "vehicles": list(rule.get("vehicles") or []),
            "mode": str(rule.get("mode") or "restricted"),
            "text": str(rule.get("text") or "").strip(),
        }

    return {
        "vehicles": list(SELECTABLE_VEHICLES),
        "mode": "universal",
        "text": "Tất cả phương tiện tham gia giao thông trong phạm vi hiệu lực của biển.",
    }


def infer_applicable_vehicles(code):
    return get_sign_applicability(code)["vehicles"]


def infer_warning_level(code, group):
    group_norm = normalize_vietnamese_text(group)

    if code == "den_do":
        return "Nguy hiểm"
    if "bien cam" in group_norm or code.startswith("P."):
        return "Cấm"
    if "canh bao" in group_norm or code.startswith("W."):
        return "Cảnh báo"
    if "hieu lenh" in group_norm or code.startswith("R."):
        return "Bắt buộc"
    if code == "den_vang":
        return "Cảnh báo"
    return "Thông tin"


def infer_actions(code, name, warning):
    code = str(code)
    name_lower = normalize_vietnamese_text(name)

    allowed = "Tiếp tục di chuyển nhưng cần quan sát và tuân thủ chỉ dẫn trên biển báo."
    forbidden = "Không có hành vi bị cấm cụ thể."

    if code.startswith("P.") or "cam" in name_lower:
        allowed = "Chỉ được di chuyển khi không vi phạm nội dung cấm của biển báo."
        forbidden = warning

    if code.startswith("W."):
        allowed = "Được tiếp tục di chuyển nhưng cần giảm tốc độ và chú ý quan sát."
        forbidden = "Không nên chạy nhanh hoặc chủ quan tại khu vực có cảnh báo nguy hiểm."

    if code.startswith("R."):
        allowed = "Được di chuyển theo đúng hướng hoặc làn đường được quy định."
        forbidden = "Không được đi sai hướng hoặc sai làn so với chỉ dẫn."

    if code.startswith("I."):
        allowed = "Được sử dụng thông tin chỉ dẫn để lựa chọn hướng đi phù hợp."
        forbidden = "Không có hành vi bị cấm trực tiếp."

    if code.startswith("S.") or code.startswith("bp_"):
        allowed = "Cần kết hợp với biển chính để xác định phạm vi hoặc đối tượng áp dụng."
        forbidden = "Không được bỏ qua nội dung bổ sung của biển phụ."

    if code == "P.106b":
        allowed = "Xe tải được đi qua nếu khối lượng chuyên chở ghi trong giấy kiểm định không vượt trị số trên biển."
        forbidden = "Xe tải, máy kéo hoặc xe máy chuyên dùng thuộc điều kiện khối lượng của biển không được đi vào."

    if code == "P.108a":
        allowed = "Phương tiện không kéo rơ-moóc được xét theo các biển báo khác trên tuyến."
        forbidden = "Xe sơ-mi rơ-moóc và phương tiện kéo rơ-moóc không được đi vào, trừ trường hợp ưu tiên theo quy định."

    if code == "P.111a":
        allowed = "Xe đạp vẫn được đi qua nếu không có biển cấm khác; người điều khiển xe gắn máy phải chọn đường khác."
        forbidden = "Xe gắn máy không được đi vào đoạn đường có hiệu lực của biển."

    if code == "P.115":
        allowed = "Được đi qua khi tổng trọng lượng thực tế không vượt trị số ghi trên biển."
        forbidden = "Không được đi qua khi tổng trọng lượng thực tế của xe, người, hành lý và hàng hóa vượt trị số ghi trên biển."

    if code == "P.116":
        allowed = "Được đi qua khi tải trọng trên mọi trục xe không vượt trị số ghi trên biển."
        forbidden = "Không được đi qua khi tải trọng trên một trục bất kỳ vượt trị số ghi trên biển."

    if code == "P.117":
        allowed = "Được đi qua khi chiều cao thực tế của xe và hàng không vượt trị số ghi trên biển."
        forbidden = "Không được đi qua khi chiều cao của xe hoặc hàng vượt trị số ghi trên biển."

    if code in {"P.130", "P.131a", "P.131b", "P.131c"}:
        allowed = "Xe cơ giới chỉ được dừng hoặc đỗ khi không vi phạm nội dung, thời gian và phạm vi hiệu lực của biển."
        forbidden = warning

    if code == "bien_phu_chu":
        allowed = "Xe hai bánh được phép rẽ phải khi đèn đỏ theo chỉ dẫn của biển."
        forbidden = "Các phương tiện không thuộc đối tượng của biển không được áp dụng chỉ dẫn này."

    if code == "den_do":
        allowed = "Dừng lại trước vạch dừng hoặc vị trí an toàn."
        forbidden = "Không được vượt đèn đỏ."

    if code == "den_vang":
        allowed = "Giảm tốc độ và chuẩn bị dừng nếu chưa đi qua vạch dừng."
        forbidden = "Không được tăng tốc để cố vượt qua giao lộ."

    if code == "den_xanh":
        allowed = "Được phép di chuyển nếu bảo đảm an toàn."
        forbidden = "Không được chủ quan, vẫn cần quan sát người đi bộ và phương tiện khác."

    return allowed, forbidden


def build_sign_detail(code, vehicle_type="all"):
    info = get_sign_info(code)
    group = info["group"]
    name = info["name"]
    warning = info["warning"]

    applicability = get_sign_applicability(code)
    applies_to = applicability["vehicles"]
    applicability_mode = applicability["mode"]
    applicability_text = applicability["text"]
    applies_to_labels = [VEHICLE_LABELS.get(v, v) for v in applies_to]

    warning_level = infer_warning_level(code, group)
    allowed, forbidden = infer_actions(code, name, warning)

    if vehicle_type == "all":
        selection_status = "overview"
        applies_to_selected = True
        vehicle_note = "Đang xem tổng quan, chưa lọc theo phương tiện."
        recommendation = warning
    elif vehicle_type not in applies_to:
        selection_status = "not_applies"
        applies_to_selected = False
        vehicle_note = f"Biển báo này không áp dụng trực tiếp cho {vehicle_name(vehicle_type)}."
        recommendation = (
            f"Biển này không áp dụng trực tiếp cho {vehicle_name(vehicle_type)}, "
            "tuy nhiên người lái vẫn cần quan sát các biển báo khác trên tuyến đường."
        )
    elif applicability_mode == "conditional":
        selection_status = "conditional"
        applies_to_selected = True
        vehicle_note = f"Chỉ áp dụng cho {vehicle_name(vehicle_type)} khi thỏa điều kiện: {applicability_text}"
        recommendation = warning
    elif applicability_mode == "contextual":
        selection_status = "depends"
        applies_to_selected = True
        vehicle_note = f"Cần đọc biển chính, hình hoặc chữ đi kèm để kết luận cho {vehicle_name(vehicle_type)}."
        recommendation = warning
    else:
        selection_status = "applies"
        applies_to_selected = True
        vehicle_note = f"Biển báo này áp dụng cho {vehicle_name(vehicle_type)}."
        recommendation = warning

    return {
        "code": code,
        "group": group,
        "name": name,
        "meaning": warning,
        "warning": warning,
        "warning_level": warning_level,
        "applies_to": applies_to,
        "applies_to_labels": applies_to_labels,
        "applicability_mode": applicability_mode,
        "applicability_text": applicability_text,
        "selection_status": selection_status,
        "applies_to_selected": applies_to_selected,
        "vehicle_type": vehicle_type,
        "vehicle_name": vehicle_name(vehicle_type),
        "vehicle_note": vehicle_note,
        "allowed_behavior": allowed,
        "forbidden_behavior": forbidden,
        "recommendation": recommendation,
        "voice_text": recommendation,
        "applicability_source_title": "QCVN 41:2024/BGTVT",
        "applicability_source_url": QCVN_41_2024_URL,
    }


def search_signs(query="", group="all", vehicle_type="all"):
    q_norm = normalize_vietnamese_text(query)
    group_norm = normalize_vietnamese_text(group)

    results = []

    for code in get_all_sign_codes():
        detail = build_sign_detail(code, vehicle_type)
        search_text = " ".join([
            detail["code"],
            detail["name"],
            detail["group"],
            detail["meaning"],
            detail["warning"],
            detail["warning_level"],
        ])
        search_norm = normalize_vietnamese_text(search_text)

        if group != "all":
            detail_group_norm = normalize_vietnamese_text(detail["group"])
            if group_norm not in detail_group_norm and detail_group_norm not in group_norm:
                continue

        if vehicle_type != "all" and detail["selection_status"] == "not_applies":
            continue

        score = 100

        if q_norm:
            code_norm = normalize_vietnamese_text(detail["code"])
            name_norm = normalize_vietnamese_text(detail["name"])

            if q_norm == code_norm:
                score = 0
            elif code_norm.startswith(q_norm):
                score = 1
            # Cho phép người dùng gõ tự nhiên như
            # "biển báo cấm đi ngược chiều" thay vì phải đúng y tên biển.
            elif name_norm and name_norm in q_norm:
                score = 1
            elif name_norm.startswith(q_norm):
                score = 2
            elif q_norm in search_norm:
                score = 3
            else:
                stop_words = {
                    "bien", "bao", "bienbao", "cho", "toi", "minh",
                    "tim", "tra", "cuu", "giup", "ve", "cai",
                }
                tokens = [t for t in q_norm.split() if t not in stop_words]
                if tokens and all(t in search_norm for t in tokens):
                    score = 4
                else:
                    continue
        else:
            score = 5

        detail["score"] = score
        results.append(detail)

    results.sort(key=lambda x: (x["score"], x["code"]))
    return results


@app.route("/api/signs", methods=["GET"])
@login_required
def api_search_signs():
    query = request.args.get("query", "").strip()
    group = request.args.get("group", "all").strip()
    vehicle_type = request.args.get("vehicle_type", "all").strip()

    results = search_signs(query=query, group=group, vehicle_type=vehicle_type)

    return jsonify({
        "query": query,
        "group": group,
        "vehicle_type": vehicle_type,
        "count": len(results),
        "results": results[:100],
    })


@app.route("/api/signs/<path:code>", methods=["GET"])
@login_required
def api_sign_detail(code):
    vehicle_type = request.args.get("vehicle_type", "all").strip()
    detail = build_sign_detail(code, vehicle_type)
    return jsonify(detail)



# ==================================================
# TÍCH HỢP RAG PHÁP LUẬT GIAO THÔNG OFFLINE
# ==================================================
def export_sign_knowledge_for_rag():
    """Xuất cơ sở tri thức biển báo hiện tại thành TXT để ChromaDB lập chỉ mục."""
    try:
        data_dir = BASE_DIR / "legal_ai" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "traffic_sign_knowledge.txt"

        lines = [
            "CƠ SỞ TRI THỨC BIỂN BÁO GIAO THÔNG CỦA HỆ THỐNG",
            "Nguồn: Bộ nhãn và quy tắc nghiệp vụ của đồ án nhận diện biển báo giao thông.",
            "",
        ]
        for code in get_all_sign_codes():
            detail = build_sign_detail(code, "all")
            lines.extend([
                f"Mã biển: {detail['code']}",
                f"Tên biển: {detail['name']}",
                f"Nhóm biển: {detail['group']}",
                f"Ý nghĩa: {detail['meaning']}",
                f"Đối tượng/điều kiện áp dụng: {detail['applicability_text']}",
                f"Hành vi được phép: {detail['allowed_behavior']}",
                f"Hành vi bị cấm hoặc cần chú ý: {detail['forbidden_behavior']}",
                f"Khuyến nghị an toàn: {detail['recommendation']}",
                "",
            ])

        new_content = "\n".join(lines).strip() + "\n"
        old_content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if old_content != new_content:
            output_path.write_text(new_content, encoding="utf-8")
            print("[RAG] Đã cập nhật cơ sở tri thức biển báo:", output_path)
        return str(output_path)
    except Exception as exc:
        print("[RAG] Không thể xuất cơ sở tri thức biển báo:", exc)
        return ""


def build_rag_sign_context(code, vehicle_type="all"):
    """Tạo ngữ cảnh biển báo chính xác từ SIGN_INFO để đưa thẳng cho Qwen."""
    code = str(code or "").strip()
    if not code:
        return ""
    try:
        detail = build_sign_detail(code, vehicle_type)
        status_text = {
            "overview": "Chưa lọc phương tiện",
            "applies": "Có áp dụng",
            "not_applies": "Không áp dụng trực tiếp",
            "conditional": "Áp dụng có điều kiện",
            "depends": "Phụ thuộc biển chính, hình hoặc chữ đi kèm",
        }.get(detail.get("selection_status"), "Chưa xác định")
        return (
            f"Mã biển: {detail['code']}\n"
            f"Tên biển: {detail['name']}\n"
            f"Nhóm biển: {detail['group']}\n"
            f"Ý nghĩa: {detail['meaning']}\n"
            f"Phương tiện đang chọn: {detail['vehicle_name']}\n"
            f"Kết luận áp dụng: {status_text}\n"
            f"Đối tượng/điều kiện áp dụng: {detail['applicability_text']}\n"
            f"Hành vi được phép: {detail['allowed_behavior']}\n"
            f"Hành vi bị cấm hoặc cần chú ý: {detail['forbidden_behavior']}\n"
            f"Khuyến nghị: {detail['recommendation']}"
        )
    except Exception as exc:
        print("[RAG] Lỗi tạo ngữ cảnh biển báo:", exc)
        return ""


def sanitize_chat_history(history):
    """
    Chỉ nhận các tin gần nhất nhưng giữ lại metadata cần cho câu hỏi nối tiếp.

    Trước đây frontend chỉ gửi role/text nên các câu như “mức phạt này căn cứ ở
    đâu?” hoặc “nguồn câu vừa rồi?” bị coi là câu hoàn toàn mới. Các trường dưới
    đây đều là dữ liệu do chính server trả về, không phải nội dung HTML tự do.
    """
    if not isinstance(history, list):
        return []
    result = []
    for item in history[-10:]:
        if not isinstance(item, dict):
            continue
        role = "assistant" if item.get("role") in {"assistant", "bot"} else "user"
        content = str(item.get("text") or item.get("content") or "").strip()
        if content:
            clean_item = {"role": role, "text": content[:3000]}

            for key in (
                "mode", "context_code", "vehicle_type", "structured_law_id"
            ):
                value = str(item.get(key) or "").strip()
                if value:
                    clean_item[key] = value[:200]

            raw_sources = item.get("sources")
            if isinstance(raw_sources, list):
                clean_sources = []
                for source in raw_sources[:5]:
                    if not isinstance(source, dict):
                        continue
                    clean_sources.append({
                        "file": str(source.get("file") or source.get("title") or "").strip()[:300],
                        "title": str(source.get("title") or "").strip()[:300],
                        "url": str(source.get("url") or "").strip()[:1000],
                        "url_key": str(source.get("url_key") or "").strip()[:200],
                        "page": source.get("page"),
                        "score": source.get("score"),
                        "source_type": str(source.get("source_type") or "").strip()[:100],
                    })
                if clean_sources:
                    clean_item["sources"] = clean_sources

            result.append(clean_item)
    return result


def rag_status_payload():
    if not RAG_MODULE_AVAILABLE:
        return {
            "available": False,
            "ready": False,
            "initializing": False,
            "stage": "Không tải được module RAG",
            "error": RAG_IMPORT_ERROR,
        }
    try:
        return get_rag_status()
    except Exception as exc:
        return {
            "available": False,
            "ready": False,
            "initializing": False,
            "stage": "Lỗi đọc trạng thái RAG",
            "error": str(exc),
        }


# ==================================================
# CHATBOT TRỢ LÝ ẢO AN TOÀN GIAO THÔNG
# ==================================================
CHATBOT_SUGGESTIONS = [
    "Vi phạm biển này bị phạt bao nhiêu?",
    "Hãy giải thích ý nghĩa của biển này.",
    "Biển này có áp dụng cho xe máy không?",
    "Biển này có áp dụng cho ô tô không?",
    "Mức phạt này căn cứ điều, khoản, điểm nào?",
]

LEGAL_NOTE = (
    "Lưu ý: Thông tin xử phạt trong hệ thống chỉ dùng để tham khảo, phục vụ học tập và demo đồ án. "
    "Khi sử dụng thực tế cần đối chiếu văn bản pháp luật hiện hành hoặc hỏi cơ quan chức năng."
)


TRAFFIC_LAW_KNOWLEDGE = [
    {
        "id": "vuot_den_do",
        "title": "Không chấp hành hiệu lệnh đèn tín hiệu giao thông",
        "keywords": [
            "den do", "vuot den do", "khong chap hanh den", "tin hieu den",
            "den vang", "vuot den vang", "den tin hieu"
        ],
        "related_codes": ["den_do", "den_vang"],
        "fine": {
            "oto": "Đối với ô tô, hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông có thể bị phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng và bị trừ điểm giấy phép lái xe theo quy định hiện hành.",
            "xe_may": "Đối với xe máy, hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông có thể bị phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng và bị trừ điểm giấy phép lái xe theo quy định hiện hành.",
            "xe_tai": "Đối với xe tải/ô tô, hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông có thể bị phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng và bị trừ điểm giấy phép lái xe theo quy định hiện hành.",
            "xe_khach": "Đối với xe khách/ô tô, hành vi không chấp hành hiệu lệnh đèn tín hiệu giao thông có thể bị phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng và bị trừ điểm giấy phép lái xe theo quy định hiện hành.",
            "xe_dap": "Đối với xe đạp, mức xử phạt thường nhẹ hơn xe cơ giới và phụ thuộc quy định cụ thể trong văn bản hiện hành.",
            "all": "Hành vi vượt đèn đỏ/không chấp hành đèn tín hiệu có mức phạt khác nhau theo từng loại phương tiện. Ô tô thường bị xử phạt cao hơn xe máy và có thể bị trừ điểm giấy phép lái xe.",
        },
        "advice": "Khi gặp đèn đỏ cần dừng lại trước vạch dừng. Khi gặp đèn vàng cần giảm tốc độ và dừng lại nếu chưa đi qua vạch dừng, trừ trường hợp đã đi quá gần vạch dừng mà dừng lại có thể gây nguy hiểm.",
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
    {
        "id": "di_nguoc_chieu",
        "title": "Đi ngược chiều hoặc đi vào đường cấm",
        "keywords": ["di nguoc chieu", "nguoc chieu", "duong cam", "cam di nguoc", "p.102", "p102"],
        "related_codes": ["P.102"],
        "fine": {
            "all": "Hành vi đi ngược chiều hoặc đi vào đường cấm là lỗi nguy hiểm, có thể bị phạt tiền, bị trừ điểm giấy phép lái xe và bị xử lý nặng hơn nếu gây tai nạn giao thông.",
        },
        "advice": "Khi gặp biển cấm đi ngược chiều, người lái tuyệt đối không đi vào theo hướng đó và cần chọn hướng đi khác phù hợp.",
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
    {
        "id": "cam_re",
        "title": "Không chấp hành biển cấm rẽ hoặc cấm quay đầu",
        "keywords": ["cam re", "cam re trai", "cam re phai", "cam quay dau", "p.123", "p.124"],
        "related_codes": ["P.123a", "P.123b", "P.124a1", "P.124a2", "P.124b1", "P.124b2", "P.124c", "P.124d", "P.124e"],
        "fine": {
            "all": "Hành vi rẽ hoặc quay đầu xe tại nơi có biển cấm có thể bị xử phạt theo quy định hiện hành. Mức phạt phụ thuộc loại phương tiện và tình huống vi phạm.",
        },
        "advice": "Khi gặp biển cấm rẽ hoặc cấm quay đầu, người lái cần đi theo hướng được phép và không thực hiện hành vi bị cấm trong phạm vi hiệu lực của biển.",
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
    {
        "id": "dung_do_sai",
        "title": "Dừng xe, đỗ xe nơi có biển cấm",
        "keywords": ["cam dung", "cam do", "dung do", "do xe", "dung xe", "p.130", "p.131"],
        "related_codes": ["P.130", "P.131a", "P.131b", "P.131c", "bp_cam_dung_do_ngay_chan", "bp_cam_dung_do_ngay_le"],
        "fine": {
            "all": "Hành vi dừng xe, đỗ xe tại nơi có biển cấm có thể bị xử phạt. Mức phạt phụ thuộc vị trí dừng/đỗ, loại phương tiện và việc có gây cản trở hoặc nguy hiểm hay không.",
        },
        "advice": "Khi gặp biển cấm dừng và đỗ xe, người lái không được dừng hoặc đỗ xe trong phạm vi hiệu lực của biển. Nếu có biển phụ, cần đọc thêm thời gian, ngày áp dụng hoặc phạm vi áp dụng.",
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
    {
        "id": "sai_lan",
        "title": "Đi sai làn đường hoặc không tuân thủ biển phân làn",
        "keywords": ["sai lan", "di sai lan", "phan lan", "dung lan", "r.411", "r.415"],
        "related_codes": ["R.411", "R.415a"],
        "fine": {
            "all": "Hành vi đi sai làn hoặc không tuân thủ chỉ dẫn phân làn có thể bị xử phạt theo quy định hiện hành. Mức phạt phụ thuộc loại phương tiện và tình huống cụ thể.",
        },
        "advice": "Khi gặp biển phân làn hoặc hướng đi trên mỗi làn đường, người lái cần chọn đúng làn dành cho phương tiện của mình và đi đúng hướng được chỉ dẫn.",
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
    {
        "id": "toc_do",
        "title": "Không tuân thủ biển báo tốc độ",
        "keywords": ["toc do", "qua toc do", "gioi han toc do", "p.127", "p.127c"],
        "related_codes": ["P.127c"],
        "fine": {
            "all": "Vi phạm quy định về tốc độ có thể bị phạt theo mức vượt quá tốc độ cho phép. Mức phạt phụ thuộc số km/h vượt quá, loại phương tiện và tình huống cụ thể.",
        },
        "advice": "Khi gặp biển giới hạn tốc độ, người lái cần điều chỉnh tốc độ không vượt quá mức cho phép, đồng thời chú ý biển phụ nếu có quy định riêng theo loại phương tiện.",
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
    {
        "id": "khong_chap_hanh_bien_cam_phuong_tien",
        "title": "Không chấp hành biển cấm phương tiện hoặc giới hạn phương tiện",
        "keywords": [
            "cam xe", "cam oto", "cam xe tai", "cam xe may",
            "khong chap hanh bien cam", "di vao duong cam",
            "p.103", "p.106", "p.108", "p.111", "p.115", "p.116", "p.117"
        ],
        "related_codes": [
            "P.103a", "P.106a", "P.106b", "P.108a",
            "P.111a", "P.115", "P.116", "P.117"
        ],
        "fine": {
            "all": (
                "Hành vi điều khiển phương tiện đi vào nơi có biển cấm hoặc vượt quá "
                "giới hạn ghi trên biển có thể bị xử phạt. Mức tiền cụ thể phải xác định "
                "theo loại phương tiện, nội dung biển và tình tiết vi phạm."
            ),
        },
        "advice": (
            "Không đi vào đoạn đường hoặc khu vực bị cấm đối với phương tiện của mình; "
            "hãy chọn lộ trình khác và chú ý biển phụ nếu có."
        ),
        "legal_basis": "Nghị định 168/2024/NĐ-CP",
    },
]


def find_sign_code_in_question(question_norm):
    """
    Tìm mã biển trong câu hỏi.
    Hỗ trợ:
    - Mã P.130 / P130
    - Tên biển nằm trong câu tự nhiên, ví dụ
      "biển báo cấm đi ngược chiều" -> P.102
    """
    question_norm = normalize_vietnamese_text(question_norm)

    # 1) Ưu tiên mã biển viết trực tiếp.
    for code in get_all_sign_codes():
        c1 = normalize_vietnamese_text(code)
        c2 = c1.replace(".", "")
        if c1 and (c1 in question_norm or c2 in question_norm):
            return code

    # 2) Các tên/biến thể rất thường dùng. Biến thể cụ thể đặt trước.
    aliases = [
        # Hai nhãn tự xây dựng phải đặt trước các cụm “rẽ trái/rẽ phải” chung,
        # nếu không câu hỏi sẽ bị nhận nhầm thành P.123a/P.123b.
        ("cam xe hai banh re trai", "cam_xe_hai_banh_re_trai"),
        ("cam xe 2 banh re trai", "cam_xe_hai_banh_re_trai"),
        ("xe hai banh khong duoc re trai", "cam_xe_hai_banh_re_trai"),
        ("cho phep xe hai banh re phai khi den do", "bien_phu_chu"),
        ("xe hai banh duoc re phai khi den do", "bien_phu_chu"),
        ("xe 2 banh duoc re phai khi den do", "bien_phu_chu"),
        ("bien cho phep re phai khi den do", "bien_phu_chu"),
        ("cam o to re trai", "P.103c"),
        ("cam o to re phai", "P.103b"),
        ("cam di nguoc chieu", "P.102"),
        ("di nguoc chieu", "P.102"),
        ("cam re trai", "P.123a"),
        ("cam re phai", "P.123b"),
        ("cam dung va do xe", "P.130"),
        ("cam dung do", "P.130"),
        ("cam do xe", "P.131a"),
        ("cam xe may", "P.111a"),
        ("cam xe o to", "P.103a"),
        ("cam o to", "P.103a"),
        ("den do", "den_do"),
        ("den xanh", "den_xanh"),
        ("den vang", "den_vang"),
    ]
    for phrase, code in aliases:
        if phrase in question_norm:
            return code

    # 3) Tên chính thức của biển nằm trong câu hỏi dài hơn.
    for code in get_all_sign_codes():
        info = get_sign_info(code)
        name_norm = normalize_vietnamese_text(info.get("name", ""))
        if len(name_norm) >= 4 and name_norm in question_norm:
            return code

    return ""


def find_law_item(question_norm, code=""):
    """
    Tìm nội dung pháp luật/xử phạt gần nhất theo từ khóa hoặc theo mã biển đang chọn.
    """
    if code:
        for item in TRAFFIC_LAW_KNOWLEDGE:
            if code in item.get("related_codes", []):
                return item

    for item in TRAFFIC_LAW_KNOWLEDGE:
        for kw in item.get("keywords", []):
            if normalize_vietnamese_text(kw) in question_norm:
                return item

    return None


def latest_assistant_history_item(history):
    """Lấy câu trả lời gần nhất để xử lý “mức phạt/nguồn vừa rồi”."""
    for item in reversed(history or []):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return item
    return {}


def _history_sources(item):
    sources = item.get("sources") if isinstance(item, dict) else None
    return [source for source in (sources or []) if isinstance(source, dict)]


def _extract_legal_bases_from_answer(text):
    """Lấy TOÀN BỘ căn cứ pháp lý từ câu trả lời trước.

    Hỗ trợ hai dạng server đang sinh ra:
    1) Một phương tiện: ``📜 **Căn cứ pháp lý:** ...``
    2) Nhiều trường hợp: ``• **Ô tô, Xe tải:** ...; căn cứ ...``

    Trả về list ``[(nhãn_trường_hợp, căn_cứ), ...]``. Nhãn rỗng khi câu trả
    lời chỉ có một căn cứ chung. Cách này cũng chạy được với Mobile vì Mobile
    có thể đã bỏ Markdown ** khỏi nội dung trước khi gửi history về server.
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    found = []
    seen = set()

    # Dạng kết quả gộp nhiều nhóm phương tiện.
    # Ví dụ:
    # • **Ô tô, Xe tải, Xe khách:** ...; căn cứ Điểm d khoản 9 ...
    # • **Xe máy:** ...; căn cứ Điểm a khoản 7 ...
    for line in raw.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue

        match = re.search(
            r"^[•\-*]\s*(?:\*\*)?([^:\n]+?)(?:\*\*)?\s*:\s*.*?"
            r"\bcăn\s*cứ\s+(.+?)(?:\.\s*)?$",
            clean_line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        label = match.group(1).strip().strip("* ")
        basis = match.group(2).strip().strip("* .")
        key = (label.casefold(), basis.casefold())
        if basis and key not in seen:
            seen.add(key)
            found.append((label, basis))

    if found:
        return found

    # Dạng một kết quả có nhãn "Căn cứ pháp lý" riêng.
    # Cho phép cả Markdown (**...**) lẫn chuỗi đã được Mobile clean Markdown.
    match = re.search(
        r"Căn\s*cứ\s*pháp\s*lý\s*:\s*\*{0,2}\s*([^\n]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        basis = match.group(1).strip().strip("* .")
        if basis:
            return [("", basis)]

    return []


def _format_previous_legal_bases(bases):
    """Định dạng căn cứ của lượt trước, giữ riêng từng nhóm phương tiện."""
    if not bases:
        return ""

    if len(bases) == 1 and not bases[0][0]:
        return "📜 **Căn cứ của kết quả trước:** " + bases[0][1]

    lines = ["📜 **Căn cứ của kết quả trước theo từng trường hợp:**"]
    for label, basis in bases:
        if label:
            lines.append(f"• **{label}:** {basis}")
        else:
            lines.append(f"• {basis}")
    return "\n".join(lines)


def build_conversation_meta_answer(question, history, rag_status=None):
    """
    Trả lời cố định cho các câu hỏi về nguồn, căn cứ của lượt trước và cơ chế
    dự phòng. Những câu này không được đẩy vào penalty_engine hay RAG vì sẽ dễ
    khớp nhầm một điều khoản không liên quan.

    Giá trị trả về: (answer, sources, mode) hoặc None.
    """
    q_norm = normalize_vietnamese_text(question)
    previous = latest_assistant_history_item(history)
    previous_sources = _history_sources(previous)

    asks_previous_basis = (
        any(phrase in q_norm for phrase in [
            "muc phat nay", "ket qua vua roi", "cau tra loi vua roi",
            "muc phat vua roi", "can cu phap ly vua roi",
        ])
        and any(token in q_norm for token in ["dieu", "khoan", "diem", "can cu"])
    )

    if asks_previous_basis:
        bases = _extract_legal_bases_from_answer(previous.get("text", ""))
        if bases:
            return (
                _format_previous_legal_bases(bases) + "\n\n" + LEGAL_NOTE,
                previous_sources,
                previous.get("mode") or "structured_law",
            )
        return (
            "Mình chưa thấy một kết quả mức phạt có căn cứ pháp lý ở lượt ngay trước. "
            "Bạn hãy hỏi lại đầy đủ hành vi và loại phương tiện, ví dụ: “Ô tô đi sai "
            "làn bị phạt bao nhiêu và căn cứ điều nào?”.",
            previous_sources,
            "rule_based",
        )

    asks_previous_source = (
        "nguon" in q_norm
        and any(token in q_norm for token in [
            "vua roi", "cau hoi truoc", "cau tra loi truoc", "ket qua truoc",
            "dung de tra loi", "lay o dau",
        ])
    )
    if asks_previous_source:
        if previous_sources:
            names = []
            for source in previous_sources:
                name = str(source.get("title") or source.get("file") or "Nguồn dữ liệu").strip()
                if name and name not in names:
                    names.append(name)
            return (
                "📚 **Nguồn của câu trả lời trước:**\n- " + "\n- ".join(names),
                previous_sources,
                previous.get("mode") or "knowledge",
            )

        previous_mode = str(previous.get("mode") or "")
        previous_text = str(previous.get("text") or "")
        if previous_mode == "structured_law" or "Nghị định 168/2024" in previous_text:
            source = {
                "file": "Nghị định 168/2024/NĐ-CP",
                "title": "Nghị định 168/2024/NĐ-CP – Toàn văn",
                "url_key": "nghi_dinh_168_fulltext",
                "url": NGHI_DINH_168_FULLTEXT_URL,
                "source_type": "structured_law",
            }
            return (
                "📚 **Nguồn của câu trả lời trước:** Nghị định 168/2024/NĐ-CP "
                "trong kho mức phạt có cấu trúc của hệ thống.",
                [source],
                "structured_law",
            )

        if previous.get("context_code") or previous_mode == "knowledge":
            source = {
                "file": "Cơ sở tri thức biển báo của hệ thống",
                "title": "Cơ sở tri thức biển báo của hệ thống",
                "source_type": "sign_knowledge",
            }
            return (
                "📚 **Nguồn của câu trả lời trước:** cơ sở tri thức biển báo được "
                "khai báo trong hệ thống.",
                [source],
                "knowledge",
            )

        return (
            "Lượt trước chưa lưu được thông tin nguồn. Bạn hãy hỏi lại câu đầy đủ; "
            "hệ thống sẽ hiển thị nguồn ngay dưới câu trả lời.",
            [],
            "rule_based",
        )

    asks_active_decree = (
        "nghi dinh nao" in q_norm
        and any(token in q_norm for token in ["he thong", "tra muc phat", "dang dung", "du lieu"])
    )
    if asks_active_decree:
        source = {
            "file": "Nghị định 168/2024/NĐ-CP",
            "title": "Nghị định 168/2024/NĐ-CP – Toàn văn",
            "url_key": "nghi_dinh_168_fulltext",
            "url": NGHI_DINH_168_FULLTEXT_URL,
            "source_type": "structured_law",
        }
        return (
            "Phiên bản dữ liệu của đồ án hiện đang tra mức phạt từ kho có cấu trúc "
            "được gắn nguồn **Nghị định 168/2024/NĐ-CP**. Đây là mô tả nguồn dữ liệu "
            "đang nạp trong hệ thống; trước khi áp dụng thực tế vẫn phải đối chiếu văn "
            "bản đang có hiệu lực và các văn bản sửa đổi, bổ sung.",
            [source],
            "structured_law",
        )

    asks_rag_fallback = (
        any(token in q_norm for token in ["rag", "llm", "qwen"])
        and any(token in q_norm for token in ["chua san sang", "khong san sang", "bi loi", "khong hoat dong", "du phong"])
    )
    if asks_rag_fallback:
        ready = bool((rag_status or {}).get("ready"))
        current_state = "Hiện RAG đang sẵn sàng." if ready else "Hiện RAG chưa sẵn sàng."
        return (
            current_state + " Khi RAG/LLM chưa dùng được, hệ thống vẫn trả lời theo thứ tự:\n"
            "1. Tra mức phạt trong `penalties.json` và các bản ghi đã đối chiếu.\n"
            "2. Tra ý nghĩa/phương tiện áp dụng từ cơ sở tri thức biển báo.\n"
            "3. Nếu không có dữ liệu đủ chắc chắn, yêu cầu người dùng nêu rõ hành vi "
            "hoặc thông báo chưa có dữ liệu; hệ thống không tự bịa mức tiền hay điều khoản.",
            [],
            "rule_based",
        )

    return None


def build_two_wheel_scope_answer(question, vehicle_type="all"):
    """Xử lý câu hỏi tự nhiên không nêu rõ mã của hai biển xe hai bánh."""
    q_norm = normalize_vietnamese_text(question)
    mentions_two_wheel_sign = any(phrase in q_norm for phrase in [
        "bien danh cho xe hai banh", "bien cua xe hai banh",
        "bien ap dung cho xe hai banh", "bien xe hai banh",
    ])
    asks_car = vehicle_type in {"oto", "xe_tai", "xe_khach"} or any(
        phrase in q_norm for phrase in ["o to", "xe hoi", "xe con", "xe tai", "xe khach"]
    )
    if not (mentions_two_wheel_sign and asks_car):
        return None

    return (
        "Không. Trong bộ dữ liệu của hệ thống, các biển ghi rõ **xe hai bánh** chỉ "
        "áp dụng cho xe máy/mô tô và xe đạp; không áp dụng trực tiếp cho ô tô. "
        "Nếu bạn muốn biết hành vi cụ thể, hãy nêu mã hoặc tên biển: “Cấm xe hai "
        "bánh rẽ trái” hay “Cho phép xe hai bánh rẽ phải khi đèn đỏ”."
    )


def question_has_concrete_violation(question):
    q_norm = normalize_vietnamese_text(question)
    behaviors = [
        "vuot den do", "khong chap hanh den", "di nguoc chieu", "duong cam",
        "sai lan", "nham lan", "lan duong", "qua toc do", "vuot toc do",
        "cam re", "re trai noi cam", "re phai noi cam", "quay dau noi cam",
        "cam dung", "cam do", "dung xe", "do xe", "nong do con",
        "khong doi mu", "mu bao hiem", "su dung dien thoai", "cho qua so nguoi",
        "khong gan bien so", "che bien so", "khong chap hanh bien",
    ]
    return any(behavior in q_norm for behavior in behaviors)


def question_has_legal_citation(question):
    q_norm = normalize_vietnamese_text(question)
    return bool(
        re.search(r"\bdieu\s*\d+\b", q_norm)
        or re.search(r"\bkhoan\s*\d+\b", q_norm)
        or re.search(r"\bdiem\s*[a-z]\b", q_norm)
    )


def sign_supports_direct_penalty_lookup(code):
    if not code:
        return False
    try:
        group_norm = normalize_vietnamese_text(build_sign_detail(code, "all").get("group", ""))
    except Exception:
        group_norm = ""
    return (
        code.startswith(("P.", "R."))
        or code.startswith("den_")
        or "bien cam" in group_norm
        or "hieu lenh" in group_norm
        or "den tin hieu" in group_norm
    )


def build_penalty_clarification(question, code=""):
    """Chặn việc khớp nhầm mức phạt khi câu hỏi chưa có một hành vi vi phạm."""
    q_norm = normalize_vietnamese_text(question)

    if any(phrase in q_norm for phrase in [
        "khong co trong kho du lieu", "khong co trong du lieu", "khong ton tai trong kho",
    ]):
        return (
            "Mình không thể đưa ra mức phạt cho một điều khoản không có trong kho dữ "
            "liệu. Hệ thống chỉ trả số tiền, điểm GPLX và căn cứ khi tìm được bản ghi "
            "phù hợp; nếu không sẽ từ chối thay vì suy đoán."
        )

    if question_has_concrete_violation(question) or question_has_legal_citation(question):
        return ""

    if code and not sign_supports_direct_penalty_lookup(code):
        detail = build_sign_detail(code, "all")
        return (
            f"Biển **{detail['code']} – {detail['name']}** thuộc nhóm {detail['group'].lower()}. "
            "Bản thân việc nhìn thấy/đi qua biển này chưa phải là một hành vi có mức "
            "phạt cố định. Bạn hãy nêu hành vi thực tế cần tra, ví dụ: đi sai làn, "
            "vượt đèn đỏ, đi ngược chiều hoặc quá tốc độ."
        )

    if not code:
        return (
            "Để tra đúng mức phạt, bạn cần nêu **loại phương tiện + hành vi cụ thể**. "
            "Ví dụ: “Xe máy vượt đèn đỏ bị phạt bao nhiêu?” hoặc “Ô tô đi sai làn "
            "bị trừ mấy điểm?”. Mình sẽ không chọn một lỗi gần giống để trả lời thay."
        )

    return ""


def build_law_answer(law_item, vehicle_type="all"):
    fine_map = law_item.get("fine", {})
    fine_text = (
        fine_map.get(vehicle_type)
        or fine_map.get("all")
        or "Chưa có dữ liệu mức phạt cụ thể cho phương tiện này."
    )
    point_map = law_item.get("point_deduction", {})
    point_text = point_map.get(vehicle_type) or point_map.get("all") or "Chưa có dữ liệu trừ điểm cụ thể."
    extra_map = law_item.get("extra_penalty", {})
    extra_text = extra_map.get(vehicle_type) or extra_map.get("all") or "Chưa có dữ liệu hình phạt bổ sung cụ thể."
    vehicle_name = VEHICLE_LABELS.get(vehicle_type, "Tất cả phương tiện")

    return (
        f"⚖️ **Hành vi:** {law_item['title']}\n\n"
        f"🚗 **Phương tiện:** {vehicle_name}\n\n"
        f"💰 **Mức phạt:** {fine_text}\n\n"
        f"🪪 **Trừ điểm GPLX:** {point_text}\n\n"
        f"📌 **Hình phạt bổ sung:** {extra_text}\n\n"
        f"📜 **Căn cứ pháp lý:** "
        f"{law_item.get('legal_basis', 'Văn bản pháp luật hiện hành')}.\n\n"
        f"✅ **Khuyến nghị:** {law_item.get('advice', '')}\n\n"
        f"{LEGAL_NOTE}"
    )


def build_chatbot_answer(question, vehicle_type="all", context_code=""):
    q_raw = str(question or "").strip()
    q_norm = normalize_vietnamese_text(q_raw)
    vehicle_type = vehicle_type if vehicle_type in VEHICLE_LABELS else "all"

    if not q_norm:
        return (
            "Chào bạn, mình là trợ lý ảo an toàn giao thông. "
            "Bạn có thể hỏi về ý nghĩa biển báo, biển báo áp dụng cho phương tiện nào, "
            "khi gặp biển cần làm gì hoặc một số lỗi vi phạm thường gặp.\n\n"
            "Ví dụ: Biển P.130 là gì? Xe máy vượt đèn đỏ bị phạt như thế nào?"
        ), ""

    code = context_code.strip() if context_code else ""
    if not code:
        code = find_sign_code_in_question(q_norm)

    # Nếu người dùng hỏi nối tiếp, dùng biển đang chọn trong tab tra cứu.
    if context_code and question_uses_selected_sign(q_raw):
        code = context_code.strip()

    asks_fine = any(x in q_norm for x in [
        "phat", "muc phat", "bao nhieu tien", "xu phat", "vi pham", "bi phat", "loi"
    ])

    asks_meaning = any(x in q_norm for x in [
        "la gi", "y nghia", "nghia la", "giai thich", "bien gi"
    ])

    asks_vehicle = any(x in q_norm for x in [
        "ap dung", "xe may", "o to", "xe tai", "xe khach", "xe dap",
        "xe hai banh", "xe 2 banh", "phuong tien"
    ])

    asks_action = any(x in q_norm for x in [
        "can lam gi", "gap", "nen lam", "phai lam", "xu ly", "di nhu nao",
        "co duoc", "thi sao", "the nao", "can nhu nao"
    ])

    asks_group = any(x in q_norm for x in [
        "bien cam", "bien canh bao", "bien nguy hiem", "bien hieu lenh", "bien chi dan", "bien phu", "den giao thong"
    ])

    if any(x in q_norm for x in ["goi y", "huong dan hoi", "hoi duoc gi", "tro giup", "help"]):
        return "Bạn có thể hỏi một trong các câu sau:\n- " + "\n- ".join(CHATBOT_SUGGESTIONS), code

    if asks_fine:
        law_item = find_law_item(q_norm, code)
        if law_item:
            return build_law_answer(law_item, vehicle_type), code

        return (
            "🔎 **Kết quả tra cứu:** Mình chưa tìm thấy mức phạt cụ thể "
            "cho câu hỏi này trong cơ sở dữ liệu hiện tại.\n\n"
            "👉 Bạn có thể thử nêu rõ loại phương tiện và hành vi, ví dụ: "
            "“Xe máy vượt đèn đỏ bị phạt bao nhiêu?” hoặc "
            "“Ô tô đi ngược chiều bị trừ mấy điểm?”.\n\n"
            + LEGAL_NOTE
        ), code

    if code:
        detail = build_sign_detail(code, vehicle_type)
        answer_parts = []

        if asks_meaning or not (asks_vehicle or asks_action):
            answer_parts.append(
                f"📍 **Biển:** {detail['code']} – {detail['name']}\n\n"
                f"🚦 **Ý nghĩa:** {detail['meaning']}\n\n"
                f"📌 **Nhóm biển:** {detail['group']}"
            )

        if asks_vehicle or vehicle_type != "all":
            if vehicle_type == "all":
                vehicle_answer = (
                    "🚗 **Đối tượng/điều kiện áp dụng:** "
                    + detail["applicability_text"]
                )
            else:
                status = detail.get("selection_status")
                if status == "not_applies":
                    vehicle_answer = f"🚗 **Áp dụng:** Biển không áp dụng trực tiếp cho {detail['vehicle_name']}."
                elif status == "conditional":
                    vehicle_answer = (
                        f"🚗 **Áp dụng có điều kiện cho {detail['vehicle_name']}:** "
                        f"{detail['applicability_text']}"
                    )
                elif status == "depends":
                    vehicle_answer = (
                        f"🚗 **Chưa thể kết luận riêng cho {detail['vehicle_name']}:** "
                        f"{detail['applicability_text']}"
                    )
                else:
                    vehicle_answer = f"🚗 **Áp dụng:** Biển có áp dụng cho {detail['vehicle_name']}."
            answer_parts.append(vehicle_answer)

        if asks_action or asks_meaning or vehicle_type != "all":
            answer_parts.append(
                f"✅ **Bạn cần làm:** {detail['allowed_behavior']}\n\n"
                f"⚠️ **Cần tránh:** {detail['forbidden_behavior']}\n\n"
                f"📌 **Khuyến nghị:** {detail['recommendation']}"
            )

        law_item = find_law_item(q_norm, code)
        if law_item and asks_fine:
            answer_parts.append(build_law_answer(law_item, vehicle_type))

        return "\n\n".join(answer_parts), code

    if asks_group:
        if "bien cam" in q_norm:
            return (
                "Biển cấm là nhóm biển báo thể hiện những điều người tham gia giao thông không được thực hiện, "
                "ví dụ cấm đi ngược chiều, cấm rẽ, cấm dừng đỗ hoặc cấm một số loại phương tiện. "
                "Khi gặp biển cấm, người lái cần tuân thủ đúng nội dung cấm trong phạm vi hiệu lực của biển."
            ), code

        if "canh bao" in q_norm or "nguy hiem" in q_norm:
            return (
                "Biển cảnh báo nguy hiểm dùng để báo trước các tình huống có thể gây nguy hiểm như đường giao nhau, "
                "chỗ ngoặt nguy hiểm, công trường, trẻ em hoặc người đi bộ cắt ngang. Khi gặp nhóm biển này, "
                "người lái cần giảm tốc độ và chú ý quan sát."
            ), code

        if "hieu lenh" in q_norm:
            return (
                "Biển hiệu lệnh yêu cầu người tham gia giao thông phải đi theo hướng, làn đường hoặc quy định bắt buộc. "
                "Ví dụ biển vòng xuyến, biển phân làn theo phương tiện hoặc biển hướng đi trên mỗi làn đường."
            ), code

        if "chi dan" in q_norm:
            return (
                "Biển chỉ dẫn cung cấp thông tin hỗ trợ người đi đường như hướng đi, địa điểm, bệnh viện, bến xe buýt "
                "hoặc vị trí người đi bộ sang ngang."
            ), code

        if "bien phu" in q_norm:
            return (
                "Biển phụ thường được đặt dưới biển chính để bổ sung thông tin về thời gian, phạm vi, khoảng cách, "
                "hướng tác dụng hoặc loại phương tiện áp dụng. Khi có biển phụ, cần đọc kết hợp với biển chính."
            ), code

        if "den giao thong" in q_norm:
            return (
                "Đèn giao thông điều khiển quyền đi của các phương tiện tại giao lộ. Đèn đỏ phải dừng lại, "
                "đèn xanh được đi nếu an toàn, đèn vàng cần giảm tốc độ và chuẩn bị dừng nếu chưa qua vạch dừng."
            ), code

    # Tìm gần đúng trong biển báo nếu người dùng hỏi bằng từ khóa
    matched_signs = search_signs(query=q_raw, vehicle_type=vehicle_type)
    if matched_signs:
        top = matched_signs[0]
        return (
            f"Mình tìm thấy biển gần phù hợp nhất là {top['code']} - {top['name']}.\n"
            f"Nhóm biển: {top['group']}.\n"
            f"Ý nghĩa: {top['meaning']}\n"
            f"Khuyến nghị: {top['recommendation']}\n\n"
            "Bạn có thể bấm vào kết quả tra cứu bên trái để xem chi tiết hơn."
        ), top["code"]

    return (
        "Mình chưa hiểu rõ câu hỏi này. Bạn có thể hỏi theo dạng:\n"
        "- Biển P.130 là gì?\n"
        "- Biển này áp dụng cho xe máy không?\n"
        "- Vượt đèn đỏ bị phạt như thế nào?\n"
        "- Gặp biển cảnh báo nguy hiểm cần làm gì?"
    ), code


def is_legal_chat_question(question):
    """
    Nhận diện câu hỏi pháp luật/xử phạt theo cả cách hỏi chính thức
    và cách nói tự nhiên. Mục tiêu là không phụ thuộc vào đúng một từ
    như "phạt" mới đi vào penalty_engine/RAG.
    """
    q_norm = normalize_vietnamese_text(question)

    direct_patterns = [
        r"\bphat\b", r"\bmuc phat\b", r"\bxu phat\b",
        r"\bvi pham\b", r"\bbao nhieu tien\b", r"\bbi phat\b",
        r"\btru diem\b", r"\bmat diem\b", r"\bmay diem\b",
        r"\bbao nhieu diem\b", r"\bgiu bang\b", r"\bgiam bang\b",
        r"\btuoc bang\b", r"\btuoc giay phep\b", r"\bgplx\b",
        r"\bdieu\s*\d+\b", r"\bkhoan\s*\d+\b", r"\bdiem\s*[a-z]\b",
    ]
    if any(
        re.search(pattern, q_norm, flags=re.IGNORECASE)
        for pattern in direct_patterns
    ):
        return True

    # Các cụm “thì sao/có sao không/xử lý thế nào” chỉ là hỏi pháp luật khi
    # câu đã nêu một hành vi vi phạm. Nhờ vậy “Còn ô tô thì sao?” không bị
    # chuyển nhầm sang tra mức phạt.
    colloquial_sanction = any(phrase in q_norm for phrase in [
        "bi gi", "thi bi gi", "bi sao", "co sao khong", "bi xu ly",
        "xu ly sao", "xu ly the nao", "mat gi", "mat bao nhieu",
    ])
    if colloquial_sanction and question_has_concrete_violation(question):
        return True

    # Câu hỏi luật mở chỉ chuyển RAG khi có hành vi cụ thể đi kèm.
    asks_rule = any(phrase in q_norm for phrase in [
        "can cu", "phap luat", "quy dinh", "nghi dinh",
    ])
    return asks_rule and question_has_concrete_violation(question)



def infer_vehicle_type_from_question(question, selected_vehicle="all"):
    """Phương tiện được nói rõ trong câu hỏi luôn ưu tiên hơn bộ chọn giao diện."""
    q_norm = normalize_vietnamese_text(question)
    rules = [
        ("xe_khach", ["xe khach", "xe buyt", "o to khach"]),
        ("xe_tai", ["xe tai", "o to tai"]),
        ("xe_may", ["xe may", "xe mo to", "xe gan may", "xe tay ga"]),
        ("oto", ["o to", "xe hoi", "xe con"]),
        ("xe_dap", ["xe dap"]),
    ]
    for vehicle_type, keywords in rules:
        if any(keyword in q_norm for keyword in keywords):
            return vehicle_type
    if selected_vehicle in VEHICLE_LABELS:
        return selected_vehicle
    return "all"


def question_uses_selected_sign(question):
    """Nhận diện cả câu hỏi trực tiếp lẫn câu nối tiếp về biển đang chọn."""
    q_norm = normalize_vietnamese_text(question)
    references = [
        # Các cách gọi trực tiếp biển đang chọn trên giao diện.
        "bien nay", "bien bao nay", "bien do", "bien bao do", "cai bien nay",
        "vi pham bien nay", "vi pham bien bao nay", "bien dang chon",
        "trich luat bien nay", "trich luat bien bao nay",
        "giai thich bien", "giai thich y nghia cua bien",
        "ap dung cho xe may", "ap dung cho o to", "dang tra cuu",
        "hanh vi dang tra cuu",

        # Các câu nối tiếp dựa trên câu trả lời ngay trước đó.
        "muc phat nay", "truong hop nay", "cau vua roi",
        "noi dung vua roi", "ket qua vua roi", "can cu phap ly vua roi",
        "con o to", "con xe may", "voi o to thi sao", "voi xe may thi sao",
        "o to thi sao", "xe may thi sao", "bien thi sao",
    ]
    return any(reference in q_norm for reference in references)


def build_enriched_rag_question(question, code="", vehicle_type="all"):
    """Bổ sung hành vi, mã biển và phương tiện để RAG tìm đúng điều khoản."""
    parts = [str(question or "").strip()]
    vehicle_label = VEHICLE_LABELS.get(vehicle_type, "Tất cả phương tiện")
    parts.append(f"Loại phương tiện cần tra cứu: {vehicle_label}.")

    if code:
        try:
            detail = build_sign_detail(code, vehicle_type)
            parts.extend([
                f"Biển báo đang xét: {detail['code']} - {detail['name']}.",
                f"Ý nghĩa/hành vi liên quan: {detail['meaning']}",
                f"Hành vi bị cấm hoặc cần chú ý: {detail['forbidden_behavior']}",
            ])
        except Exception:
            parts.append(f"Mã biển báo đang xét: {code}.")

    if is_legal_chat_question(question):
        parts.append(
            "Yêu cầu tra cứu: nêu chính xác mức tiền, trừ điểm hoặc hình phạt bổ sung "
            "và căn cứ điểm, khoản, điều của văn bản nếu dữ liệu có. Không trả lời chung chung."
        )

    return "\n".join(part for part in parts if part)


def legal_answer_is_too_generic(answer):
    """Phát hiện câu trả lời xử phạt chưa có con số hoặc căn cứ cụ thể."""
    text = str(answer or "")
    norm = normalize_vietnamese_text(text)
    generic_phrases = [
        "muc phat phu thuoc", "co the bi xu phat", "theo quy dinh hien hanh",
        "tinh huong cu the", "khong tim thay quy dinh cu the"
    ]
    has_amount = bool(re.search(r"\d[\d\.\s]*(?:đồng|dong|triệu|trieu)", text, flags=re.IGNORECASE))
    has_basis = bool(re.search(r"\b(?:Điều|Khoản|Điểm)\s+\w+", text, flags=re.IGNORECASE))
    return any(phrase in norm for phrase in generic_phrases) or not (has_amount and has_basis)


def law_item_has_specific_fine(law_item, vehicle_type="all"):
    if not law_item:
        return False
    fine_map = law_item.get("fine", {})
    text = str(fine_map.get(vehicle_type) or fine_map.get("all") or "")
    norm = normalize_vietnamese_text(text)
    return bool(re.search(r"\d[\d\.\s]*(?:đồng|dong|triệu|trieu)", text, flags=re.IGNORECASE)) and not any(
        phrase in norm for phrase in ["phu thuoc", "co the bi", "quy dinh hien hanh"]
    )


def build_structured_penalty_source(record):
    """Chuẩn hóa nguồn của bản ghi mức phạt để frontend hiển thị đúng."""
    record = record or {}
    source_title = str(
        record.get("source_title") or "Nghị định 168/2024/NĐ-CP"
    ).strip()
    source_url_key = str(record.get("source_url_key") or "").strip()
    source_url = str(record.get("source_url") or "").strip()

    if not source_url and source_url_key == "nghi_dinh_168_fulltext":
        source_url = NGHI_DINH_168_FULLTEXT_URL

    # Nếu dữ liệu chưa khai báo URL nhưng nguồn là Nghị định 168,
    # vẫn trỏ tới trang toàn văn chính thức thay vì trang metadata/PDF.
    if not source_url and "168/2024" in source_title:
        source_url = NGHI_DINH_168_FULLTEXT_URL

    return {
        "file": source_title,
        "title": f"{source_title} – Toàn văn",
        "page": None,
        "score": 1.0,
        "url_key": source_url_key or "nghi_dinh_168_fulltext",
        "url": source_url,
        "source_type": "structured_law",
    }


def lookup_structured_penalty(question, vehicle_type="all", sign_code=""):
    """
    Thứ tự:
    1) penalties.json nếu bản ghi đã điền số tiền/căn cứ thật.
    2) Bộ fallback đã đối chiếu trong server cho các lỗi demo chính.
    3) None -> chuyển xuống RAG.
    """
    if PENALTY_ENGINE_AVAILABLE and lookup_penalty is not None:
        try:
            record = lookup_penalty(
                question=question,
                vehicle_type=vehicle_type,
                sign_code=sign_code,
            )
            if structured_penalty_record_is_usable(record, vehicle_type):
                return record
            if record:
                print("[LAW] Bỏ qua bản ghi penalties.json còn placeholder/chưa đủ căn cứ:", record.get("id"))
        except Exception as exc:
            print("[LAW] Lỗi tra cứu penalties.json, thử fallback đã đối chiếu:", exc)

    return lookup_verified_builtin_penalty(
        question=question,
        vehicle_type=vehicle_type,
        sign_code=sign_code,
    )


def _structured_field(record, key, vehicle_type="all", default=""):
    value = (record or {}).get(key, default)
    if isinstance(value, dict):
        value = value.get(vehicle_type) or value.get("all") or default
    return str(value or default).strip()


def build_structured_penalty_answer(record, vehicle_type="all"):
    """Trả lời gọn: nhãn và nội dung nằm cùng một dòng."""
    record = record or {}

    # Khi tra theo Điều/Khoản hoặc chưa xác định phương tiện, engine có thể
    # trả nhiều bản ghi cùng lúc. Hiển thị từng trường hợp thay vì hiện
    # "Chưa có dữ liệu mức phạt" ở bản ghi gộp.
    combined = record.get("_combined_records")
    if isinstance(combined, list) and combined:
        title = str(record.get("title") or "Kết quả xử phạt theo từng trường hợp")
        lines = [f"⚖️ **{title}:**"]
        for item in combined:
            item_vehicles = item.get("vehicle_types") or []
            item_vehicle_label = ", ".join(
                VEHICLE_LABELS.get(v, str(v)) for v in item_vehicles
            ) or "Chưa xác định"
            lines.append(
                f"• **{item_vehicle_label}:** "
                f"{item.get('behavior', 'Chưa xác định')}; "
                f"phạt {item.get('fine', 'chưa có dữ liệu')}; "
                f"{item.get('points', 'chưa có dữ liệu trừ điểm')}; "
                f"căn cứ {item.get('legal_basis', 'chưa xác định')}."
            )
        lines.append(LEGAL_NOTE)
        return "\n".join(lines)

    vehicle_label = VEHICLE_LABELS.get(vehicle_type, "Tất cả phương tiện")
    record_vehicles = record.get("vehicle_types") or []
    if vehicle_type == "all" and record_vehicles:
        vehicle_label = ", ".join(
            VEHICLE_LABELS.get(item, str(item)) for item in record_vehicles
        )

    behavior = _structured_field(record, "behavior", vehicle_type, "Chưa xác định")
    fine = _structured_field(record, "fine", vehicle_type, "Chưa có dữ liệu mức phạt cụ thể")
    points = _structured_field(record, "points", vehicle_type, "Chưa có dữ liệu trừ điểm cụ thể")
    extra = _structured_field(record, "additional_penalty", vehicle_type, "Không thấy hình phạt bổ sung riêng")
    basis = _structured_field(record, "legal_basis", vehicle_type, "Chưa xác định")

    # Không chèn dòng trống giữa tiêu đề và nội dung để chatbox thấp, dễ đọc.
    return "\n".join([
        f"⚖️ **Hành vi:** {behavior}",
        f"🚗 **Phương tiện:** {vehicle_label}",
        f"💰 **Mức phạt:** {fine}",
        f"🪪 **Trừ điểm GPLX:** {points}",
        f"📌 **Hình phạt bổ sung:** {extra}",
        f"📜 **Căn cứ pháp lý:** {basis}",
        LEGAL_NOTE,
    ])


@app.route("/api/chatbot/status", methods=["GET"])
@web_or_mobile_auth_required
def api_chatbot_status():
    return jsonify(rag_status_payload())


@app.route("/api/chatbot", methods=["POST"])
@web_or_mobile_auth_required
def api_chatbot():
    try:
        payload = request.get_json(force=True) or {}
        question = str(payload.get("question", "")).strip()
        selected_vehicle = str(payload.get("vehicle_type", "all")).strip()
        context_code = str(payload.get("context_code", "")).strip()
        history = sanitize_chat_history(payload.get("history", []))

        if not question:
            return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400

        vehicle_type = infer_vehicle_type_from_question(question, selected_vehicle)
        detected_code = find_sign_code_in_question(normalize_vietnamese_text(question))

        # Mã viết trực tiếp trong câu hỏi luôn được ưu tiên. Biển đang chọn chỉ được
        # dùng cho các câu như “biển này”, tránh làm lệch câu hỏi độc lập khác.
        resolved_code = detected_code
        if not resolved_code and context_code and question_uses_selected_sign(question):
            resolved_code = context_code

        status = rag_status_payload()

        # Câu hỏi nối tiếp về nguồn/căn cứ và câu hỏi về cơ chế hệ thống phải
        # được xử lý trước bộ phân loại pháp luật. Nếu không, từ “nghị định”,
        # “điều/khoản” rất dễ làm engine khớp nhầm một lỗi giao thông khác.
        meta_answer = build_conversation_meta_answer(question, history, status)
        if meta_answer:
            answer, sources, mode = meta_answer
            return jsonify({
                "answer": answer,
                "context_code": context_code or resolved_code,
                "vehicle_type": vehicle_type,
                "suggestions": CHATBOT_SUGGESTIONS,
                "legal_note": LEGAL_NOTE,
                "mode": mode,
                "rag_status": status,
                "sources": sources,
            })

        # “Biển dành cho xe hai bánh” có thể chỉ một trong hai nhãn tự xây dựng.
        # Khi câu hỏi chỉ hỏi ô tô có áp dụng không, vẫn có thể trả lời chắc chắn
        # mà không tự đoán nhãn nào trong hai nhãn đó.
        two_wheel_answer = build_two_wheel_scope_answer(question, vehicle_type)
        if two_wheel_answer:
            return jsonify({
                "answer": two_wheel_answer,
                "context_code": context_code,
                "vehicle_type": vehicle_type,
                "suggestions": CHATBOT_SUGGESTIONS,
                "legal_note": LEGAL_NOTE,
                "mode": "knowledge",
                "rag_status": status,
                "sources": [{
                    "file": "Cơ sở tri thức biển báo của hệ thống",
                    "title": "Cơ sở tri thức biển báo của hệ thống",
                    "source_type": "sign_knowledge",
                }],
            })

        legal_question = is_legal_chat_question(question)

        # Câu hỏi tra cứu biển báo cơ bản trả lời trực tiếp từ SIGN_INFO.
        # Không cần gọi Qwen nên phản hồi nhanh hơn nhiều trên máy RAM 8 GB.
        if not legal_question:
            quick_answer, quick_code = build_chatbot_answer(
                question=question,
                vehicle_type=vehicle_type,
                context_code=resolved_code,
            )
            return jsonify({
                "answer": quick_answer,
                "context_code": quick_code or resolved_code,
                "vehicle_type": vehicle_type,
                "suggestions": CHATBOT_SUGGESTIONS,
                "legal_note": LEGAL_NOTE,
                "mode": "knowledge",
                "rag_status": status,
                "sources": [
                    {
                        "file": "Cơ sở tri thức biển báo của hệ thống",
                        "page": None,
                        "score": None,
                    }
                ] if resolved_code else [],
            })

        # Không cho penalty_engine chọn một bản ghi gần giống khi câu hỏi mới chỉ
        # nói “hành vi đang tra cứu”, đặc biệt lúc đang chọn biển cảnh báo/chỉ dẫn.
        clarification = build_penalty_clarification(question, resolved_code)
        if clarification:
            clarification_sources = []
            if resolved_code:
                clarification_sources.append({
                    "file": "Cơ sở tri thức biển báo của hệ thống",
                    "title": "Cơ sở tri thức biển báo của hệ thống",
                    "source_type": "sign_knowledge",
                })
            return jsonify({
                "answer": clarification,
                "context_code": resolved_code,
                "vehicle_type": vehicle_type,
                "suggestions": CHATBOT_SUGGESTIONS,
                "legal_note": LEGAL_NOTE,
                "mode": "rule_based",
                "rag_status": status,
                "sources": clarification_sources,
            })

        sign_context = build_rag_sign_context(resolved_code, vehicle_type)

        # ==========================================================
        # TẦNG 1: MỨC PHẠT CÓ CẤU TRÚC
        # Các câu hỏi xử phạt phổ biến ưu tiên penalties.json để lấy
        # mức tiền + trừ điểm + điều/khoản chính xác, không bắt Qwen đoán.
        # Nếu chưa có bản ghi phù hợp mới chuyển xuống RAG ở tầng 2.
        # ==========================================================
        penalty_record = lookup_structured_penalty(
            question=question,
            vehicle_type=vehicle_type,
            sign_code=resolved_code,
        )

        if penalty_record:
            structured_answer = build_structured_penalty_answer(
                penalty_record,
                vehicle_type=vehicle_type,
            )
            return jsonify({
                "answer": structured_answer,
                "context_code": resolved_code,
                "vehicle_type": vehicle_type,
                "suggestions": CHATBOT_SUGGESTIONS,
                "legal_note": LEGAL_NOTE,
                "mode": "structured_law",
                "rag_status": status,
                "sources": [build_structured_penalty_source(penalty_record)],
                "structured_law_id": penalty_record.get("id"),
            })

        # ==========================================================
        # TẦNG 2: RAG + QWEN
        # Dùng để giải thích luật/câu hỏi mở hoặc trường hợp chưa có trong
        # penalties.json. Link nguồn chỉ để đối chiếu, không phải để Qwen
        # tự mở trình duyệt mỗi lần người dùng hỏi.
        # ==========================================================
        if status.get("ready") and RAG_MODULE_AVAILABLE:
            try:
                question_for_rag = build_enriched_rag_question(
                    question=question,
                    code=resolved_code,
                    vehicle_type=vehicle_type,
                )
                rag_result = rag_answer_question(
                    question=question_for_rag,
                    messages=history,
                    sign_context=sign_context,
                )
                answer = str(rag_result.get("answer", "")).strip()

                law_item = find_law_item(
                    normalize_vietnamese_text(question_for_rag),
                    resolved_code,
                )

                # Chỉ dùng fallback khi bảng dự phòng thực sự có con số cụ thể.
                # Không thay câu trả lời RAG bằng các câu “mức phạt phụ thuộc...”.
                if legal_answer_is_too_generic(answer) and law_item_has_specific_fine(law_item, vehicle_type):
                    fallback_answer = build_law_answer(law_item, vehicle_type)
                    return jsonify({
                        "answer": fallback_answer,
                        "context_code": resolved_code,
                        "vehicle_type": vehicle_type,
                        "suggestions": CHATBOT_SUGGESTIONS,
                        "legal_note": LEGAL_NOTE,
                        "mode": "knowledge",
                        "rag_status": status,
                        "sources": rag_result.get("sources", []),
                        "rewritten_query": rag_result.get("rewritten_query", question_for_rag),
                    })

                if answer:
                    return jsonify({
                        "answer": answer,
                        "context_code": resolved_code,
                        "vehicle_type": vehicle_type,
                        "suggestions": CHATBOT_SUGGESTIONS,
                        "legal_note": LEGAL_NOTE,
                        "mode": "rag",
                        "rag_status": status,
                        "sources": rag_result.get("sources", []),
                        "rewritten_query": rag_result.get("rewritten_query", question_for_rag),
                    })
            except Exception as rag_exc:
                print("[RAG] Lỗi trả lời, chuyển sang rule-based:", rag_exc)
                status = {**status, "error": str(rag_exc)}

        # Dự phòng khi model đang tải, thiếu model hoặc RAG gặp lỗi.
        fallback_answer, fallback_code = build_chatbot_answer(
            question=question,
            vehicle_type=vehicle_type,
            context_code=resolved_code,
        )
        return jsonify({
            "answer": fallback_answer,
            "context_code": fallback_code or resolved_code,
            "vehicle_type": vehicle_type,
            "suggestions": CHATBOT_SUGGESTIONS,
            "legal_note": LEGAL_NOTE,
            "mode": "rule_based",
            "rag_status": status,
            "sources": [],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/model_info")
@login_required
def model_info():
    return jsonify({
        "model_path": str(MODEL_PATH),
        "data_yaml": str(DATA_YAML_PATH),
        "num_classes": len(MODEL_NAMES),
        "names": MODEL_NAMES,
    })


if __name__ == "__main__":
    export_sign_knowledge_for_rag()

    if PENALTY_ENGINE_AVAILABLE:
        print("[LAW] penalty_engine + penalties.json: sẵn sàng ưu tiên tra mức phạt có cấu trúc.")
    else:
        print("[LAW] Chưa tải được penalty_engine, hệ thống vẫn chạy RAG bình thường:", PENALTY_IMPORT_ERROR)

    if RAG_MODULE_AVAILABLE:
        initialize_ai_background()
        print("[RAG] Đang khởi tạo chatbot pháp luật ở chế độ nền. Web vẫn mở bình thường.")
    else:
        print("[RAG] Không tải được module RAG, web dùng chatbot dự phòng:", RAG_IMPORT_ERROR)
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=5000)