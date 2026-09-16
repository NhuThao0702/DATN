from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PENALTY_DATA_PATH = BASE_DIR / "penalties.json"

VEHICLE_LABELS = {
    "oto": "Ô tô",
    "xe_tai": "Xe tải",
    "xe_khach": "Xe khách",
    "xe_may": "Xe máy",
    "all": "Chưa xác định",
}

PHRASE_REPLACEMENTS = [
    (r"\bxe\s+hoi\b", " oto "),
    (r"\bxe\s+con\b", " oto "),
    (r"\bo\s*to\b", " oto "),
    (r"\bmo\s*to\b", " xe_may "),
    (r"\bxe\s+tay\s+ga\b", " xe_may "),
    (r"\bxe\s+gan\s+may\b", " xe_may "),
    (r"\bxe\s+may\b", " xe_may "),
    (r"\bdau\s+xe\b", " do xe "),
    (r"\blon\s+chieu\b", " nguoc chieu "),
    (r"\bnguoc\s+duong\b", " nguoc chieu "),
    (r"\bnham\s+lan\b", " sai lan "),
    (r"\blon\s+lan\b", " sai lan "),
    (r"\bdi\s+lon\s+lan\b", " sai lan "),
    (r"\bnham\s+phan\s+duong\b", " sai phan duong "),
    (r"\bdi\s+nham\s+phan\s+duong\b", " sai phan duong "),
    (r"\blan\s+lan\b", " sai lan "),
    (r"\bchay\s+den\s+do\b", " vuot den do "),
    (r"\bphat\s+may\s+diem\b", " tru diem "),
    (r"\btru\s+may\s+diem\b", " tru diem "),
    (r"\bgiu\s+bang\b", " gplx "),
    (r"\bgiam\s+bang\b", " gplx "),
    (r"\bu[\s-]?turn\b", " quay dau "),
]
SIGN_BEHAVIOR_MAP = {
    "P.102": "di_nguoc_chieu",
    "P.123A": "cam_re",
    "P.123B": "cam_re",
    "P.124A1": "cam_quay_dau",
    "P.124A2": "cam_quay_dau",
    "P.124B1": "cam_quay_dau",
    "P.124B2": "cam_quay_dau",
    "P.124C": "cam_quay_dau",
    "P.124D": "cam_quay_dau",
    "P.124E": "cam_quay_dau",
    "P.130": "cam_dung_do",
    "P.131A": "cam_do",
    "P.131B": "cam_do",
    "P.131C": "cam_do",
    "P.103A": "duong_cam",
    "P.106A": "duong_cam",
    "P.106B": "duong_cam",
    "P.111A": "duong_cam",
    "DEN_DO": "den_do",
}

SIGN_VEHICLE_HINTS = {
    "P.103A": "oto",
    "P.106A": "xe_tai",
    "P.106B": "xe_tai",
    "P.111A": "xe_may",
}

BEHAVIOR_PATTERNS = {
    "di_nguoc_chieu": [r"\bdi nguoc chieu\b", r"\bnguoc chieu\b", r"\bcam di nguoc\b"],
    "sai_lan": [r"\bsai lan\b", r"\bkhong dung lan\b", r"\bkhong dung phan duong\b", r"\bsai phan duong\b"],
    "cam_re": [r"\bcam re trai\b", r"\bre trai.*cam\b", r"\bcam re phai\b", r"\bre phai.*cam\b"],
    "cam_quay_dau": [r"\bcam quay dau\b", r"\bquay dau.*cam\b", r"\bkhong duoc quay dau\b"],
    "den_do": [r"\bvuot den do\b", r"\bden do.*di\b", r"\bkhong chap hanh.*den tin hieu\b"],
    "duong_cam": [r"\bdi vao duong cam\b", r"\bvao duong cam\b", r"\bkhu vuc cam\b", r"\bcam oto\b", r"\bcam xe tai\b", r"\bcam xe_may\b"],
    "qua_toc_do": [r"\bqua toc do\b", r"\bvuot toc do\b", r"\bchay qua.*km\b", r"\bchay nhanh hon.*quy dinh\b"],
    "cam_vuot": [r"\bcam vuot\b", r"\bvuot xe.*cam\b", r"\bvuot.*noi cam\b"],
    "dien_thoai": [r"\bdung dien thoai\b", r"\bbam dien thoai\b", r"\bcam.*dien thoai.*lai\b"],
    "mu_bao_hiem": [r"\bkhong doi mu bao hiem\b", r"\bkhong doi non bao hiem\b", r"\bmu bao hiem.*khong cai quai\b"],
    "dung_do_cao_toc": [r"\bdung.*cao toc\b", r"\bdo.*cao toc\b"],
    "vao_cao_toc": [r"\bxe_may.*cao toc\b", r"\bmo to.*cao toc\b"],
    "quay_dau_cao_toc": [r"\bquay dau.*cao toc\b"],
    "bien_bao_chung": [r"\bkhong chap hanh.*bien bao\b", r"\bvi pham.*bien bao\b", r"\bkhong tuan.*bien bao\b"],
}

PENALTY_HINTS = ("phat", "bao nhieu", "tru diem", "gplx", "bang lai", "vi pham", "xu phat", "can cu", "dieu", "khoan", "diem", "luat")


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[_/\\|,;:!?()\[\]{}\"']", " ", text)
    text = re.sub(r"km\s*h\b", "kmh", text)
    for pattern, replacement in PHRASE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _load_payload() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not PENALTY_DATA_PATH.exists():
        return {}, []
    try:
        raw = json.loads(PENALTY_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, []
    if isinstance(raw, list):
        return {}, raw
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        return raw.get("meta") or {}, raw["records"]
    return {}, []


def load_penalties() -> list[dict[str, Any]]:
    return _load_payload()[1]


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _is_active(record: dict[str, Any], reference_date: date) -> bool:
    start = _parse_date(record.get("effective_from"))
    end = _parse_date(record.get("effective_to"))
    return not ((start and reference_date < start) or (end and reference_date > end))


def infer_vehicle_type(question: str, fallback: str = "all") -> str:
    q = normalize_text(question)
    patterns = [
        ("xe_may", [r"\bxe_may\b", r"\bxe may\b", r"\bmo to\b"]),
        ("xe_tai", [r"\bxe tai\b"]),
        ("xe_khach", [r"\bxe khach\b", r"\bxe buyt\b"]),
        ("oto", [r"\boto\b", r"\bo to\b"]),
    ]
    for vehicle, regs in patterns:
        if any(re.search(reg, q) for reg in regs):
            return vehicle
    return fallback if fallback in VEHICLE_LABELS else "all"


def _normalize_sign_code(sign_code: str) -> str:
    code = str(sign_code or "").strip().upper().replace(" ", "")
    if code.startswith("DEN_"):
        return code
    if "." not in code and code and code[0] in "PWRIS":
        m = re.match(r"^([PWRIS])(\d+)([A-Z0-9]*)$", code)
        if m:
            code = f"{m.group(1)}.{m.group(2)}{m.group(3)}"
    return code


def detect_behavior(question: str, sign_code: str = "") -> str | None:
    q = normalize_text(question)
    for key in ("quay_dau_cao_toc", "dung_do_cao_toc", "vao_cao_toc"):
        if any(re.search(p, q) for p in BEHAVIOR_PATTERNS[key]):
            return key

    has_do = bool(re.search(r"\bdo\b|\bdo xe\b|\bdau\b", q)) and "quay dau" not in q
    has_dung = bool(re.search(r"\bdung\b|\bdung xe\b", q))
    has_cam = "cam" in q
    if has_cam and has_do and not has_dung:
        return "cam_do"
    if has_cam and has_dung and not has_do:
        return "cam_dung"
    if has_cam and has_do and has_dung:
        return "cam_dung_do"

    for key in ("di_nguoc_chieu", "sai_lan", "cam_quay_dau", "cam_re", "den_do", "duong_cam", "qua_toc_do", "cam_vuot", "dien_thoai", "mu_bao_hiem", "bien_bao_chung"):
        if any(re.search(p, q) for p in BEHAVIOR_PATTERNS[key]):
            return key

    code = _normalize_sign_code(sign_code)
    behavior = SIGN_BEHAVIOR_MAP.get(code)
    if behavior == "cam_dung_do":
        if has_do:
            return "cam_do"
        if has_dung:
            return "cam_dung"
    return behavior


def extract_speed_overage(question: str) -> float | None:
    q = normalize_text(question)
    for pattern in (
        r"(?:vuot|qua)\s+(?:toc do\s+)?(\d+(?:[.,]\d+)?)\s*kmh",
        r"(?:vuot|qua)\s+(\d+(?:[.,]\d+)?)\b",
    ):
        m = re.search(pattern, q)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                pass
    speeds = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*kmh", q)]
    if len(speeds) >= 2 and any(w in q for w in ("cho phep", "gioi han", "quy dinh")):
        if speeds[0] >= speeds[1]:
            return speeds[0] - speeds[1]
    return None


def _speed_matches(record: dict[str, Any], overage: float) -> bool:
    low = record.get("speed_over_min")
    high = record.get("speed_over_max")
    low_inc = record.get("speed_min_inclusive", True)
    high_inc = record.get("speed_max_inclusive", True)
    if low is not None:
        if low_inc and overage < float(low): return False
        if not low_inc and overage <= float(low): return False
    if high is not None:
        if high_inc and overage > float(high): return False
        if not high_inc and overage >= float(high): return False
    return True


def _vehicle_matches(record: dict[str, Any], vehicle_type: str) -> bool:
    return vehicle_type == "all" or vehicle_type in (record.get("vehicle_types") or [])


def _record_aliases(record: dict[str, Any]) -> list[str]:
    values = []
    values.extend(record.get("aliases") or [])
    values.append(record.get("title") or "")
    values.append(record.get("behavior") or "")
    # Cho phép fuzzy matcher nhận cả câu hỏi tra theo Điều/Khoản/Điểm.
    values.append(record.get("legal_basis") or "")
    values.extend(record.get("sign_codes") or [])
    return [normalize_text(v) for v in values if str(v or "").strip()]


def _token_similarity(a: str, b: str) -> float:
    A = {t for t in a.split() if len(t) > 1}
    B = {t for t in b.split() if len(t) > 1}
    if not A or not B:
        return 0.0
    inter = len(A & B)
    return max(inter / len(A | B), inter / len(B))


def _fuzzy_score(question: str, record: dict[str, Any]) -> float:
    q = normalize_text(question)
    best = 0.0
    for alias in _record_aliases(record):
        if alias in q:
            best = max(best, 1.0)
            continue
        seq = SequenceMatcher(None, q, alias).ratio()
        tok = _token_similarity(q, alias)
        best = max(best, seq * 0.45 + tok * 0.75)
    return best


def _combine_records(records: list[dict[str, Any]], title: str | None = None) -> dict[str, Any]:
    ordered = sorted(records, key=lambda r: (0 if any(v in {"oto", "xe_tai", "xe_khach"} for v in (r.get("vehicle_types") or [])) else 1, str(r.get("id", ""))))
    return {
        "id": "combined:" + ",".join(str(r.get("id")) for r in ordered),
        "title": title or "Kết quả xử phạt theo từng trường hợp",
        "behavior": ordered[0].get("behavior", "") if ordered else "",
        "vehicle_types": ["all"],
        "_combined_records": ordered,
        "source_title": "Nghị định 168/2024/NĐ-CP",
        "source_url": next((r.get("source_url") for r in ordered if r.get("source_url")), ""),
    }


def find_by_legal_reference(
    question: str,
    records: list[dict[str, Any]],
    vehicle_type: str = "all",
) -> dict[str, Any] | None:
    """
    Tra ngược dữ liệu theo căn cứ pháp lý, không phụ thuộc thứ tự người hỏi.

    Ví dụ đều hiểu như nhau:
    - "Điểm d khoản 3 Điều 7 quy định gì?"
    - "Điều 7 khoản 3 điểm d là lỗi gì?"
    - "Khoản 3 Điều 7 phạt thế nào?"
    """
    q = normalize_text(question)
    article_match = re.search(r"\bdieu\s*(\d+)\b", q)
    clause_match = re.search(r"\bkhoan\s*(\d+)\b", q)
    point_match = re.search(r"\bdiem\s*([a-z])\b", q)

    # Chỉ kích hoạt tra ngược khi câu hỏi thực sự có ít nhất Điều/Khoản/Điểm.
    if not (article_match or clause_match or point_match):
        return None

    article = article_match.group(1) if article_match else None
    clause = clause_match.group(1) if clause_match else None
    point = point_match.group(1) if point_match else None

    matched: list[dict[str, Any]] = []
    for record in records:
        if not _vehicle_matches(record, vehicle_type):
            continue

        basis = normalize_text(record.get("legal_basis", ""))
        if not basis:
            continue
        if article and not re.search(rf"\bdieu\s*{re.escape(article)}\b", basis):
            continue
        if clause and not re.search(rf"\bkhoan\s*{re.escape(clause)}\b", basis):
            continue
        if point and not re.search(rf"\bdiem\s*{re.escape(point)}\b", basis):
            continue
        matched.append(record)

    if not matched:
        return None

    # Ưu tiên record có căn cứ cụ thể nhất và độ tương đồng cao nhất.
    matched.sort(
        key=lambda r: (
            int(r.get("priority", 0)),
            _fuzzy_score(question, r),
        ),
        reverse=True,
    )

    if len(matched) == 1:
        return matched[0]

    # Nếu người dùng hỏi đủ Điểm + Khoản + Điều, thường chỉ cần record tốt nhất.
    # Tránh trả một danh sách dài nếu nhiều record vô tình cùng nhắc tới căn cứ đó.
    if article and clause and point:
        best = matched[0]
        best_score = _fuzzy_score(question, best)
        close = [r for r in matched if _fuzzy_score(question, r) >= best_score - 0.03]
        if len(close) == 1:
            return best
        matched = close

    title_parts = ["Quy định pháp lý"]
    if point:
        title_parts.append(f"điểm {point}")
    if clause:
        title_parts.append(f"khoản {clause}")
    if article:
        title_parts.append(f"Điều {article}")

    return _combine_records(matched[:8], " ".join(title_parts))


def lookup_penalty(question: str, vehicle_type: str = "all", sign_code: str = "", reference_date: date | None = None) -> dict[str, Any] | None:
    question = str(question or "").strip()
    if not question:
        return None
    reference_date = reference_date or date.today()
    records = [r for r in load_penalties() if _is_active(r, reference_date)]
    if not records:
        return None

    explicit_vehicle = infer_vehicle_type(question, "all")
    resolved_vehicle = explicit_vehicle if explicit_vehicle != "all" else vehicle_type
    if resolved_vehicle not in VEHICLE_LABELS:
        resolved_vehicle = "all"
    code = _normalize_sign_code(sign_code)
    if resolved_vehicle == "all" and code in SIGN_VEHICLE_HINTS:
        resolved_vehicle = SIGN_VEHICLE_HINTS[code]

    # Tầng ưu tiên 0: người dùng hỏi trực tiếp theo Điều/Khoản/Điểm.
    legal_reference_record = find_by_legal_reference(
        question=question,
        records=records,
        vehicle_type=resolved_vehicle,
    )
    if legal_reference_record:
        return legal_reference_record

    behavior = detect_behavior(question, sign_code)
    overage = extract_speed_overage(question)
    q_norm = normalize_text(question)
    if overage is not None and any(token in q_norm for token in ("vuot", "qua toc do", "chay")):
        behavior = "qua_toc_do"

    if behavior == "qua_toc_do":
        pool = [r for r in records if r.get("behavior_key") == "qua_toc_do" and _vehicle_matches(r, resolved_vehicle)]
        if overage is not None:
            matched = [r for r in pool if _speed_matches(r, overage)]
            if matched:
                return _combine_records(matched, f"Chạy quá tốc độ {overage:g} km/h") if resolved_vehicle == "all" and len(matched) > 1 else max(matched, key=lambda r: int(r.get("priority", 0)))
        if pool:
            return _combine_records(pool, "Mức phạt chạy quá tốc độ theo mức vượt")

    if behavior:
        behavior_keys = {behavior}
        if behavior == "cam_dung_do": behavior_keys.update({"cam_dung", "cam_do"})
        if behavior == "cam_re": behavior_keys.add("cam_re_quay_dau")
        if behavior == "cam_quay_dau": behavior_keys.add("cam_re_quay_dau")
        candidates = [r for r in records if r.get("behavior_key") in behavior_keys and _vehicle_matches(r, resolved_vehicle)]
        if code:
            direct = [r for r in candidates if code in {_normalize_sign_code(c) for c in (r.get("sign_codes") or [])}]
            if direct:
                candidates = direct
        if candidates:
            candidates.sort(key=lambda r: (int(r.get("priority", 0)), _fuzzy_score(question, r)), reverse=True)
            if behavior == "cam_dung_do" and len(candidates) > 1:
                return _combine_records(candidates, "Mức phạt dừng/đỗ tại biển cấm")
            if resolved_vehicle == "all":
                groups = {}
                for r in candidates:
                    group = "xe_may" if "xe_may" in (r.get("vehicle_types") or []) else "oto"
                    groups.setdefault(group, r)
                selected = list(groups.values())
                if len(selected) > 1:
                    return _combine_records(selected)
            return candidates[0]

    q = normalize_text(question)
    scored = [(_fuzzy_score(question, r), int(r.get("priority", 0)), r) for r in records if _vehicle_matches(r, resolved_vehicle)]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not scored or scored[0][0] < 0.48:
        return None
    if resolved_vehicle == "all":
        best_score = scored[0][0]
        close = [r for score, _, r in scored if score >= max(0.48, best_score - 0.08)]
        groups = {}
        for r in close:
            group = "xe_may" if "xe_may" in (r.get("vehicle_types") or []) else "oto"
            groups.setdefault(group, r)
        if len(groups) > 1:
            return _combine_records(list(groups.values()))
    return scored[0][2]


def _display_vehicle(record: dict[str, Any]) -> str:
    labels = []
    for item in record.get("vehicle_types") or []:
        label = VEHICLE_LABELS.get(item, str(item))
        if label not in labels:
            labels.append(label)
    return ", ".join(labels) or "Chưa xác định"


def format_penalty_answer(record: dict[str, Any]) -> str:
    if not record:
        return "Xin lỗi, tôi chưa tìm thấy quy định cụ thể trong cơ sở dữ liệu hiện tại."
    combined = record.get("_combined_records")
    if isinstance(combined, list) and combined:
        lines = [f"⚖️ **{record.get('title', 'Kết quả xử phạt')}**"]
        for item in combined:
            lines.append(
                f"• **{_display_vehicle(item)} - {item.get('behavior', '')}:** "
                f"Phạt {item.get('fine', 'chưa có dữ liệu')}; "
                f"{item.get('points', 'chưa có dữ liệu trừ điểm')}; "
                f"căn cứ {item.get('legal_basis', 'chưa xác định')}."
            )
        lines.append("ℹ️ **Lưu ý:** Nếu gây tai nạn hoặc có tình tiết đặc biệt, mức xử phạt có thể thuộc điều khoản khác.")
        return "\n".join(lines)

    lines = [
        f"📍 **Hành vi:** {record.get('behavior', 'Chưa xác định')}",
        f"🚗 **Phương tiện:** {_display_vehicle(record)}",
        f"💰 **Mức phạt:** {record.get('fine', 'Chưa có dữ liệu')}",
        f"🪪 **Trừ điểm GPLX:** {record.get('points', 'Chưa có dữ liệu')}",
        f"📜 **Căn cứ pháp lý:** {record.get('legal_basis', 'Chưa xác định')}",
    ]
    note = str(record.get("additional_penalty") or "").strip()
    if note and not note.lower().startswith("không có hình phạt bổ sung"):
        lines.append(f"📌 **Lưu ý:** {note}")
    lines.append("ℹ️ **Lưu ý:** Kết quả áp dụng cho trường hợp thông thường; nếu gây tai nạn hoặc có tình tiết đặc biệt cần đối chiếu điều khoản tương ứng.")
    return "\n".join(lines)


def debug_lookup(question: str, vehicle_type: str = "all", sign_code: str = "") -> str:
    record = lookup_penalty(question, vehicle_type=vehicle_type, sign_code=sign_code)
    return format_penalty_answer(record) if record else "NO_MATCH"