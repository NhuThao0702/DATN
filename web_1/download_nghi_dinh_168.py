from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import requests
import trafilatura


URL = (
    "https://xaydungchinhsach.chinhphu.vn/"
    "toan-van-nghi-dinh-168-2024-nd-cp-quy-dinh-xu-phat-"
    "vi-pham-hanh-chinh-ve-trat-tu-atgt-duong-bo-119241231164556785.htm"
)

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = (
    BASE_DIR
    / "legal_ai"
    / "data"
    / "nghi_dinh_168_2024_toan_van.txt"
)


def clean_text(text: str) -> str:
    """Chuẩn hóa nội dung HTML đã trích xuất thành văn bản sạch."""
    text = str(text or "").replace("\xa0", " ")

    lines: list[str] = []

    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if line:
            lines.append(line)

    cleaned = "\n".join(lines)

    # Cắt bớt phần menu phía trên, bắt đầu từ tiêu đề Nghị định nếu tìm thấy.
    match = re.search(
        r"\bNGHỊ\s+ĐỊNH\b",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match:
        cleaned = cleaned[match.start():]

    return cleaned.strip()


def validate_text(cleaned_text: str) -> None:
    """Kiểm tra nội dung lấy về có phải toàn văn nghị định hay không."""
    validation_checks = {
        "phần Nghị định": re.search(
            r"\bNGHỊ\s+ĐỊNH\b",
            cleaned_text,
            flags=re.IGNORECASE,
        ),
        "Điều 1": re.search(
            r"\bĐiều\s+1\b",
            cleaned_text,
            flags=re.IGNORECASE,
        ),
        "Điều 6": re.search(
            r"\bĐiều\s+6\b",
            cleaned_text,
            flags=re.IGNORECASE,
        ),
        "nội dung phạt tiền": re.search(
            r"\bPhạt\s+tiền\b",
            cleaned_text,
            flags=re.IGNORECASE,
        ),
    }

    missing = [
        name
        for name, result in validation_checks.items()
        if result is None
    ]

    if not missing:
        return

    debug_path = (
        BASE_DIR
        / "legal_ai"
        / "data"
        / "debug_nghi_dinh_168_extract.txt"
    )

    debug_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_path.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    raise RuntimeError(
        "Nội dung tải về chưa đầy đủ, thiếu: "
        + ", ".join(missing)
        + "\nĐã lưu nội dung kiểm tra tại: "
        + str(debug_path)
    )


def main() -> None:
    print("Đang tải toàn văn Nghị định 168 từ Chinhphu.vn...")

    response = requests.get(
        URL,
        timeout=90,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/137.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    # Giúp requests nhận đúng bảng mã tiếng Việt.
    if not response.encoding:
        response.encoding = response.apparent_encoding

    extracted_text = trafilatura.extract(
        response.text,
        output_format="txt",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )

    if not extracted_text:
        raise RuntimeError(
            "Không lấy được nội dung chữ từ trang Chinhphu.vn."
        )

    # Biến này phải được tạo trước khi kiểm tra.
    cleaned_text = clean_text(extracted_text)

    if len(cleaned_text) < 10_000:
        raise RuntimeError(
            "Nội dung tải về quá ngắn, có thể website đã chặn "
            "hoặc thay đổi cấu trúc HTML."
        )

    validate_text(cleaned_text)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = (
        "NGHỊ ĐỊNH SỐ 168/2024/NĐ-CP\n"
        "Nguồn chính thức: Cổng Thông tin điện tử Chính phủ Việt Nam\n"
        f"URL nguồn: {URL}\n"
        f"Ngày tải dữ liệu: {datetime.now():%d/%m/%Y %H:%M}\n"
        "Loại dữ liệu: Toàn văn dạng chữ dùng cho hệ thống RAG\n"
        "\n"
        "============================================================\n\n"
    )

    OUTPUT_PATH.write_text(
        header + cleaned_text,
        encoding="utf-8",
    )

    print("\nĐã lưu thành công:")
    print(OUTPUT_PATH)
    print(f"Số ký tự nội dung: {len(cleaned_text):,}")


if __name__ == "__main__":
    main()