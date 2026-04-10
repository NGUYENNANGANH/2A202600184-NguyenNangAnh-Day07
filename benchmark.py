"""
benchmark.py -- Chay 5 benchmark queries cua nhom voi strategy ca nhan cua ban.
Thay doi STRATEGY ben duoi de thu strategy khac nhau.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import SentenceChunker  # ← ĐỔI STRATEGY Ở ĐÂY

load_dotenv()

# ══════════════════════════════════════════════
# CẤU HÌNH — Sửa theo strategy của bạn
# ══════════════════════════════════════════════

# Lựa chọn strategy:
# from src.chunking import FixedSizeChunker;  chunker = FixedSizeChunker(chunk_size=300, overlap=50)
# from src.chunking import SentenceChunker;   chunker = SentenceChunker(max_sentences_per_chunk=3)
# from src.chunking import RecursiveChunker;  chunker = RecursiveChunker(chunk_size=400)

STRATEGY_NAME = "SentenceChunker(max_sentences=3)"  # ← ĐỔI TÊN THEO STRATEGY
chunker = SentenceChunker(max_sentences_per_chunk=3)

# Chỉ load file tài liệu nhóm (bỏ qua file thừa)
GROUP_FILES = [
    "bat_dau_lam_viec.md",
    "cach_lam_viec.md",
    "he_thong_noi_bo.md",
    "lam_them_ngoai_gio.md",
    "nghi_le_va_truyen_thong.md",
    "nghi_viec_va_tro_cap.md",
    "phat_trien_nghe_nghiep.md",
    "phuc_loi_va_quyen_loi.md",
    "quan_ly_thiet_bi.md",
]

# Metadata đã chốt với nhóm — category + topic cho từng file
FILE_METADATA = {
    "bat_dau_lam_viec":      {"category": "onboarding", "topic": "first_week"},
    "cach_lam_viec":         {"category": "culture",    "topic": "work_style"},
    "he_thong_noi_bo":       {"category": "technical",  "topic": "internal_systems"},
    "lam_them_ngoai_gio":    {"category": "policy",     "topic": "moonlighting"},
    "nghi_le_va_truyen_thong": {"category": "benefits", "topic": "holidays"},
    "nghi_viec_va_tro_cap":  {"category": "policy",     "topic": "resignation"},
    "phat_trien_nghe_nghiep":{"category": "career",     "topic": "growth_salary"},
    "phuc_loi_va_quyen_loi": {"category": "benefits",   "topic": "pto_insurance"},
    "quan_ly_thiet_bi":      {"category": "technical",  "topic": "device_management"},
}

# ══════════════════════════════════════════════
# 5 BENCHMARK QUERIES CỦA NHÓM
# ══════════════════════════════════════════════

QUERIES = [
    {
        "id": "Q1",
        "query": "Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?",
        "gold": "20 ngày nghỉ phép + 11 ngày lễ. Tối đa tích lũy 27 ngày.",
    },
    {
        "id": "Q2",
        "query": "Công ty có chính sách gì về làm thêm ngoài giờ?",
        "gold": "Cho phép công việc phụ thỉnh thoảng, diễn thuyết, kinh doanh phụ vài giờ/tuần. Không được làm cho đối thủ.",
    },
    {
        "id": "Q3",
        "query": "Nhân viên mới cần gặp ai trong tuần đầu tiên?",
        "gold": "Quản lý, nhóm, buddy 37signals, và People Ops (Andrea).",
    },
    {
        "id": "Q4",
        "query": "Mức lương tối thiểu và cách tính lương tại công ty?",
        "gold": "Lương tối thiểu $73,500. Top 10% San Francisco. Cùng role cùng level trả như nhau.",
    },
    {
        "id": "Q5",
        "query": "Công ty dùng hệ thống nào để theo dõi lỗi lập trình?",
        "gold": "Sentry theo dõi lỗi. Grafana giám sát hệ thống. Dash cho logging.",
    },
]

# ══════════════════════════════════════════════
# CHẠY BENCHMARK
# ══════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  BENCHMARK — Strategy: {STRATEGY_NAME}")
    print(f"{'='*60}")

    # Bước 1: Load & chunk tài liệu nhóm
    docs = []
    data_dir = Path("data")
    for filename in GROUP_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  [!] Không tìm thấy file: {filename}")
            continue
        text = filepath.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            meta = FILE_METADATA.get(filepath.stem, {})
            docs.append(Document(
                id=f"{filepath.stem}_chunk_{i}",
                content=chunk,
                metadata={
                    "source": filepath.stem,
                    "chunk_index": i,
                    "category": meta.get("category", ""),
                    "topic": meta.get("topic", ""),
                },
            ))

    # Bước 2: Nạp vào store — dùng OpenAI embedder thật
    from src.embeddings import OpenAIEmbedder
    embedder = OpenAIEmbedder()
    store = EmbeddingStore(embedding_fn=embedder)
    store.add_documents(docs)
    print(f"\n  [OK] Da nap {store.get_collection_size()} chunks tu {len(GROUP_FILES)} tai lieu\n")

    # Bước 3: Chạy 5 queries
    score_total = 0
    for q in QUERIES:
        print(f"\n  {q['id']}: {q['query']}")
        print(f"  Gold answer: {q['gold']}")
        print(f"  {'-'*50}")

        results = store.search(q["query"], top_k=3)
        for i, r in enumerate(results, 1):
            preview = r["content"][:120].replace("\n", " ")
            print(f"  [{i}] score={r['score']:.4f} | src={r['metadata'].get('source', '?')} | {preview}...")

        # Tự đánh giá (điền thủ công sau khi chạy)
        print(f"\n  >> Tự đánh giá: Top-3 có chứa thông tin liên quan không? (Y/N)")

    print(f"\n{'='*60}")
    print(f"  Kết quả đã chạy xong. Điền vào REPORT.md Section 6.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
