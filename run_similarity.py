import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.chunking import compute_similarity
from src.embeddings import OpenAIEmbedder
from dotenv import load_dotenv
load_dotenv()

embedder = OpenAIEmbedder()

pairs = [
    ("Nhân viên được nghỉ 20 ngày phép mỗi năm.",
     "Mỗi năm công ty cấp cho nhân viên 20 ngày nghỉ phép có lương.",
     "HIGH - cùng ý nghĩa, khác từ"),
    ("Nhân viên được nghỉ 20 ngày phép mỗi năm.",
     "Công ty dùng Sentry để theo dõi lỗi lập trình.",
     "LOW - hai chủ đề hoàn toàn khác"),
    ("Buddy 37signals sẽ hướng dẫn nhân viên mới.",
     "Nhân viên mới cần gặp quản lý và nhóm trong tuần đầu tiên.",
     "HIGH - cùng chủ đề onboarding"),
    ("Không được làm việc cho đối thủ cạnh tranh.",
     "Có thể kinh doanh phụ vài giờ mỗi tuần nếu không ảnh hưởng đến công việc chính.",
     "MEDIUM - cùng topic làm thêm nhưng đối lập"),
    ("Mức lương tối thiểu là $73,500 mỗi năm.",
     "Trời hôm nay rất đẹp và nắng.",
     "LOW - hoàn toàn không liên quan"),
]

print("\n=== SIMILARITY PREDICTIONS ===\n")
for i, (a, b, predict) in enumerate(pairs, 1):
    va = embedder(a)
    vb = embedder(b)
    score = compute_similarity(va, vb)
    print(f"Pair {i}: {predict}")
    print(f"  A: {a[:60]}...")
    print(f"  B: {b[:60]}...")
    print(f"  Score: {score:.4f}\n")
