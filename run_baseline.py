import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.chunking import ChunkingStrategyComparator
from pathlib import Path

files = ["phuc_loi_va_quyen_loi.md", "lam_them_ngoai_gio.md", "bat_dau_lam_viec.md"]
comp = ChunkingStrategyComparator()

for f in files:
    text = Path("data/" + f).read_text(encoding="utf-8")
    result = comp.compare(text, chunk_size=200)
    print(f"\n=== {f} ===")
    for name, stats in result.items():
        print(f"  {name}: count={stats['count']}, avg={round(stats['avg_length'])}, min={stats['min_length']}, max={stats['max_length']}")
