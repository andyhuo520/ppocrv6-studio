#!/usr/bin/env python3
"""Apple Vision (macOS) OCR on OmniDocBench 18 images via ocrmac. Local, offline."""
import sys, time, json
from pathlib import Path
from ocrmac import ocrmac

IMG_DIR = Path("OmniDocBench/demo_data/omnidocbench_demo/images")
OUT_DIR = Path("predictions/apple_vision"); OUT_DIR.mkdir(parents=True, exist_ok=True)
imgs = sorted(IMG_DIR.glob("*.jpg"))
results = []
for i, p in enumerate(imgs):
    t0 = time.perf_counter()
    # accurate level, prefer zh + en
    ann = ocrmac.OCR(str(p), recognition_level="accurate",
                     language_preference=["zh-Hans", "en-US"]).recognize()
    dt = (time.perf_counter() - t0) * 1000
    # ann: list of (text, confidence, bbox). bbox normalized (x,y,w,h), y from bottom
    lines = sorted(ann, key=lambda a: (-a[2][1], a[2][0]))  # top→bottom, left→right
    text = "\n".join(t for t, c, b in lines)
    (OUT_DIR / (p.stem + ".md")).write_text(text, encoding="utf-8")
    results.append({"name": p.name, "totalMs": round(dt, 1), "nLines": len(ann)})
    print(f"({i+1}/{len(imgs)}) {p.name[:40]} {dt:.0f}ms {len(ann)}lines")

valid = results[1:]  # drop warmup
tot = [r["totalMs"] for r in valid]
import statistics as st
summary = {"engine": "Apple Vision (macOS, ocrmac, accurate)", "results": results,
           "avg_ms": round(st.mean(tot), 1), "median_ms": round(st.median(tot), 1),
           "min_ms": min(tot), "max_ms": max(tot)}
json.dump(summary, open("predictions/apple_vision_results.json", "w"),
          ensure_ascii=False, indent=2)
print(f"\n端到端 平均{st.mean(tot):.0f}ms 中位{st.median(tot):.0f}ms 最快{min(tot):.0f} 最慢{max(tot):.0f}")
