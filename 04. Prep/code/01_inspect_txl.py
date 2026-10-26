"""
01. TXL-PBC 데이터 확인
- 출처(BCCD/BCDD/PBC/Raabin-WBC)별 학습/검증/테스트 이미지 수
- 출처별 이미지 크기
- 출처별 클래스(WBC/RBC/혈소판) 박스 수와 박스 크기 중앙값(px)

사용법: python 01_inspect_txl.py <TXL-PBC 폴더>
"""
import collections
import glob
import os
import statistics as st
import sys

import openpyxl
from PIL import Image

r = sys.argv[1]
ws = openpyxl.load_workbook(os.path.join(r, "metadata_file.xlsx"), read_only=True).active
src = {row[2]: row[1] for row in list(ws.iter_rows(values_only=True))[1:] if row[2]}

cnt = collections.Counter()
sizes = collections.defaultdict(collections.Counter)
box = collections.defaultdict(lambda: collections.defaultdict(list))
for sp in ["train", "val", "test"]:
    for im in glob.glob(f"{r}/images/{sp}/*.png"):
        n = os.path.basename(im)
        s = src.get(n, "?")
        cnt[(sp, s)] += 1
        W, H = Image.open(im).size
        sizes[s][(W, H)] += 1
        for line in open(f"{r}/labels/{sp}/{n[:-4]}.txt"):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            box[s][c].append((float(p[3]) * W + float(p[4]) * H) / 2)

names = {0: "WBC", 1: "RBC", 2: "PLT"}
srcs = sorted({s for _, s in cnt})
print("[출처별 train/val/test]")
for s in srcs:
    v = [cnt[(sp, s)] for sp in ["train", "val", "test"]]
    print(f"  {s}: {v} (총 {sum(v)})")
print("[출처별 이미지 크기]")
for s in srcs:
    print(f"  {s}: {sizes[s].most_common(3)}")
print("[출처별 박스 수, 크기 중앙값 px]")
for s in srcs:
    print(f"  {s}:", {names[c]: (len(v), round(st.median(v), 1)) for c, v in sorted(box[s].items())})
