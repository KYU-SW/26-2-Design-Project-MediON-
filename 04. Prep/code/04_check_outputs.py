"""
04. 전처리 결과 확인 (v3)
- 각 결과 폴더의 개수 출력
- 미리보기 이미지 1장 저장
    1행: TXL-PBC 출처별 학습 영상 + 라벨 (빨강 WBC, 초록 RBC)
    2행: RedTell 평가 영상 + 정답 박스 2장, 가상 염색 RBC 전용 패치 (비염색 trainA 1장, 염색 trainB 1장)

사용법: python 04_check_outputs.py <03_prep_data.py 출력 폴더> [미리보기 저장 경로]
"""
import collections
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np

def imread(path, flags=cv2.IMREAD_COLOR):
    """한글 등 비ASCII 경로에서도 읽히는 imread (윈도우 cv2.imread는 한글 경로를 못 읽음)"""
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img):
    """한글 경로에서도 저장되는 imwrite"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok


p = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "preview.png"
random.seed(1)

print("[A] 검출 학습 (박스 수)")
for sp in ["train", "val", "test"]:
    c = collections.Counter()
    for f in glob.glob(f"{p}/1_detection_train/labels/{sp}/*.txt"):
        for l in open(f):
            c[l.split()[0]] += 1
    print(f"  {sp}: 영상 {len(glob.glob(f'{p}/1_detection_train/images/{sp}/*.png'))}, WBC {c['0']}, RBC {c['1']}")
print("[B] 세포 크롭", {sp: {k: len(glob.glob(f"{p}/2_cell_crops/{sp}/{k}/*.png")) for k in ["RBC", "WBC"]} for sp in ["train", "val", "test"]})
for sub in ["rbc_only", "all"]:
    print(f"[C] 가상 염색 {sub}", {d: len(glob.glob(f"{p}/3_virtual_staining/{sub}/{d}/*.png")) for d in ["trainA", "trainB", "testA", "testB"]})
cnt = list(csv.DictReader(open(f"{p}/4_eval_unstained/redtell_dse/counts.csv", encoding="utf-8-sig")))
print(f"[D] 평가 영상 {len(cnt)}장, 정답 RBC {sum(int(r['n_RBC']) for r in cnt)}개")


def draw(im, lab_path, tile=420):
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]
    for l in open(lab_path):
        c, x, y, bw, bh = l.split()
        x, y, bw, bh = map(float, (x, y, bw, bh))
        cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h)),
                      (0, 0, 255) if c == "0" else (0, 200, 0), 2)
    t = np.full((tile, tile, 3), 255, np.uint8)
    t[:min(h, tile), :min(w, tile)] = im[:tile, :tile]
    return t


rows = list(csv.DictReader(open(f"{p}/meta/txl_images.csv", encoding="utf-8-sig")))
row1 = []
for s in ["BCCD", "BCDD", "PBC", "Raabin-WBC"]:
    r = [x for x in rows if x["source"] == s and x["split"] == "train" and int(x["n_WBC"]) > 0][0]
    t = draw(imread(f"{p}/1_detection_train/images/train/{r['image']}"), f"{p}/1_detection_train/labels/train/{r['image'][:-4]}.txt")
    cv2.putText(t, s, (5, 20), 0, 0.6, (0, 0, 0), 2)
    row1.append(t)
row2 = []
for f in sorted(glob.glob(f"{p}/4_eval_unstained/redtell_dse/images/*.png"))[:2]:
    stem = f.replace("\\", "/").split("/")[-1][:-4]
    t = draw(imread(f, cv2.IMREAD_GRAYSCALE), f"{p}/4_eval_unstained/redtell_dse/labels/{stem}.txt")
    cv2.putText(t, "eval " + stem, (5, 20), 0, 0.6, (0, 0, 255), 2)
    row2.append(t)
for d in ["trainA", "trainB"]:
    im = imread(random.choice(glob.glob(f"{p}/3_virtual_staining/rbc_only/{d}/*.png")))
    t = np.full((420, 420, 3), 255, np.uint8)
    t[:256, :256] = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, "rbc_only " + d + " (x2)", (5, 280), 0, 0.6, (0, 0, 0), 2)
    row2.append(t)
imwrite(out, np.vstack([np.hstack(row1), np.hstack(row2)]))
print("저장:", out)
