"""
07. YOLO 추가 학습(fine-tune)용: RedTell dse 학습 영상 + 정답 만들기

왜 필요한가
  "가상 염색 영상으로 YOLO를 추가 학습하면 더 잘 되나?" 실험(B)에 쓸 학습 데이터.
  dse 1~83번은 원래 가상 염색 모델의 학습용(정답은 안 썼음)이라, 정답 박스를 만들어 YOLO 추가 학습에 씁니다.
  val(84~93)·test(94~186)는 건드리지 않습니다.

하는 일
  dse 1~83번 (65번은 정답 마스크 오류로 제외 → 82장)을 4_eval_unstained 와 똑같이 전처리
  (밝기 보정 + RBC 64 px 배율) → 6_redtell_dse_train/{images, labels, masks}, counts.csv

사용법
  python 07_make_dse_train.py --prep "D:/05. Blood/04. Prep" --dse_zip "D:/05. Blood/03. Data/6. RedTell/dse_data.zip"
"""
import argparse
import csv
import importlib.util
import os
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image
from skimage import measure

TRAIN_IDS = [i for i in range(1, 84) if i != 65]


def load_prep_module(prep):
    spec = importlib.util.spec_from_file_location("prep03", os.path.join(prep, "code", "03_prep_data.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dse_zip")
    g.add_argument("--dse")
    a = ap.parse_args()

    P = load_prep_module(a.prep)
    k = P.TARGET_RBC_PX / P.REDTELL_RBC_PX
    D = os.path.join(a.prep, "6_redtell_dse_train")
    if os.path.exists(os.path.join(D, "counts.csv")):
        print("이미 있습니다:", D)
        return
    tmp = tempfile.mkdtemp()
    try:
        if a.dse_zip:
            with zipfile.ZipFile(a.dse_zip) as z:
                for i in TRAIN_IDS:
                    for sub in ["images", "annotations"]:
                        z.extract(f"dse_data/{sub}/{i}.tif", tmp)
            dse = os.path.join(tmp, "dse_data")
        else:
            dse = a.dse
        for d in ["images", "labels", "masks"]:
            os.makedirs(os.path.join(D, d), exist_ok=True)
        rows = []
        for i in TRAIN_IDS:
            img = P.unstained_prep(P.read_dse_image(dse, i), k)
            h, w = img.shape[:2]
            P.imwrite(os.path.join(D, "images", f"dse_{i}.png"), img)
            lab = np.array(Image.open(os.path.join(dse, "annotations", f"{i}.tif")))
            lab = cv2.resize(lab.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
            P.imwrite(os.path.join(D, "masks", f"dse_{i}.png"), lab)
            props = measure.regionprops(lab)
            with open(os.path.join(D, "labels", f"dse_{i}.txt"), "w") as f:
                for p in props:
                    y0, x0, y1, x1 = p.bbox
                    f.write(f"1 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
            rows.append({"image": f"dse_{i}.png", "n_RBC": len(props), "n_WBC": 0})
        P.write_csv(os.path.join(D, "counts.csv"), rows)
        print(f"[학습용] {D} : 영상 {len(rows)}장, 정답 RBC {sum(r['n_RBC'] for r in rows)}개")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
