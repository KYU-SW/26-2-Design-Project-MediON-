"""
20. 가상 염색용 검증(val) 분할 만들기  (처음 한 번만)

왜 필요한가
  가상 염색 모델은 "몇 번째 epoch 가중치를 쓸지" 골라야 합니다.
  그런데 4_eval_unstained (dse 94~186번)는 최종 채점(test)용이라, 고르는 데 쓰면 안 됩니다.
  그래서 학습용이던 dse 84~93번(10장)을 학습에서 빼서 검증용으로 씁니다.

하는 일
  1) 3_virtual_staining/{rbc_only,all}/trainA 에서 dse_84~93 조각을 꺼내
     3_virtual_staining/_holdout_val/ 로 "이동" (삭제 아님, 되돌릴 수 있음)
  2) 4_eval_unstained/redtell_dse_val/ 생성 (images, labels, masks, counts.csv)
     - 4_eval_unstained/redtell_dse 와 같은 전처리 (밝기 보정 + 배율 정규화)
  3) 정답 마스크가 망가진 영상 제외 (아래 BAD_GT 참고)
     - 최종 채점셋 redtell_dse 에서 97, 163번을 _excluded_bad_gt/ 로 이동, counts.csv 갱신
     - val 에서는 88번 제외 → val 9장
  여러 번 실행해도 결과는 같습니다.

사용법
  python 20_make_staining_val.py --prep "D:/05. Blood/04. Prep" --dse_zip "D:/05. Blood/03. Data/6. RedTell/dse_data.zip"
  (dse_data.zip 대신 압축 푼 dse_data 폴더를 --dse 로 줘도 됩니다)
"""
import argparse
import glob
import importlib.util
import os
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image
from skimage import measure

# 정답 마스크가 망가진 dse 영상 (영상엔 세포 약 50개인데 마스크엔 255개가 겹쳐 찍혀 있고 영상과 안 맞음)
#   65  : 가상 염색 학습용(trainA) — 정답을 안 쓰므로 영향 없음
#   88  : val 에서 제외
#   97, 163 : 최종 채점(redtell_dse)에서 제외
BAD_GT = {65, 88, 97, 163}
VAL_IDS = [i for i in range(84, 94) if i not in BAD_GT]  # dse 84~93번 중 9장 (88 제외)
TEST_EXCLUDE = [97, 163]


def load_prep_module(prep):
    """04. Prep/code/03_prep_data.py 의 전처리 함수를 그대로 가져와 씀 (전처리 방식을 똑같이 맞추기 위해)"""
    path = os.path.join(prep, "code", "03_prep_data.py")
    spec = importlib.util.spec_from_file_location("prep03", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def extract_needed(zip_path, ids, tmp):
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for i in ids:
            for sub in ["images", "annotations"]:
                n = f"dse_data/{sub}/{i}.tif"
                if n not in names:
                    raise FileNotFoundError(f"zip 안에 {n} 가 없습니다")
                z.extract(n, tmp)
    return os.path.join(tmp, "dse_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="04. Prep 폴더")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dse_zip", help="dse_data.zip 경로")
    g.add_argument("--dse", help="압축 푼 dse_data 폴더 경로")
    a = ap.parse_args()

    P = load_prep_module(a.prep)
    k = P.TARGET_RBC_PX / P.REDTELL_RBC_PX

    # 1) trainA 에서 val 조각 빼기
    vs = os.path.join(a.prep, "3_virtual_staining")
    moved = {}
    for sub in ["rbc_only", "all"]:
        src = os.path.join(vs, sub, "trainA")
        dst = os.path.join(vs, "_holdout_val", sub, "trainA")
        os.makedirs(dst, exist_ok=True)
        n = 0
        for i in VAL_IDS:
            for f in glob.glob(os.path.join(src, f"dse_{i}_*.png")):
                shutil.move(f, os.path.join(dst, os.path.basename(f)))
                n += 1
        moved[sub] = n
        left = len(glob.glob(os.path.join(src, "*.png")))
        print(f"[{sub}] trainA 에서 {n}개 이동 → _holdout_val (남은 trainA {left}개)")

    # 1-2) 최종 채점셋에서 정답 오류 영상 빼기 (이동, 삭제 아님)
    T = os.path.join(a.prep, "4_eval_unstained", "redtell_dse")
    for i in TEST_EXCLUDE:
        for d, ext in [("images", "png"), ("labels", "txt"), ("masks", "png")]:
            f = os.path.join(T, d, f"dse_{i}.{ext}")
            if os.path.exists(f):
                dst = os.path.join(T, "_excluded_bad_gt", d)
                os.makedirs(dst, exist_ok=True)
                shutil.move(f, os.path.join(dst, os.path.basename(f)))
    cpath = os.path.join(T, "counts.csv")
    if os.path.exists(cpath):
        import csv
        with open(cpath, encoding="utf-8-sig") as f:
            rows_t = [r for r in csv.DictReader(f) if r["image"] not in {f"dse_{i}.png" for i in TEST_EXCLUDE}]
        P.write_csv(cpath, rows_t)
        n_img = len(glob.glob(os.path.join(T, "images", "*.png")))
        print(f"[test] redtell_dse 에서 {TEST_EXCLUDE}번 제외 → 영상 {n_img}장, 정답 RBC {sum(int(r['n_RBC']) for r in rows_t)}개")

    # 2) 검증용 실제 비염색 영상 + 정답
    tmp = tempfile.mkdtemp()
    try:
        dse = extract_needed(a.dse_zip, VAL_IDS, tmp) if a.dse_zip else a.dse
        D = os.path.join(a.prep, "4_eval_unstained", "redtell_dse_val")
        for d in ["images", "labels", "masks"]:
            os.makedirs(os.path.join(D, d), exist_ok=True)
        rows = []
        for i in VAL_IDS:
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
        with open(os.path.join(D, "README.txt"), "w", encoding="utf-8") as f:
            f.write("RedTell dse 84~93번 (10장). 가상 염색 모델의 epoch 고르기용 검증 데이터.\n"
                    "이 영상의 조각은 가상 염색 학습(trainA)에서 빠져 있습니다 (3_virtual_staining/_holdout_val).\n"
                    "최종 채점은 redtell_dse (94~186번)로만 합니다.\n")
        print(f"[val] {D} : 영상 {len(rows)}장, 정답 RBC {sum(r['n_RBC'] for r in rows)}개")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
