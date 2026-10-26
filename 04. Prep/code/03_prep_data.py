"""
03. 전처리 본체 (v3)
총괄 정리 문서 "3. 학습 체계" 기준으로 학습·평가에 바로 쓸 데이터를 만듭니다. (학습은 하지 않습니다)

입력
  --txl      TXL-PBC 폴더 (images/, labels/, metadata_file.xlsx)
  --anemia   RedTell anemia_data 폴더 (control/, scd/, thalassemia/ 아래 images/)
  --dse      RedTell dse_data 폴더 (images/, annotations/)
  --out      출력 폴더

출력 (--out 아래)
  1_detection_train/   검출·계수 모델 학습용. YOLO 형식, WBC(0)/RBC(1), 배율 정규화
  2_cell_crops/        셀 기반 방식의 RBC/WBC 분류기 학습용 세포 크롭
  3_virtual_staining/  가상 염색 모델 학습용 패치 (CycleGAN/UNSB 폴더 형식)
     rbc_only/         128x128, WBC 없는 염색 패치만 (현재 데이터로 추천)
     all/              256x256, 모든 염색 패치
  4_eval_unstained/    최종 평가(평가 ②)용 실제 비염색 영상 + 정답 (RedTell dse 평가 분할)
  meta/                배율 보정 비율, 이미지별 출처·분할·용도 목록

사용법
  python 03_prep_data.py --txl <...> --anemia <...> --dse <...> --out <...>
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
from statistics import median

import cv2
import numpy as np
import openpyxl
from PIL import Image
from skimage import measure

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


# ---------------- 설정 ----------------
TARGET_RBC_PX = 64      # 정규화 후 RBC 크기(박스 (가로+세로)/2, px)
REDTELL_RBC_PX = 56.5   # RedTell RBC 크기. dse 마스크 박스 중앙값(경계에 걸린 세포 제외), 02번 스크립트로 측정
DSE_GAN_IDS = range(1, 94)     # dse 1~93번: 가상 염색 학습용 (trainA)
DSE_EVAL_IDS = range(94, 187)  # dse 94~186번: 평가용 (testA + 평가 ② 정답)
PATCH_ALL, STRIDE_ALL = 256, 128
PATCH_RBC, STRIDE_RBC = 128, 64     # 염색 RBC 전용: WBC 없는 영역을 최대한 모으려고 겹쳐서 자름
STRIDE_A_RBC = 128                   # 비염색 trainA: 겹치지 않게 (비염색 영상이 충분히 많음)
CROP_PAD, CROP_SIZE = 0.15, 96
KEEP = {0: 0, 1: 1}     # TXL 클래스 -> 새 클래스 (0 WBC, 1 RBC). 2(혈소판) 제외
NAMES = ["WBC", "RBC"]


# ---------------- 공통 함수 ----------------
def read_labels(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 5:
                out.append((int(p[0]), *map(float, p[1:5])))
    return out


def flat_field(gray):
    """비염색 명시야 영상의 불균일한 배경 밝기 보정 -> uint8"""
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 40)
    n = g / (bg + 1e-6)
    lo, hi = np.percentile(n, [0.5, 99.5])
    return np.clip((n - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def grid_patches(img, size, stride):
    h, w = img.shape[:2]
    if h < size or w < size:
        img = cv2.copyMakeBorder(img, 0, max(0, size - h), 0, max(0, size - w), cv2.BORDER_REFLECT)
        h, w = img.shape[:2]
    ys = list(range(0, h - size + 1, stride)) or [0]
    xs = list(range(0, w - size + 1, stride)) or [0]
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)
    for y in ys:
        for x in xs:
            yield y, x, img[y:y + size, x:x + size]


def unstained_prep(gray, k):
    """비염색 영상 공통 전처리: 밝기 보정 -> 배율 정규화.
    흑백 1채널로 저장 (용량 1/3). CycleGAN/UNSB 데이터 로더와 YOLO는 읽을 때 3채널(RGB)로 자동 변환함"""
    g = flat_field(gray)
    if k != 1:
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return g


def has_cells(p, thr=0.08):
    """비염색 패치에 세포 경계가 충분히 있는지 확인 (빈 배경 패치 제거용).
    dse 정답 마스크로 맞춘 기준: 128 패치에서 빈 패치 약 79% 제거, 세포 있는 패치 손실 0.5% 미만"""
    b = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 2)
    gm = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
    return (gm > 40).mean() >= thr


def save_patches(img, folder, stem, size, stride, drop_empty=False):
    n = 0
    for y, x, p in grid_patches(img, size, stride):
        if drop_empty and not has_cells(p):
            continue
        imwrite(f"{folder}/{stem}_{y}_{x}.png", p)
        n += 1
    return n


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# ---------------- A, B: 염색 TXL-PBC ----------------
def prep_txl(txl, out):
    ws = openpyxl.load_workbook(os.path.join(txl, "metadata_file.xlsx"), read_only=True).active
    src = {r[2]: r[1] for r in list(ws.iter_rows(values_only=True))[1:] if r and r[2]}

    sizes = defaultdict(list)
    for sp in ["train", "val", "test"]:
        for im in glob.glob(f"{txl}/images/{sp}/*.png"):
            n = os.path.basename(im)
            h, w = imread(im).shape[:2]
            for c, _, _, bw, bh in read_labels(f"{txl}/labels/{sp}/{n[:-4]}.txt"):
                if c == 1:
                    sizes[src[n]].append((bw * w + bh * h) / 2)
    med = {s: median(v) for s, v in sizes.items()}
    scale = {s: TARGET_RBC_PX / m for s, m in med.items()}

    A, B = f"{out}/1_detection_train", f"{out}/2_cell_crops"
    rows = []
    for sp in ["train", "val", "test"]:
        os.makedirs(f"{A}/images/{sp}", exist_ok=True)
        os.makedirs(f"{A}/labels/{sp}", exist_ok=True)
        for c in NAMES:
            os.makedirs(f"{B}/{sp}/{c}", exist_ok=True)
        for im_path in sorted(glob.glob(f"{txl}/images/{sp}/*.png")):
            n = os.path.basename(im_path)
            s, img = src[n], imread(im_path)
            h, w = img.shape[:2]
            k = scale[s]
            nw, nh = round(w * k), round(h * k)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
            imwrite(f"{A}/images/{sp}/{n}", img)
            labs = [(KEEP[c], x, y, bw, bh) for c, x, y, bw, bh in read_labels(f"{txl}/labels/{sp}/{n[:-4]}.txt") if c in KEEP]
            with open(f"{A}/labels/{sp}/{n[:-4]}.txt", "w") as f:  # YOLO 정규화 좌표는 리사이즈해도 그대로
                for c, x, y, bw, bh in labs:
                    f.write(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
            for i, (c, x, y, bw, bh) in enumerate(labs):
                cx, cy = x * nw, y * nh
                ww, hh = bw * nw * (1 + 2 * CROP_PAD), bh * nh * (1 + 2 * CROP_PAD)
                x0, y0, x1, y1 = int(max(0, cx - ww / 2)), int(max(0, cy - hh / 2)), int(min(nw, cx + ww / 2)), int(min(nh, cy + hh / 2))
                if x1 - x0 >= 4 and y1 - y0 >= 4:
                    imwrite(f"{B}/{sp}/{NAMES[c]}/{n[:-4]}_{i:03d}.png", cv2.resize(img[y0:y1, x0:x1], (CROP_SIZE, CROP_SIZE)))
            rows.append({"image": n, "source": s, "split": sp, "scale": round(k, 4), "new_w": nw, "new_h": nh,
                         "n_WBC": sum(l[0] == 0 for l in labs), "n_RBC": sum(l[0] == 1 for l in labs)})

    with open(f"{A}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(A)}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames: {NAMES}\n")
    os.makedirs(f"{A}/splits", exist_ok=True)
    for s in sorted(med):
        with open(f"{A}/splits/test_{s}.txt", "w") as f:
            f.writelines(f"./images/test/{r['image']}\n" for r in rows if r["split"] == "test" and r["source"] == s)
    return src, med, scale, rows


# ---------------- C: 가상 염색 패치 ----------------
def prep_virtual_staining(out, src, anemia, dse):
    C = f"{out}/3_virtual_staining"
    for sub in ["rbc_only", "all"]:
        for d in ["trainA", "trainB", "testA", "testB"]:
            os.makedirs(f"{C}/{sub}/{d}", exist_ok=True)
    k = TARGET_RBC_PX / REDTELL_RBC_PX
    cnt = defaultdict(int)
    red_rows = []

    # trainA: anemia 전체 + dse 학습 분할 (라벨 안 씀)
    for g in ["control", "scd", "thalassemia"]:
        for f in sorted(glob.glob(f"{anemia}/{g}/images/*.tif")):
            stem = f"anemia_{g}_{os.path.splitext(os.path.basename(f))[0]}"
            img = unstained_prep(imread(f, cv2.IMREAD_GRAYSCALE), k)
            cnt["rbc_only/trainA"] += save_patches(img, f"{C}/rbc_only/trainA", stem, PATCH_RBC, STRIDE_A_RBC, drop_empty=True)
            cnt["all/trainA"] += save_patches(img, f"{C}/all/trainA", stem, PATCH_ALL, STRIDE_ALL)
            red_rows.append({"image": f"anemia_data/{g}/images/{os.path.basename(f)}", "use": "가상 염색 학습 (trainA)"})
    for i in DSE_GAN_IDS:
        img = unstained_prep(read_dse_image(dse, i), k)
        stem = f"dse_{i}"
        cnt["rbc_only/trainA"] += save_patches(img, f"{C}/rbc_only/trainA", stem, PATCH_RBC, STRIDE_A_RBC, drop_empty=True)
        cnt["all/trainA"] += save_patches(img, f"{C}/all/trainA", stem, PATCH_ALL, STRIDE_ALL)
        red_rows.append({"image": f"dse_data/images/{i}.tif", "use": "가상 염색 학습 (trainA)"})
    # testA: dse 평가 분할 (4_eval_unstained 과 같은 영상)
    for i in DSE_EVAL_IDS:
        img = unstained_prep(read_dse_image(dse, i), k)
        stem = f"dse_{i}"
        cnt["rbc_only/testA"] += save_patches(img, f"{C}/rbc_only/testA", stem, PATCH_RBC, PATCH_RBC, drop_empty=True)
        cnt["all/testA"] += save_patches(img, f"{C}/all/testA", stem, PATCH_ALL, PATCH_ALL)

    # trainB / testB: TXL 학습 / 검증 분할 (정규화된 A 영상 사용)
    A = f"{out}/1_detection_train"
    for sp, d in [("train", "trainB"), ("val", "testB")]:
        for im in sorted(glob.glob(f"{A}/images/{sp}/*.png")):
            img = imread(im)
            ih, iw = img.shape[:2]
            n = os.path.basename(im)
            stem = f"{src[n]}_{n[:-4]}"
            cnt[f"all/{d}"] += save_patches(img, f"{C}/all/{d}", stem, PATCH_ALL, STRIDE_ALL)
            wbc = [((x - bw / 2) * iw, (y - bh / 2) * ih, (x + bw / 2) * iw, (y + bh / 2) * ih)
                   for c, x, y, bw, bh in read_labels(f"{A}/labels/{sp}/{n[:-4]}.txt") if c == 0]
            for y, x, p in grid_patches(img, PATCH_RBC, STRIDE_RBC):
                if not any(a < x + PATCH_RBC and c > x and b < y + PATCH_RBC and e > y for a, b, c, e in wbc):
                    imwrite(f"{C}/rbc_only/{d}/{stem}_{y}_{x}.png", p)
                    cnt[f"rbc_only/{d}"] += 1
    return dict(cnt), red_rows


# ---------------- D: 평가용 실제 비염색 + 정답 ----------------
def read_dse_image(dse, i):
    a = np.array(Image.open(f"{dse}/images/{i}.tif")).astype(np.float32)  # float32, 0~1
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def prep_eval(out, dse):
    D = f"{out}/4_eval_unstained/redtell_dse"
    for d in ["images", "labels", "masks"]:
        os.makedirs(f"{D}/{d}", exist_ok=True)
    k = TARGET_RBC_PX / REDTELL_RBC_PX
    rows = []
    for i in DSE_EVAL_IDS:
        img = unstained_prep(read_dse_image(dse, i), k)
        h, w = img.shape[:2]
        imwrite(f"{D}/images/dse_{i}.png", img)
        lab = np.array(Image.open(f"{dse}/annotations/{i}.tif"))
        lab = cv2.resize(lab.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
        imwrite(f"{D}/masks/dse_{i}.png", lab)  # 16bit 인스턴스 마스크 (0 배경, 1.. 세포 번호)
        props = measure.regionprops(lab)
        with open(f"{D}/labels/dse_{i}.txt", "w") as f:
            for p in props:
                y0, x0, y1, x1 = p.bbox
                f.write(f"1 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
        rows.append({"image": f"dse_{i}.png", "n_RBC": len(props), "n_WBC": 0})
    write_csv(f"{D}/counts.csv", rows)
    with open(f"{D}/README.txt", "w", encoding="utf-8") as f:
        f.write("RedTell dse 평가 분할 (94~186번, 93장). 실제 비염색 RBC 정답.\n"
                "images/ : 학습 영상과 같은 전처리(밝기 보정 + 배율 정규화) 적용. 가상 염색 모델 입력으로 사용\n"
                "labels/ : YOLO 형식 박스 (클래스 1 = RBC)\n"
                "masks/  : 16bit 인스턴스 마스크 (셀 기반 방식 평가용)\n"
                "counts.csv : 이미지별 정답 RBC 개수 (MAE 계산용)\n"
                "WBC 정답은 없음. 짝 데이터 확보 시 추가.\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txl", required=True)
    ap.add_argument("--anemia", required=True)
    ap.add_argument("--dse", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/meta", exist_ok=True)

    src, med, scale, txl_rows = prep_txl(a.txl, a.out)
    write_csv(f"{a.out}/meta/txl_images.csv", txl_rows)
    write_csv(f"{a.out}/meta/scale_factors.csv",
              [{"source": s, "rbc_median_px": round(med[s], 1), "scale": round(scale[s], 4)} for s in sorted(med)]
              + [{"source": "RedTell (anemia, dse)", "rbc_median_px": REDTELL_RBC_PX, "scale": round(TARGET_RBC_PX / REDTELL_RBC_PX, 4)}])
    cnt, red_rows = prep_virtual_staining(a.out, src, a.anemia, a.dse)
    eval_rows = prep_eval(a.out, a.dse)
    red_rows += [{"image": f"dse_data/images/{i}.tif", "use": "평가 (testA + 4_eval_unstained)"} for i in DSE_EVAL_IDS]
    write_csv(f"{a.out}/meta/redtell_images.csv", red_rows)

    print("[A] 검출 학습:", {sp: sum(r['split'] == sp for r in txl_rows) for sp in ['train', 'val', 'test']})
    print("[C] 가상 염색 패치:", cnt)
    print("[D] 평가: 영상", len(eval_rows), "장, 정답 RBC", sum(r['n_RBC'] for r in eval_rows), "개")


if __name__ == "__main__":
    main()
