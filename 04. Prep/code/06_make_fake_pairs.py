"""
06. 가짜 짝 데이터 만들기 (짝이 필요한 가상 염색 모델 pix2pix·BBDM용)

왜 필요한가
  pix2pix·BBDM은 "같은 세포의 비염색 사진 ↔ 염색 사진" 짝이 있어야 학습됩니다. 우리에게는 짝이 없습니다.
  그래서 실제 염색 조각(TXL-PBC)을 규칙으로 "비염색처럼" 바꿔 (가짜 비염색, 진짜 염색) 짝을 만듭니다.

규칙 (실제 비염색 RedTell 명시야의 특징을 흉내 냄)
  1) 염색 영상에서 세포 밀도(흡광도, 초록 채널 위주)를 구해 세포 영역을 찾음 (가운데 옅은 부분도 세포 안으로)
  2) 세포 경계에 얇고 진한 어두운 선 + 그 옆에 밝은 선  ← 실제 비염색에서 가장 뚜렷한 특징
  3) 세포 안은 배경과 비슷한 회색, 염색 영상의 세포 안 명암을 약하게 남김
  4) 배경의 넓은 얼룩 + 노이즈 + 약한 흐림
  5) 밝기·선 굵기·진하기 등은 조각마다 범위 안에서 무작위 (한 가지 모양에 치우치지 않게)
  값의 범위는 RedTell 학습용 영상(dse 1~83)에서 잰 세포 경계 밝기 변화에 맞췄습니다. val·test 영상은 보지 않았습니다.

입력  : 04. Prep/3_virtual_staining/rbc_only/trainB (염색, 학습용), testB (염색, TXL 검증 분할)
--rule v1 | v2 (기본 v2). v1 결과는 rbc_only/, v2 결과는 rbc_only_v2/ 에 따로 저장 (서로 덮어쓰지 않음)

출력  : 04. Prep/5_paired_synthetic/rbc_only[_v2]/
          A/train, A/val     가짜 비염색 (흑백 128×128)
          B/train, B/val     진짜 염색 (같은 이름 = 짝)
          AB/train, AB/val   A|B 를 옆으로 붙인 그림 (pix2pix 공식 코드 형식)
          pairs.csv          짝마다 쓴 무작위 값 (재현용)
          preview.png        확인용 (왼쪽 염색 / 오른쪽 가짜 비염색)
        04. Prep/5_paired_synthetic/check_txl_val/  (--check 일 때)
          TXL 검증 영상 전체를 가짜 비염색으로 바꾼 것 + 정답 박스
          → YOLO로 채점해서 "실제 비염색만큼 어려운지" 확인용 (실제 비염색에서 YOLO recall 약 0.72)

사용법
  python 06_make_fake_pairs.py --prep "D:/05. Blood/04. Prep" --rule v2 --variants 2 --check
  --variants N : 염색 조각 1개당 가짜 비염색을 N개(서로 다른 무작위 값) 만듦 → 학습 짝 수 N배
"""
import argparse
import csv
import glob
import os
import shutil

import cv2
import numpy as np
from scipy import ndimage as ndi


def imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)


# ---------------- 가짜 비염색 규칙 ----------------
def cell_density(bgr):
    """염색 영상 -> 세포 밀도 지도 (0 배경 ~ 1 세포). 흡광도(OD), 초록 채널 위주"""
    f = bgr.astype(np.float32) + 1
    bg = np.percentile(f.reshape(-1, 3), 95, axis=0)
    od = (-np.log(f / bg)).clip(0, None)
    d = od[:, :, 1] * 0.7 + od.mean(2) * 0.3
    d = cv2.GaussianBlur(d, (0, 0), 1.0)
    return (d / (np.percentile(d, 99) + 1e-6)).clip(0, 1)


def sample_params(rng):
    return dict(bg=rng.uniform(125, 150), thr=rng.uniform(0.25, 0.4),
                depth=rng.uniform(45, 90), sig=rng.uniform(0.8, 1.4),
                bright=rng.uniform(10, 40), bright_in=bool(rng.random() < 0.7),
                inner=rng.uniform(-4, 8), tex=rng.uniform(5, 20),
                mottle=rng.uniform(2, 7), noise=rng.uniform(2, 4), blur=rng.uniform(0.3, 0.7))


def fake_unstained(bgr, rng, p):
    d = cell_density(bgr)
    cell = ndi.binary_opening(ndi.binary_fill_holes(d > p["thr"]), iterations=1)
    m = cv2.GaussianBlur(cell.astype(np.float32), (0, 0), 0.7)
    ms = cv2.GaussianBlur(m, (0, 0), p["sig"])
    ring = np.hypot(cv2.Sobel(ms, cv2.CV_32F, 1, 0), cv2.Sobel(ms, cv2.CV_32F, 0, 1))
    ring /= ring.max() + 1e-6                                    # 경계의 어두운 선
    L = cv2.Laplacian(ms, cv2.CV_32F, ksize=3)
    L /= np.abs(L).max() + 1e-6
    side = (-L if p["bright_in"] else L).clip(0, None)           # 어두운 선 옆의 밝은 선
    tex = d - cv2.GaussianBlur(d, (0, 0), 6)                     # 세포 안 명암
    H, W = d.shape
    mottle = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 8)
    mottle /= mottle.std() + 1e-6                                # 배경 얼룩
    g = (p["bg"] - p["depth"] * ring + p["bright"] * side - p["inner"] * m
         - p["tex"] * tex * m + p["mottle"] * mottle)
    g = cv2.GaussianBlur(g, (0, 0), p["blur"]) + rng.normal(0, p["noise"], g.shape)
    return g.clip(0, 255).astype(np.uint8)


# ---------------- 규칙 v2 (v1은 세포가 "선 그림"처럼 나와 실제와 달랐음) ----------------
#   v1 대비 바뀐 점
#   - 입체감(relief): 세포 가장자리 한쪽은 어둡고 반대쪽은 밝게 (실제 비염색 명시야에서 가장 눈에 띄는 특징). 빛 방향은 영상마다 무작위
#   - 붙은 세포를 하나씩 나눠(watershed) 세포마다 경계선을 그림 (v1은 겹친 세포가 한 덩어리 윤곽이 됨)
def split_cells(cell, min_dist=18):
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    dist = ndi.distance_transform_edt(cell)
    pk = peak_local_max(dist, min_distance=min_dist, labels=ndi.label(cell)[0], exclude_border=False)
    mk = np.zeros(cell.shape, np.int32)
    for i, (y, x) in enumerate(pk, 1):
        mk[y, x] = i
    return watershed(-dist, mk, mask=cell)


def sample_params_v2(rng):
    return dict(bg=rng.uniform(125, 150), thr=rng.uniform(0.25, 0.4),
                relief=rng.uniform(25, 60), angle=rng.uniform(0, 2 * np.pi), rsig=rng.uniform(1.5, 3.0),
                line=rng.uniform(15, 45), inner=rng.uniform(-6, 10), tex=rng.uniform(5, 20),
                mottle=rng.uniform(2, 7), noise=rng.uniform(2, 4), blur=rng.uniform(0.4, 0.9))


def fake_unstained_v2(bgr, rng, p):
    from skimage.segmentation import find_boundaries
    d = cell_density(bgr)
    cell = ndi.binary_opening(ndi.binary_fill_holes(d > p["thr"]), iterations=1)
    lab = split_cells(cell)
    m = (lab > 0).astype(np.float32)
    ms = cv2.GaussianBlur(m, (0, 0), p["rsig"])
    gx, gy = cv2.Sobel(ms, cv2.CV_32F, 1, 0), cv2.Sobel(ms, cv2.CV_32F, 0, 1)
    rel = np.cos(p["angle"]) * gx + np.sin(p["angle"]) * gy
    rel /= np.abs(rel).max() + 1e-6                               # 입체감
    bd = cv2.GaussianBlur(find_boundaries(lab, mode="thick").astype(np.float32), (0, 0), 0.8)  # 세포별 경계선
    tex = d - cv2.GaussianBlur(d, (0, 0), 6)
    H, W = d.shape
    mottle = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 8)
    mottle /= mottle.std() + 1e-6
    g = (p["bg"] + p["relief"] * rel - p["line"] * bd - p["inner"] * cv2.GaussianBlur(m, (0, 0), 1)
         - p["tex"] * tex * m + p["mottle"] * mottle)
    g = cv2.GaussianBlur(g, (0, 0), p["blur"]) + rng.normal(0, p["noise"], g.shape)
    return g.clip(0, 255).astype(np.uint8)


RULES = {"v1": (sample_params, fake_unstained), "v2": (sample_params_v2, fake_unstained_v2)}


# ---------------- 실행 ----------------
def make_split(src_dir, out, split, variants, rng, rows, rule):
    sp, fu = RULES[rule]
    for d in ["A", "B", "AB"]:
        os.makedirs(os.path.join(out, d, split), exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    for f in files:
        b = imread(f)
        stem = os.path.splitext(os.path.basename(f))[0]
        for v in range(variants):
            p = sp(rng)
            a = fu(b, rng, p)
            name = f"{stem}_v{v}.png"
            imwrite(os.path.join(out, "A", split, name), a)
            imwrite(os.path.join(out, "B", split, name), b)
            imwrite(os.path.join(out, "AB", split, name), np.hstack([cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), b]))
            rows.append({"split": split, "pair": name, "source_patch": os.path.basename(f),
                         **{k: (round(x, 3) if isinstance(x, float) else x) for k, x in p.items()}})
    return len(files) * variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="04. Prep 폴더")
    ap.add_argument("--variants", type=int, default=2, help="염색 조각 1개당 가짜 비염색 개수")
    ap.add_argument("--check", action="store_true", help="TXL 검증 영상 전체도 가짜 비염색으로 (YOLO 확인용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default="v2", choices=["v1", "v2"], help="가짜 비염색 규칙 (v1: 첫 버전, v2: 입체감·세포 분리 추가)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    src = os.path.join(a.prep, "3_virtual_staining", "rbc_only")
    sfx = "" if a.rule == "v1" else f"_{a.rule}"
    out = os.path.join(a.prep, "5_paired_synthetic", f"rbc_only{sfx}")
    if os.path.exists(out):
        print(f"이미 있습니다: {out}\n다시 만들려면 이 폴더 이름을 바꾸거나 옮긴 뒤 실행하세요 (덮어쓰지 않음).")
        return
    rows = []
    n_tr = make_split(os.path.join(src, "trainB"), out, "train", a.variants, rng, rows, a.rule)
    n_va = make_split(os.path.join(src, "testB"), out, "val", 1, rng, rows, a.rule)
    with open(os.path.join(out, "pairs.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 확인용 미리보기 (앞 8쌍)
    prev = sorted(glob.glob(os.path.join(out, "AB", "train", "*.png")))[: 8 * a.variants: a.variants]
    imwrite(os.path.join(out, "preview.png"), np.vstack([np.hstack([imread(p)[:, 128:], imread(p)[:, :128]]) for p in prev]))
    print(f"[짝] 학습 {n_tr}쌍 (염색 조각 × {a.variants}), 검증 {n_va}쌍 → {out}")

    if a.check:
        det = os.path.join(a.prep, "1_detection_train")
        C = os.path.join(a.prep, "5_paired_synthetic", f"check_txl_val{sfx}")
        sp, fu = RULES[a.rule]
        os.makedirs(os.path.join(C, "images"), exist_ok=True)
        os.makedirs(os.path.join(C, "labels"), exist_ok=True)
        files = sorted(glob.glob(os.path.join(det, "images", "val", "*.png")))
        for f in files:
            n = os.path.basename(f)
            imwrite(os.path.join(C, "images", n), fu(imread(f), rng, sp(rng)))
            shutil.copy(os.path.join(det, "labels", "val", n[:-4] + ".txt"), os.path.join(C, "labels", n[:-4] + ".txt"))
        print(f"[확인용] TXL 검증 {len(files)}장을 가짜 비염색으로 → {C}")


if __name__ == "__main__":
    main()
