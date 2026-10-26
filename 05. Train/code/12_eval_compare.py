"""
12. 검출·계수 방식 비교 평가 (총괄 문서 4-1 평가 0)
같은 테스트 영상, 같은 정답, 같은 지표로 두 방식을 비교합니다.
  - 박스 기반: YOLO
  - 셀 기반 : Cellpose(세포 분할) + 11번 분류기(RBC/WBC)

기본 입력은 실제 염색 테스트 분할(04. Prep/1_detection_train/images/test)입니다.
--images, --labels 를 바꾸면 다른 영상에도 그대로 쓸 수 있습니다.
  예) 평가 ② 하한: 비염색 영상을 가상 염색 없이 그대로 넣기
      --images "04. Prep/4_eval_unstained/redtell_dse/images" --labels "04. Prep/4_eval_unstained/redtell_dse/labels"

지표 (클래스별: WBC, RBC)
  - 개수 오차 MAE : 이미지마다 |예측 개수 - 정답 개수| 의 평균  ← 메인 (계수 시스템이므로)
  - 개수 상대오차 : 이미지마다 |예측 - 정답| / 정답 의 평균 (정답 0인 이미지 제외)
  - Precision / Recall / F1 : 예측 박스와 정답 박스를 IoU 0.5 이상이면 맞은 것으로 1:1 매칭
  - 처리 시간 : 이미지 1장당 평균 (ms)

출력 (--out 폴더)
  summary.csv      방식별 전체 지표
  by_source.csv    방식별·출처별 지표 (meta/txl_images.csv 가 있을 때)
  per_image.csv    이미지별 정답·예측 개수

사용법
  python 12_eval_compare.py --prep "D:/05. Blood/04. Prep" \
      --yolo "D:/05. Blood/05. Train/runs/yolo/yolo11s/weights/best.pt" \
      --classifier "D:/05. Blood/05. Train/runs/cell_classifier/best.pt" \
      --out "D:/05. Blood/05. Train/results/eval0_stained_test"
  한쪽만 평가하려면 --yolo 또는 --classifier 만 주면 됩니다.
"""
import argparse
import csv
import glob
import os
import time
from collections import defaultdict
from importlib.metadata import version as pkg_version

import cv2
import numpy as np

NAMES = {0: "WBC", 1: "RBC"}
CROP_PAD, CROP_SIZE = 0.15, 96  # 03_prep_data.py 와 같은 값 (분류기 학습 조각과 똑같이 자르기 위함)


def imread(path):
    """한글 경로에서도 읽히는 imread. 흑백이면 3채널로 바꿈"""
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def read_gt(path, w, h):
    """정답 박스. 완전히 같은 줄(중복 라벨)은 한 번만 셈 (TXL-PBC 테스트에 13개 있음, YOLO도 똑같이 제거함)"""
    boxes, seen = [], set()
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 5 and tuple(p[:5]) not in seen:
                seen.add(tuple(p[:5]))
                c, x, y, bw, bh = int(p[0]), *map(float, p[1:5])
                boxes.append((c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def match(gt, pred, thr=0.5):
    """클래스별 IoU 그리디 1:1 매칭 -> {class: (TP, FP, FN)}"""
    out = {}
    for c in NAMES:
        g = [b[1:] for b in gt if b[0] == c]
        p = sorted([b for b in pred if b[0] == c], key=lambda b: -b[5])
        used, tp = set(), 0
        for b in p:
            best, bi = 0, -1
            for i, gb in enumerate(g):
                if i not in used:
                    v = iou(b[1:5], gb)
                    if v > best:
                        best, bi = v, i
            if best >= thr:
                used.add(bi)
                tp += 1
        out[c] = (tp, len(p) - tp, len(g) - tp)
    return out


# ---------------- 방식 1: YOLO ----------------
class YoloMethod:
    name = "YOLO"

    def __init__(self, weights, device, conf):
        from ultralytics import YOLO
        self.m, self.device, self.conf = YOLO(weights), device, conf

    def __call__(self, img):
        r = self.m.predict(img, conf=self.conf, device=self.device, verbose=False)[0]
        b = r.boxes
        return [(int(c), *xyxy, float(s)) for c, xyxy, s in zip(b.cls.tolist(), b.xyxy.tolist(), b.conf.tolist())]


# ---------------- 방식 2: Cellpose + 분류기 ----------------
class CellposeMethod:
    name = "Cellpose+classifier"

    def __init__(self, classifier, device, diameter, cp_model):
        import torch
        from cellpose import models
        from torchvision import models as tvm
        self.torch = torch
        gpu = device != "cpu"
        self.v4 = int(pkg_version("cellpose").split(".")[0]) >= 4
        if self.v4:  # Cellpose 4 (Cellpose-SAM): 지름 자동 처리
            self.cp = models.CellposeModel(gpu=gpu)
        else:        # Cellpose 3: cyto3 + 지름 지정 (정규화 후 RBC 약 64 px)
            self.cp = models.CellposeModel(gpu=gpu, model_type=cp_model or "cyto3")
        self.diameter = diameter
        ck = torch.load(classifier, map_location="cpu")
        net = tvm.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(ck["classes"]))
        net.load_state_dict(ck["model"])
        self.dev = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.net = net.to(self.dev).eval()
        self.cls_idx = {n: i for i, n in enumerate(ck["classes"])}  # "RBC"->0, "WBC"->1
        self.mean = np.array(ck["mean"], np.float32)
        self.std = np.array(ck["std"], np.float32)

    def __call__(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.v4:
            masks = self.cp.eval(rgb)[0]
        else:
            masks = self.cp.eval(rgb, diameter=self.diameter, channels=[0, 0])[0]
        H, W = masks.shape
        boxes, crops = [], []
        for lab in np.unique(masks):
            if lab == 0:
                continue
            ys, xs = np.where(masks == lab)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            bw, bh = (x1 - x0) * (1 + 2 * CROP_PAD), (y1 - y0) * (1 + 2 * CROP_PAD)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            a0, b0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            a1, b1 = int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))
            if a1 - a0 < 4 or b1 - b0 < 4:
                continue
            crop = cv2.resize(rgb[b0:b1, a0:a1], (CROP_SIZE, CROP_SIZE)).astype(np.float32) / 255
            crops.append(((crop - self.mean) / self.std).transpose(2, 0, 1))
            boxes.append((float(x0), float(y0), float(x1), float(y1)))
        if not boxes:
            return []
        with self.torch.no_grad():
            x = self.torch.from_numpy(np.stack(crops)).to(self.dev)
            prob = self.torch.softmax(self.net(x), 1).cpu().numpy()
        out = []
        for b, p in zip(boxes, prob):
            k = int(p.argmax())
            cls_name = [n for n, i in self.cls_idx.items() if i == k][0]
            out.append((0 if cls_name == "WBC" else 1, *b, float(p[k])))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="04. Prep 폴더")
    ap.add_argument("--images", default=None, help="기본: <prep>/1_detection_train/images/test")
    ap.add_argument("--labels", default=None, help="기본: <prep>/1_detection_train/labels/test")
    ap.add_argument("--yolo", default=None, help="YOLO best.pt")
    ap.add_argument("--classifier", default=None, help="11번 분류기 best.pt (Cellpose 방식)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0", help="GPU 번호, CPU면 cpu")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO 신뢰도 기준")
    ap.add_argument("--diameter", type=float, default=64, help="Cellpose 3용 세포 지름(px)")
    ap.add_argument("--cp_model", default=None, help="Cellpose 3 모델 이름 (기본 cyto3)")
    ap.add_argument("--limit", type=int, default=0, help="테스트용: 이미지 수 제한")
    a = ap.parse_args()

    img_dir = a.images or os.path.join(a.prep, "1_detection_train", "images", "test")
    lab_dir = a.labels or os.path.join(a.prep, "1_detection_train", "labels", "test")
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if a.limit:
        files = files[: a.limit]
    src = {}
    meta = os.path.join(a.prep, "meta", "txl_images.csv")
    if os.path.exists(meta):
        src = {r["image"]: r["source"] for r in csv.DictReader(open(meta, encoding="utf-8-sig"))}

    methods = []
    if a.yolo:
        methods.append(YoloMethod(a.yolo, a.device, a.conf))
    if a.classifier:
        methods.append(CellposeMethod(a.classifier, a.device, a.diameter, a.cp_model))
    if not methods:
        raise SystemExit("--yolo 또는 --classifier 중 하나는 있어야 합니다.")

    os.makedirs(a.out, exist_ok=True)
    per_image, summary, by_src = [], [], []
    for m in methods:
        stat = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))  # stat[source][class] = TP,FP,FN
        err = defaultdict(lambda: defaultdict(list))                  # err[source][class] = |pred-gt|
        rel = defaultdict(list)
        times = []
        m(imread(files[0]))  # 첫 실행 준비(워밍업), 시간에서 제외
        for f in files:
            img = imread(f)
            h, w = img.shape[:2]
            n = os.path.basename(f)
            gt = read_gt(os.path.join(lab_dir, n[:-4] + ".txt"), w, h)
            t0 = time.perf_counter()
            pred = m(img)
            times.append((time.perf_counter() - t0) * 1000)
            s = src.get(n, "all")
            res = match(gt, pred)
            row = {"method": m.name, "image": n, "source": s}
            for c, cname in NAMES.items():
                ng, npd = sum(b[0] == c for b in gt), sum(b[0] == c for b in pred)
                row[f"gt_{cname}"], row[f"pred_{cname}"] = ng, npd
                for key in (s, "ALL"):
                    stat[key][c] += res[c]
                    err[key][c].append(abs(npd - ng))
                if ng > 0:
                    rel[c].append(abs(npd - ng) / ng)
            per_image.append(row)

        def metrics(key):
            d = {}
            for c, cname in NAMES.items():
                tp, fp, fn = stat[key][c]
                p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                d[f"{cname}_MAE"] = round(float(np.mean(err[key][c])), 3)
                d[f"{cname}_precision"] = round(p, 4)
                d[f"{cname}_recall"] = round(r, 4)
                d[f"{cname}_F1"] = round(2 * p * r / max(p + r, 1e-9), 4)
            return d

        s_all = {"method": m.name, "n_images": len(files), **metrics("ALL"),
                 "WBC_rel_error": round(float(np.mean(rel[0])), 4) if rel[0] else "",
                 "RBC_rel_error": round(float(np.mean(rel[1])), 4) if rel[1] else "",
                 "ms_per_image": round(float(np.mean(times)), 1)}
        summary.append(s_all)
        for key in sorted(k for k in stat if k != "ALL"):
            by_src.append({"method": m.name, "source": key, "n_images": len(err[key][1]), **metrics(key)})
        print(f"\n[{m.name}]")
        for k, v in s_all.items():
            print(f"  {k}: {v}")

    for name, rows in [("summary.csv", summary), ("by_source.csv", by_src), ("per_image.csv", per_image)]:
        if rows:
            with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    print("\n저장:", a.out)


if __name__ == "__main__":
    main()
