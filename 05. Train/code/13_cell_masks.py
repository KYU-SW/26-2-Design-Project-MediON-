"""
13. B 방식: YOLO로 세포를 찾고 → 찾은 박스마다 세포 윤곽(마스크)을 따서 "세포 단위" 결과 만들기

왜 이렇게 하나
  - 개수 세기는 YOLO가 더 정확했음 (평가 0: RBC MAE 1.21 vs Cellpose 2.60)
  - 하지만 박스만으로는 세포 모양(면적·둥근 정도 등)을 알 수 없음
  - 그래서 YOLO 박스 안에서 윤곽을 따는 단계를 붙임 → 세포마다 마스크 + 형태 지표

윤곽 따는 방법 (--mask, 여러 개 주면 각각 따로 결과를 냄)
  cellpose : 영상 전체를 Cellpose로 분할한 뒤, YOLO 박스와 가장 잘 겹치는 Cellpose 세포를 그 박스의 마스크로 씀
             (박스와 겹치는 Cellpose 세포가 없으면 그 세포는 마스크 없음으로 기록)
  sam      : YOLO 박스를 SAM 2.1에 "이 박스 안의 물체를 따줘"라고 넣어 박스마다 마스크를 얻음
  ellipse  : 박스에 내접하는 타원 (모델 없음, 비교 기준선)

출력 (--out/<방식>/)
  masks/<영상>.png    16bit 세포 번호 지도 (0 배경, 1.. 세포)
  cells.csv           세포 1개 = 1줄 : 클래스, YOLO 신뢰도, 박스, 면적, 둘레, 원형도, 지름, 장·단축, 마스크 출처
  overlay/<영상>.png  확인용 그림 (--overlay N 장)
  summary.csv         (--gt_masks 를 줄 때) 마스크 정확도

마스크 평가 (--gt_masks, RedTell dse 정답 마스크. RBC만 있음)
  예측 세포와 정답 세포를 마스크 IoU 0.5 이상이면 1:1로 맞은 것으로 매칭
  - seg_precision / seg_recall / seg_F1 : 세포 단위로 맞게 분할했는지
  - mean_IoU_matched : 맞은 세포들의 평균 마스크 IoU (윤곽이 얼마나 정확한지)
  - area_rel_error   : 맞은 세포들의 |예측 면적 - 정답 면적| / 정답 면적 평균 (크기 측정 정확도)
  - mask_coverage    : YOLO가 찾은 세포 중 마스크가 붙은 비율

사용법 (예: 가상 염색한 평가 영상)
  python 13_cell_masks.py --images "D:/05. Blood/05. Train/virtual_stained/cyclegan_ep50/test" ^
      --yolo "D:/05. Blood/05. Train/runs/yolo/yolo11s/weights/best.pt" --conf 0.7 ^
      --mask cellpose sam ellipse ^
      --gt_masks "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse/masks" ^
      --out "D:/05. Blood/05. Train/results/cellmask_cyclegan_ep50"
"""
import argparse
import csv
import glob
import math
import os
import time
from importlib.metadata import version as pkg_version

import cv2
import numpy as np
from skimage import measure

NAMES = {0: "WBC", 1: "RBC"}


def imread(path, flags=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)


def to_bgr(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


# ---------------- 윤곽 따는 방법 ----------------
class EllipseMasker:
    name = "ellipse"

    def __call__(self, img, dets):
        H, W = img.shape[:2]
        out = []
        for _, x0, y0, x1, y1, _ in dets:
            m = np.zeros((H, W), np.uint8)
            cv2.ellipse(m, (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))),
                        (max(1, int(round((x1 - x0) / 2))), max(1, int(round((y1 - y0) / 2)))), 0, 0, 360, 1, -1)
            out.append((m.astype(bool), "ellipse"))
        return out


class SamMasker:
    name = "sam"

    def __init__(self, weights, device):
        from ultralytics import SAM
        self.m, self.device = SAM(weights), device

    def __call__(self, img, dets):
        if not dets:
            return []
        r = self.m(img, bboxes=[list(d[1:5]) for d in dets], device=self.device, verbose=False)[0]
        ms = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else []
        out = []
        for i in range(len(dets)):
            if i < len(ms) and ms[i].any():
                out.append((ms[i], "sam"))
            else:
                out.append((None, "none"))
        return out


class CellposeMasker:
    name = "cellpose"

    def __init__(self, device, diameter=64, min_iou=0.3):
        from cellpose import models
        gpu = device != "cpu"
        self.v4 = int(pkg_version("cellpose").split(".")[0]) >= 4
        self.cp = models.CellposeModel(gpu=gpu) if self.v4 else models.CellposeModel(gpu=gpu, model_type="cyto3")
        self.diameter, self.min_iou = diameter, min_iou

    def segment(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.v4:
            return self.cp.eval(rgb)[0]
        return self.cp.eval(rgb, diameter=self.diameter, channels=[0, 0])[0]

    def __call__(self, img, dets):
        lab = self.segment(img)
        props = {p.label: p for p in measure.regionprops(lab)}
        cand = {k: (p.bbox[1], p.bbox[0], p.bbox[3], p.bbox[2]) for k, p in props.items()}
        used = set()
        # 신뢰도 높은 박스부터 가장 잘 겹치는 Cellpose 세포를 1:1로 배정
        res = [None] * len(dets)
        for i in sorted(range(len(dets)), key=lambda i: -dets[i][5]):
            best, bk = 0, None
            for k, cb in cand.items():
                if k not in used:
                    v = box_iou(dets[i][1:5], cb)
                    if v > best:
                        best, bk = v, k
            if bk is not None and best >= self.min_iou:
                used.add(bk)
                res[i] = (lab == bk, "cellpose")
            else:
                res[i] = (None, "none")
        return res


# ---------------- 형태 지표 + 평가 ----------------
def cell_features(mask):
    if mask is None or not mask.any():
        return {"area_px": "", "perimeter_px": "", "circularity": "", "eq_diameter_px": "",
                "major_axis_px": "", "minor_axis_px": "", "solidity": ""}
    p = max(measure.regionprops(mask.astype(np.uint8)), key=lambda r: r.area)
    per = p.perimeter if p.perimeter > 0 else 1
    return {"area_px": int(p.area), "perimeter_px": round(per, 1), "circularity": round(4 * math.pi * p.area / per ** 2, 3),
            "eq_diameter_px": round(p.equivalent_diameter_area, 1), "major_axis_px": round(p.axis_major_length, 1),
            "minor_axis_px": round(p.axis_minor_length, 1), "solidity": round(p.solidity, 3)}


def build_label_map(shape, masks, confs):
    """겹치는 픽셀은 신뢰도 높은 세포가 가져감"""
    lab = np.zeros(shape, np.uint16)
    for i in sorted(range(len(masks)), key=lambda i: confs[i]):
        if masks[i] is not None:
            lab[masks[i]] = i + 1
    return lab


def match_masks(pred, gt, thr=0.5):
    """마스크 IoU 그리디 1:1 매칭 → (TP, FP, FN, [IoU], [면적 상대오차])"""
    pl, gl = [v for v in np.unique(pred) if v], [v for v in np.unique(gt) if v]
    if not pl or not gl:
        return 0, len(pl), len(gl), [], []
    p_area = {v: (pred == v).sum() for v in pl}
    g_area = {v: (gt == v).sum() for v in gl}
    both = (pred > 0) & (gt > 0)
    pairs = {}
    for a, b in zip(pred[both].tolist(), gt[both].tolist()):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    cands = sorted(((inter / (p_area[a] + g_area[b] - inter), a, b) for (a, b), inter in pairs.items()), reverse=True)
    up, ug, ious, aerr = set(), set(), [], []
    for v, a, b in cands:
        if v < thr:
            break
        if a in up or b in ug:
            continue
        up.add(a)
        ug.add(b)
        ious.append(v)
        aerr.append(abs(p_area[a] - g_area[b]) / g_area[b])
    tp = len(ious)
    return tp, len(pl) - tp, len(gl) - tp, ious, aerr


def overlay(img, lab, dets, srcs):
    o = img.copy()
    rng = np.random.default_rng(0)
    for v in np.unique(lab):
        if v:
            o[lab == v] = (0.5 * o[lab == v] + 0.5 * rng.integers(60, 255, 3)).astype(np.uint8)
    for (c, x0, y0, x1, y1, s), src in zip(dets, srcs):
        col = (0, 0, 255) if c == 0 else ((0, 200, 0) if src != "none" else (0, 200, 255))
        cv2.rectangle(o, (int(x0), int(y0)), (int(x1), int(y1)), col, 1)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--conf", type=float, default=0.7, help="YOLO 신뢰도 기준 (val로 정한 값 0.7)")
    ap.add_argument("--mask", nargs="+", default=["cellpose", "sam", "ellipse"], choices=["cellpose", "sam", "ellipse"])
    ap.add_argument("--sam_weights", default="sam2.1_t.pt", help="처음 실행 때 자동 다운로드 (약 75MB)")
    ap.add_argument("--gt_masks", default=None, help="정답 마스크 폴더 (있으면 마스크 정확도 계산)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", type=int, default=5, help="확인용 그림 저장 장수")
    ap.add_argument("--device", default="0", help="GPU 번호, CPU면 cpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(a.yolo)
    dev_t = "cpu" if a.device == "cpu" else f"cuda:{a.device}"
    files = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))
    if a.limit:
        files = files[: a.limit]

    # YOLO 검출은 한 번만 하고 모든 방식이 같은 박스를 씀 (마스크 방식끼리 공정 비교)
    dets_all, t_yolo = {}, 0.0
    for f in files:
        img = to_bgr(imread(f))
        t0 = time.time()
        b = yolo.predict(img, conf=a.conf, device=a.device, verbose=False)[0].boxes
        t_yolo += time.time() - t0
        dets_all[f] = [(int(c), *xyxy, float(s)) for c, xyxy, s in zip(b.cls.tolist(), b.xyxy.tolist(), b.conf.tolist())]

    summary = []
    for mname in a.mask:
        masker = {"ellipse": lambda: EllipseMasker(),
                  "sam": lambda: SamMasker(a.sam_weights, dev_t),
                  "cellpose": lambda: CellposeMasker(a.device)}[mname]()
        od = os.path.join(a.out, mname)
        os.makedirs(os.path.join(od, "masks"), exist_ok=True)
        os.makedirs(os.path.join(od, "overlay"), exist_ok=True)
        cells, tp = [], [0, 0, 0]
        ious, aerr, n_det, n_mask, t_mask = [], [], 0, 0, 0.0
        for k, f in enumerate(files):
            img = to_bgr(imread(f))
            name = os.path.splitext(os.path.basename(f))[0]
            dets = dets_all[f]
            t0 = time.time()
            res = masker(img, dets)
            t_mask += time.time() - t0
            masks, srcs = [r[0] for r in res], [r[1] for r in res]
            lab = build_label_map(img.shape[:2], masks, [d[5] for d in dets])
            imwrite(os.path.join(od, "masks", f"{name}.png"), lab)
            if k < a.overlay:
                imwrite(os.path.join(od, "overlay", f"{name}.png"), overlay(img, lab, dets, srcs))
            for i, (d, m, s) in enumerate(zip(dets, masks, srcs)):
                # 겹침 정리 후 실제로 남은 픽셀 기준으로 형태 계산
                final = (lab == i + 1) if m is not None else None
                cells.append({"image": name, "cell_id": i + 1, "class": NAMES[d[0]], "yolo_conf": round(d[5], 3),
                              "x0": round(d[1], 1), "y0": round(d[2], 1), "x1": round(d[3], 1), "y1": round(d[4], 1),
                              "mask_source": s, **cell_features(final)})
            n_det += len(dets)
            n_mask += sum(s != "none" for s in srcs)
            if a.gt_masks:
                gp = os.path.join(a.gt_masks, f"{name}.png")
                if os.path.exists(gp):
                    gt = imread(gp)
                    # 정답이 RBC만 있으므로 RBC 예측만 비교
                    rbc_ids = [i + 1 for i, d in enumerate(dets) if d[0] == 1]
                    pred = np.where(np.isin(lab, rbc_ids), lab, 0)
                    r = match_masks(pred, gt)
                    for j in range(3):
                        tp[j] += r[j]
                    ious += r[3]
                    aerr += r[4]
        write_csv(os.path.join(od, "cells.csv"), cells)
        s = {"mask_method": mname, "n_images": len(files), "n_cells_detected": n_det,
             "mask_coverage": round(n_mask / max(n_det, 1), 4),
             "ms_per_image_yolo": round(t_yolo / max(len(files), 1) * 1000, 1),
             "ms_per_image_mask": round(t_mask / max(len(files), 1) * 1000, 1)}
        if a.gt_masks:
            P, R = tp[0] / max(tp[0] + tp[1], 1), tp[0] / max(tp[0] + tp[2], 1)
            s.update({"seg_precision": round(P, 4), "seg_recall": round(R, 4),
                      "seg_F1": round(2 * P * R / max(P + R, 1e-9), 4),
                      "mean_IoU_matched": round(float(np.mean(ious)), 4) if ious else 0,
                      "area_rel_error": round(float(np.mean(aerr)), 4) if aerr else 0})
        summary.append(s)
        print(f"\n[{mname}]")
        for kk, vv in s.items():
            print(f"  {kk}: {vv}")
    write_csv(os.path.join(a.out, "summary.csv"), summary)
    print("\n저장:", a.out)


if __name__ == "__main__":
    main()
