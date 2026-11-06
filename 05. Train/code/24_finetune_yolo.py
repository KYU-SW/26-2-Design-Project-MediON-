"""
24. YOLO 추가 학습(fine-tune): 진짜 염색(TXL-PBC) + 가상 염색한 RedTell 학습 영상

실험 B — "탐지 모델이 가상 염색 영상을 직접 보고 배우면 더 잘 잡나?"
  - 시작점: 진짜 염색으로만 학습한 YOLO (runs/yolo/yolo11s/weights/best.pt)
  - 학습 데이터: TXL-PBC 학습 영상 전체 + UNSB로 가상 염색한 RedTell dse 학습 영상 82장(×--repeat 번 복사)
    → 염색 영상을 계속 같이 보여줘서 원래 실력을 잊지 않게 함
  - 학습 중 검증(best.pt 고르기): 가상 염색한 RedTell val 9장
  - 학습률은 처음 학습보다 낮게 (lr0 0.001), 색상 증강은 처음 학습과 같게

데이터 폴더 (자동 생성): 05. Train/datasets/yolo_ft_<이름>/{images,labels}/{train,val}, data.yaml
출력: 05. Train/runs/yolo/<이름>/weights/best.pt

사용법
  python 24_finetune_yolo.py --prep "D:/05. Blood/04. Prep" --train_root "D:/05. Blood/05. Train" ^
      --vs_train "D:/05. Blood/05. Train/virtual_stained/unsb/ep15_nfe1/train" ^
      --vs_val "D:/05. Blood/05. Train/virtual_stained/unsb/ep15_nfe1/val" --name yolo11s_ft_unsb
"""
import argparse
import glob
import os
import shutil

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--train_root", required=True, help="05. Train 폴더")
    ap.add_argument("--vs_train", required=True, help="가상 염색한 RedTell 학습 영상 폴더")
    ap.add_argument("--vs_val", required=True, help="가상 염색한 RedTell val 영상 폴더")
    ap.add_argument("--name", default="yolo11s_ft_unsb")
    ap.add_argument("--base", default=None, help="시작 가중치 (기본: runs/yolo/yolo11s/weights/best.pt)")
    ap.add_argument("--repeat", type=int, default=4, help="가상 염색 영상을 몇 번 복사해 넣을지 (TXL보다 훨씬 적어서)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    base = a.base or os.path.join(a.train_root, "runs", "yolo", "yolo11s", "weights", "best.pt")
    tr_lab = os.path.join(a.prep, "6_redtell_dse_train", "labels")
    va_lab = os.path.join(a.prep, "4_eval_unstained", "redtell_dse_val", "labels")
    D = os.path.join(a.train_root, "datasets", f"yolo_ft_{a.name}")
    for s in ["train", "val"]:
        for d in ["images", "labels"]:
            os.makedirs(os.path.join(D, d, s), exist_ok=True)
    n_tr = 0
    for f in sorted(glob.glob(os.path.join(a.vs_train, "*.png"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        for r in range(a.repeat):
            shutil.copy(f, os.path.join(D, "images", "train", f"{stem}_r{r}.png"))
            shutil.copy(os.path.join(tr_lab, stem + ".txt"), os.path.join(D, "labels", "train", f"{stem}_r{r}.txt"))
        n_tr += 1
    n_va = 0
    for f in sorted(glob.glob(os.path.join(a.vs_val, "*.png"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        shutil.copy(f, os.path.join(D, "images", "val", stem + ".png"))
        shutil.copy(os.path.join(va_lab, stem + ".txt"), os.path.join(D, "labels", "val", stem + ".txt"))
        n_va += 1
    txl_train = os.path.join(a.prep, "1_detection_train", "images", "train").replace("\\", "/")
    yaml = os.path.join(D, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"train:\n  - {txl_train}\n  - {os.path.join(D, 'images', 'train').replace(chr(92), '/')}\n"
                f"val: {os.path.join(D, 'images', 'val').replace(chr(92), '/')}\n"
                "names:\n  0: WBC\n  1: RBC\n")
    print(f"데이터: 가상 염색 학습 {n_tr}장 × {a.repeat} + TXL 학습 전체, 검증 {n_va}장 → {yaml}")

    model = YOLO(base)
    model.train(data=yaml, epochs=a.epochs, imgsz=640, batch=a.batch, device=a.device, workers=a.workers,
                project=os.path.join(a.train_root, "runs", "yolo"), name=a.name, exist_ok=True, seed=0,
                lr0=0.001, lrf=0.1, warmup_epochs=1, patience=15,
                hsv_h=0.015, hsv_s=0.3, hsv_v=0.2, flipud=0.5, fliplr=0.5)
    print("\n추가 학습 완료:", os.path.join(a.train_root, "runs", "yolo", a.name, "weights", "best.pt"))


if __name__ == "__main__":
    main()
