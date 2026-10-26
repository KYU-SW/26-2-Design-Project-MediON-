"""
11. 셀 기반 방식용 RBC/WBC 분류기 학습
Cellpose는 세포 윤곽만 찾고 종류는 구분하지 않으므로, 찾은 세포 조각을 RBC/WBC로 나누는 분류기가 필요합니다.

입력 : 04. Prep/2_cell_crops/{train,val,test}/{RBC,WBC}/  (96x96 세포 조각)
출력 : 05. Train/runs/cell_classifier/best.pt, log.csv

- 모델: ImageNet 사전학습 ResNet-18 (작고 빨라서 조각 분류에 충분)
- WBC가 RBC보다 약 12배 적어서, 학습 때 WBC를 더 자주 뽑도록 가중 샘플링
- 검증 분할의 WBC F1이 가장 높은 가중치를 저장 (WBC를 놓치지 않는 게 중요)

사용법
  python 11_train_cell_classifier.py --prep "D:/05. Blood/04. Prep" --out "D:/05. Blood/05. Train/runs"
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

CLASSES = ["RBC", "WBC"]  # ImageFolder 알파벳 순서와 같음
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_tf(train):
    t = [transforms.Resize((96, 96))]
    if train:
        t += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(180),
              transforms.ColorJitter(brightness=0.2, saturation=0.3, hue=0.02)]  # 약한 색상 증강 (YOLO와 같은 취지)
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


def build_model(pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def evaluate(model, loader, device):
    model.eval()
    cm = torch.zeros(2, 2, dtype=torch.long)  # [정답, 예측]
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).argmax(1).cpu()
            for t, q in zip(y, p):
                cm[t, q] += 1
    tp, fp, fn = cm[1, 1].item(), cm[0, 1].item(), cm[1, 0].item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (cm[0, 0] + cm[1, 1]).item() / max(cm.sum().item(), 1)
    return acc, prec, rec, f1, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2, help="윈도우에서 문제 생기면 0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_pretrained", action="store_true", help="사전학습 가중치 없이 (오프라인 테스트용)")
    ap.add_argument("--limit", type=int, default=0, help="테스트용: 학습 조각 수 제한")
    a = ap.parse_args()
    torch.manual_seed(0)

    root = os.path.join(a.prep, "2_cell_crops")
    tr = datasets.ImageFolder(os.path.join(root, "train"), build_tf(True))
    va = datasets.ImageFolder(os.path.join(root, "val"), build_tf(False))
    te = datasets.ImageFolder(os.path.join(root, "test"), build_tf(False))
    assert tr.classes == CLASSES, tr.classes
    if a.limit:
        tr.samples = tr.samples[: a.limit // 2] + tr.samples[-a.limit // 2:]
        tr.targets = [s[1] for s in tr.samples]

    n = [tr.targets.count(i) for i in range(2)]
    w = [1.0 / n[t] for t in tr.targets]
    sampler = WeightedRandomSampler(w, num_samples=len(tr.targets), replacement=True)
    dl_tr = DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers)
    dl_va = DataLoader(va, batch_size=a.batch, num_workers=a.workers)
    dl_te = DataLoader(te, batch_size=a.batch, num_workers=a.workers)
    print(f"학습 조각: RBC {n[0]}, WBC {n[1]} / 검증 {len(va)} / 테스트 {len(te)} / 장치 {a.device}")

    model = build_model(not a.no_pretrained).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out = os.path.join(a.out, "cell_classifier")
    os.makedirs(out, exist_ok=True)
    best_f1, log = -1, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(a.device), y.to(a.device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        acc, prec, rec, f1, _ = evaluate(model, dl_va, a.device)
        log.append({"epoch": ep, "train_loss": round(tot / len(tr), 4), "val_acc": round(acc, 4),
                    "val_WBC_precision": round(prec, 4), "val_WBC_recall": round(rec, 4), "val_WBC_F1": round(f1, 4)})
        print(log[-1])
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "classes": CLASSES, "mean": MEAN, "std": STD}, os.path.join(out, "best.pt"))

    with open(os.path.join(out, "log.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        wr.writeheader()
        wr.writerows(log)

    ck = torch.load(os.path.join(out, "best.pt"), map_location=a.device)
    model.load_state_dict(ck["model"])
    acc, prec, rec, f1, cm = evaluate(model, dl_te, a.device)
    print(f"\n테스트: 정확도 {acc:.4f}, WBC precision {prec:.4f}, recall {rec:.4f}, F1 {f1:.4f}")
    print("혼동행렬 [정답 RBC/WBC] x [예측 RBC/WBC]:", cm.tolist())
    print("저장:", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
