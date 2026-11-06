"""
21. 학습한 CycleGAN으로 비염색 영상 전체를 가상 염색하기

CycleGAN 학습 자체는 공식 코드(junyanz/pytorch-CycleGAN-and-pix2pix)의 train.py로 합니다 (노트북 참고).
학습은 128×128 조각으로 했지만, 생성기는 크기 제한이 없는 구조라 영상 전체를 한 번에 넣어 변환합니다.
결과 파일 이름은 입력과 같아서 (dse_94.png → dse_94.png), 바로 12_eval_compare.py 로 채점할 수 있습니다.

사용법
  python 21_apply_cyclegan.py --repo "D:/05. Blood/05. Train/external/cyclegan" ^
      --ckpt "D:/05. Blood/05. Train/runs/cyclegan/cyclegan_rbc" --epoch 50 ^
      --images "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse/images" ^
      --out "D:/05. Blood/05. Train/virtual_stained/cyclegan_ep50/test"
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def imread_gray(path):
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)


def load_generator(repo, ckpt, epoch, device):
    sys.path.insert(0, repo)
    from models import networks  # 공식 코드의 생성기 정의를 그대로 사용
    # 학습 기본값과 같게: resnet_9blocks, instance norm, dropout 없음, RGB 3채널
    net = networks.define_G(3, 3, 64, "resnet_9blocks", "instance", False)
    path = os.path.join(ckpt, f"{epoch}_net_G_A.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"가중치가 없습니다: {path}")
    sd = torch.load(path, map_location="cpu")
    if hasattr(sd, "_metadata"):
        del sd._metadata
    net.load_state_dict(sd)
    return net.to(device).eval()


@torch.no_grad()
def stain(net, gray, device):
    """흑백 비염색 영상 → 가상 염색 BGR 영상. 학습 때와 같이 흑백을 3채널로 복사하고 [-1, 1]로 정규화"""
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
    h, w = gray.shape
    ph, pw = (-h) % 4, (-w) % 4  # 생성기는 4의 배수 크기가 필요
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y = net(x)[0, :, :h, :w]
    y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="공식 CycleGAN 코드 폴더")
    ap.add_argument("--ckpt", required=True, help="학습 결과 폴더 (…/runs/cyclegan/<이름>)")
    ap.add_argument("--epoch", default="latest", help="쓸 가중치 epoch (예: 25, 50, latest)")
    ap.add_argument("--images", required=True, help="비염색 영상 폴더")
    ap.add_argument("--out", required=True, help="가상 염색 영상 저장 폴더")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    net = load_generator(a.repo, a.ckpt, a.epoch, a.device)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "*.png")))
    t0 = time.time()
    for f in files:
        imwrite(os.path.join(a.out, os.path.basename(f)), stain(net, imread_gray(f), a.device))
    dt = (time.time() - t0) / max(len(files), 1) * 1000
    print(f"가상 염색 {len(files)}장 완료 (epoch {a.epoch}, 장당 {dt:.0f} ms) → {a.out}")


if __name__ == "__main__":
    main()
