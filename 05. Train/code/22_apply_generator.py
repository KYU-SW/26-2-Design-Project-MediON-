"""
22. 학습한 가상 염색 생성기로 비염색 영상을 가상 염색하기 (공식 CycleGAN/pix2pix 코드로 학습한 모델 공통)

21번과 다른 점
  - 학습 설정 파일(train_opt.txt)을 읽어 생성기 구조(netG, norm, 입력 채널 수)를 자동으로 맞춤
    → CycleGAN(G_A), pix2pix(G) 둘 다 이 스크립트 하나로
  - 기본은 "조각 단위 변환": 학습 때와 같은 128×128 조각으로 잘라 변환하고, 겹친 부분은 부드럽게 섞어 붙임
    (영상 전체를 한 번에 넣으면 학습 때와 조건이 달라 가장자리 줄무늬 등이 생길 수 있음. --tile 0 이면 전체 한 번에)
  결과 파일 이름은 입력과 같음 (dse_94.png → dse_94.png) → 바로 12_eval_compare.py 로 채점

사용법
  python 22_apply_generator.py --repo "D:/05. Blood/05. Train/external/cyclegan" ^
      --ckpt "D:/05. Blood/05. Train/runs/pix2pix/pix2pix_rbc" --epoch 50 ^
      --images "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse_val/images" ^
      --out "D:/05. Blood/05. Train/virtual_stained/pix2pix/ep50/val"
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
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)


def read_opt(ckpt):
    opt = {}
    with open(os.path.join(ckpt, "train_opt.txt"), encoding="utf-8") as f:
        for line in f:
            if ":" in line and not line.startswith("-"):
                k, v = line.split(":", 1)
                opt[k.strip()] = v.split("[default")[0].strip()
    return opt


def load_generator(repo, ckpt, epoch, device):
    sys.path.insert(0, repo)
    from models import networks
    o = read_opt(ckpt)
    model = o.get("model", "cycle_gan")
    in_nc, out_nc = int(o.get("input_nc", 3)), int(o.get("output_nc", 3))
    net = networks.define_G(in_nc, out_nc, int(o.get("ngf", 64)), o["netG"], o.get("norm", "instance"),
                            o.get("no_dropout", "True") != "True")
    fname = f"{epoch}_net_G_A.pth" if model == "cycle_gan" else f"{epoch}_net_G.pth"
    sd = torch.load(os.path.join(ckpt, fname), map_location="cpu")
    if hasattr(sd, "_metadata"):
        del sd._metadata
    net.load_state_dict(sd)
    mult = 128 if o["netG"] == "unet_128" else (256 if o["netG"] == "unet_256" else 4)
    info = f"{model}, {o['netG']}, norm {o.get('norm')}, 입력 {in_nc}채널"
    return net.to(device).eval(), in_nc, mult, info


@torch.no_grad()
def run(net, x, mult):
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return net(x)[..., :h, :w]


@torch.no_grad()
def stain(net, gray, in_nc, mult, device, tile, stride):
    g = gray.astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(g)[None, None].to(device)
    if in_nc == 3:
        x = x.repeat(1, 3, 1, 1)
    H, W = gray.shape
    if tile <= 0:
        y = run(net, x, mult)[0]
    else:
        # 겹치는 조각으로 변환 후, 조각 가운데일수록 큰 가중치로 섞기
        pad = tile
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        Hp, Wp = xp.shape[-2:]
        w1 = torch.hann_window(tile, periodic=False, device=device).clamp_min(1e-3)
        win = (w1[:, None] * w1[None, :])[None]
        acc = torch.zeros(3, Hp, Wp, device=device)
        wsum = torch.zeros(1, Hp, Wp, device=device)
        for yy in range(0, Hp - tile + 1, stride):
            for xx in range(0, Wp - tile + 1, stride):
                out = run(net, xp[..., yy:yy + tile, xx:xx + tile], mult)[0]
                acc[:, yy:yy + tile, xx:xx + tile] += out * win
                wsum[:, yy:yy + tile, xx:xx + tile] += win
        y = (acc / wsum.clamp_min(1e-6))[:, pad:pad + H, pad:pad + W]
    y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="공식 CycleGAN/pix2pix 코드 폴더")
    ap.add_argument("--ckpt", required=True, help="학습 결과 폴더 (train_opt.txt 가 있는 곳)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=128, help="조각 크기 (학습 조각과 같게). 0 = 영상 전체 한 번에")
    ap.add_argument("--stride", type=int, default=64, help="조각 간격 (작을수록 이음새가 부드럽고 느림)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    net, in_nc, mult, info = load_generator(a.repo, a.ckpt, a.epoch, a.device)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "*.png")))
    t0 = time.time()
    for f in files:
        imwrite(os.path.join(a.out, os.path.basename(f)), stain(net, imread_gray(f), in_nc, mult, a.device, a.tile, a.stride))
    dt = (time.time() - t0) / max(len(files), 1) * 1000
    print(f"가상 염색 {len(files)}장 완료 ({info}, epoch {a.epoch}, 조각 {a.tile}, 장당 {dt:.0f} ms) → {a.out}")


if __name__ == "__main__":
    main()
