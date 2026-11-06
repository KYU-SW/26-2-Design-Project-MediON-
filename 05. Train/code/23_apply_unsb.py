"""
23. 학습한 UNSB로 비염색 영상을 가상 염색하기

UNSB는 한 번에 바꾸지 않고 여러 단계(기본 5단계, NFE)에 걸쳐 조금씩 바꿉니다.
공식 코드(cyclomon/UNSB, models/sb_model.py 의 test 부분)의 단계별 샘플링을 그대로 옮겼고,
공식 코드가 GPU 전용(.cuda())이라 장치를 고를 수 있게만 바꿨습니다.

  - 단계 수: --nfe (기본 = 학습 때 num_timesteps, 보통 5). 논문에서는 단계가 너무 많으면 과하게 바뀌는 경우도 있다고 함
  - 변환 방식: 22번과 같이 128 조각 단위로 바꾼 뒤 겹쳐 붙임 (--tile 0 이면 영상 전체 한 번에)
  - 무작위 노이즈가 들어가므로 --seed 로 고정 (같은 seed = 같은 결과)
  결과 파일 이름은 입력과 같음 → 바로 12_eval_compare.py 로 채점

사용법
  python 23_apply_unsb.py --repo "D:/05. Blood/05. Train/external/unsb" ^
      --ckpt "D:/05. Blood/05. Train/runs/unsb/unsb_rbc" --epoch 50 ^
      --images "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse_val/images" ^
      --out "D:/05. Blood/05. Train/virtual_stained/unsb/ep50/val"
"""
import argparse
import glob
import os
import sys
import time
from argparse import Namespace

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


def _val(s):
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(s, s)


def read_opt(ckpt):
    o = {}
    with open(os.path.join(ckpt, "train_opt.txt"), encoding="utf-8") as f:
        for line in f:
            if ":" in line and not line.startswith("-"):
                k, v = line.split(":", 1)
                o[k.strip()] = _val(v.split("[default")[0].strip())
    return Namespace(**o)


def load_generator(repo, ckpt, epoch, device):
    sys.path.insert(0, repo)
    from models import networks
    opt = read_opt(ckpt)
    net = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                            opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, [], opt)
    sd = torch.load(os.path.join(ckpt, f"{epoch}_net_G.pth"), map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}  # 학습 때 DataParallel로 저장된 이름 정리
    net.load_state_dict(sd)
    return net.to(device).eval(), opt


def sb_times(T, device):
    incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return torch.tensor(np.concatenate([np.zeros(1), times])).float().to(device)


@torch.no_grad()
def sb_sample(net, x, opt, nfe, gen):
    """공식 sb_model.py test 부분과 같은 단계별 샘플링. 마지막 단계 결과를 돌려줌"""
    T, tau = opt.num_timesteps, opt.tau
    times = sb_times(T, x.device)
    Xt, Xt_1 = x, None
    for t in range(nfe):
        if t > 0:
            delta = times[t] - times[t - 1]
            denom = times[-1] - times[t - 1]
            inter = delta / denom
            scale = delta * (1 - delta / denom)
            Xt = (1 - inter) * Xt + inter * Xt_1 + (scale * tau).sqrt() * torch.randn(Xt.shape, generator=gen, device=gen.device).to(x.device)
        tidx = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        z = torch.randn((x.shape[0], 4 * opt.ngf), generator=gen, device=gen.device).to(x.device)
        Xt_1 = net(Xt, tidx, z)
    return Xt_1


@torch.no_grad()
def stain(net, opt, gray, nfe, device, tile, stride, gen):
    g = gray.astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(g)[None, None].repeat(1, 3, 1, 1).to(device)   # 학습 때처럼 흑백 → 3채널
    H, W = gray.shape

    def run(t):
        h, w = t.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="reflect")
        return sb_sample(net, t, opt, nfe, gen)[..., :h, :w]

    if tile <= 0:
        y = run(x)[0]
    else:
        pad = tile
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        Hp, Wp = xp.shape[-2:]
        w1 = torch.hann_window(tile, periodic=False, device=device).clamp_min(1e-3)
        win = (w1[:, None] * w1[None, :])[None]
        acc = torch.zeros(3, Hp, Wp, device=device)
        wsum = torch.zeros(1, Hp, Wp, device=device)
        for yy in range(0, Hp - tile + 1, stride):
            for xx in range(0, Wp - tile + 1, stride):
                acc[:, yy:yy + tile, xx:xx + tile] += run(xp[..., yy:yy + tile, xx:xx + tile])[0] * win
                wsum[:, yy:yy + tile, xx:xx + tile] += win
        y = (acc / wsum.clamp_min(1e-6))[:, pad:pad + H, pad:pad + W]
    y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="UNSB 공식 코드 폴더")
    ap.add_argument("--ckpt", required=True, help="학습 결과 폴더 (train_opt.txt 가 있는 곳)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=0, help="변환 단계 수 (0 = 학습 때 num_timesteps)")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    net, opt = load_generator(a.repo, a.ckpt, a.epoch, a.device)
    nfe = a.nfe or opt.num_timesteps
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "*.png")))
    t0 = time.time()
    for f in files:
        imwrite(os.path.join(a.out, os.path.basename(f)), stain(net, opt, imread_gray(f), nfe, a.device, a.tile, a.stride, gen))
    dt = (time.time() - t0) / max(len(files), 1) * 1000
    print(f"UNSB 가상 염색 {len(files)}장 완료 (epoch {a.epoch}, NFE {nfe}, 조각 {a.tile}, 장당 {dt:.0f} ms) → {a.out}")


if __name__ == "__main__":
    main()
