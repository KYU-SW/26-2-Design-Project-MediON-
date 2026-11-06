# 05. Train — 검출·계수 모델, 가상 염색, 세포 단위 분할

| 노트북 | 내용 |
| --- | --- |
| `run_train.ipynb` | 검출·계수 모델 학습과 비교 (평가 0), conf 정하기, 하한 측정 — **완료** |
| `run_staining.ipynb` | CycleGAN 1차 가상 염색 학습·평가, B 방식 세포 윤곽 — 완료 (아래 5장) |
| `run_pix2pix.ipynb` | 결과 폴더 정리 + 가짜 짝 만들기 + pix2pix 가상 염색 (아래 6장) — 가짜 짝 규칙 v2 확인 중 |
| `run_unsb.ipynb` | UNSB 가상 염색 학습 (아래 7장) — 완료 |
| `run_unsb_finish.ipynb` | UNSB epoch × 변환 단계 수(NFE) 고르기 + 최종 채점 — 완료 |
| `run_detector_ab.ipynb` | 탐지 모델 개선 A (신뢰도 기준) · B (YOLO 추가 학습) — 완료 (아래 8장) |
| `run_control_c.ipynb` | 대조 실험 C (비염색 그대로 YOLO 추가 학습) — 완료 (아래 8장) |

## 폴더 구조

```
05. Train/
├── code/                  스크립트 (번호 = 하는 일 순서)
│   ├── 00_organize_folders.py   결과 폴더 정리 (한 번만, 옮기기만)
│   ├── 10~12                    검출 모델 학습·비교
│   ├── 13_cell_masks.py         B 방식 세포 윤곽
│   ├── 20~23                    가상 염색 (val 분할, 적용: 21·22 CycleGAN/pix2pix, 23 UNSB)
│   ├── 24_finetune_yolo.py      YOLO 추가 학습 (실험 B·C)
├── external/cyclegan/     CycleGAN·pix2pix 공식 코드 (GitHub에 안 올림)
├── external/unsb/         UNSB 공식 코드 (GitHub에 안 올림)
├── runs/                  학습 가중치 — 모델별
│   ├── yolo/  cell_classifier/  cyclegan/  pix2pix/  unsb/
├── datasets/              YOLO 추가 학습용 데이터 (자동 생성, GitHub에 안 올림)
├── virtual_stained/       가상 염색한 영상 — 모델별/epoch별 (GitHub에 안 올림)
│   ├── cyclegan/ep15/{val,test}   pix2pix/ep…/{val,test}
├── results/               채점 결과 (summary.csv, per_image.csv …)
│   ├── 1_detection/       검출 모델 비교·conf 정하기 (평가 0)
│   ├── 2_no_staining/     가상 염색 없이 비염색 그대로 (하한)
│   ├── 3_staining/        가상 염색 모델별 (cyclegan/, pix2pix/ …), 이름 = ep<숫자>_<val|test>
│   ├── 4_cell_mask/       B 방식 세포 윤곽
│   ├── 5_secondary_border/ 보조 지표 (테두리 세포 제외)
│   └── 6_detector/        탐지 모델 개선 A·B·C
└── run_*.ipynb            실행 노트북
```

## 지금까지 결정한 것

- 검출·계수: **YOLO (yolo11s), conf 0.7** — conf는 val(252장)에서 RBC F1 최고값으로 선택
  - 평가 0 (실제 염색 test 126장): RBC MAE 1.21, RBC F1 0.937, WBC F1 1.000 (Cellpose+분류기: RBC MAE 2.60)
- 세포 단위 결과: **B 방식** — YOLO 박스 안에서 Cellpose / SAM으로 윤곽 (교수님 의견: 추후 세포 단위 분석)
- 가상 염색: CycleGAN(기준선) → UNSB(메인), 총괄 정리 문서 6장
- RedTell dse 정답 오류 영상 제외: 97·163번(test), 88번(val). 마스크가 영상과 맞지 않음 (세포 약 50개인 영상에 255개가 겹쳐 있음)

## 검출·계수 모델 비교 (평가 0)

두 방식을 같은 데이터로 학습하고, **같은 테스트 영상·같은 지표**로 비교합니다.

| 방식 | 학습하는 것 | 코드 |
| --- | --- | --- |
| 박스 기반 | YOLO (세포 찾기 + RBC/WBC 구분을 한 번에) | `10_train_yolo.py` |
| 셀 기반 | Cellpose(세포 윤곽 찾기, 학습 없음) + RBC/WBC 분류기 | `11_train_cell_classifier.py` |
| 비교 | 두 방식을 같은 테스트셋에서 채점 | `12_eval_compare.py` |

## 0. 설치 (처음 한 번)

Anaconda Prompt에서:

```bash
conda create -n blood python=3.11 -y
conda activate blood
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics cellpose opencv-python pandas jupyterlab ipykernel
```

- `cu128`(CUDA 12.8)은 **RTX 50 시리즈(예: RTX 5060 Ti)에 필요**합니다. `cu124` 이하로 설치하면 `no kernel image is available` 에러가 납니다.
- `nvidia-smi` 오른쪽 위 CUDA Version이 12.8보다 낮으면 NVIDIA 드라이버를 먼저 업데이트하세요.
- 확인: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"` → `True`와 목록에 `sm_120`이 나와야 합니다.

## 1. 실행 순서

아래 명령은 모두 `conda activate blood` 후, `D:\05. Blood\05. Train\code` 폴더에서 실행합니다.

```bash
cd "D:\05. Blood\05. Train\code"
```

### ① YOLO 학습 (수십 분 ~ 1시간대)

```bash
python 10_train_yolo.py --prep "D:/05. Blood/04. Prep" --out "D:/05. Blood/05. Train/runs"
```

- 결과: `runs/yolo/yolo11s/weights/best.pt`
- 끝나면 테스트 mAP50 / mAP50-95가 출력됩니다. TXL-PBC 논문 기준선은 YOLOv8s mAP50 0.978입니다.
- 기본 모델은 `yolo11s`(논문 기준선과 같은 크기)입니다. 최신 모델로 바꾸려면 `--model yolo26s.pt`

### ② 셀 기반 분류기 학습 (수 분 ~ 십수 분)

```bash
python 11_train_cell_classifier.py --prep "D:/05. Blood/04. Prep" --out "D:/05. Blood/05. Train/runs"
```

- 결과: `runs/cell_classifier/best.pt`, `log.csv`
- Cellpose 자체는 학습하지 않습니다 (미리 학습된 모델을 그대로 씀).

### ③ 두 방식 비교 (평가 0)

```bash
python 12_eval_compare.py --prep "D:/05. Blood/04. Prep" ^
  --yolo "D:/05. Blood/05. Train/runs/yolo/yolo11s/weights/best.pt" ^
  --classifier "D:/05. Blood/05. Train/runs/cell_classifier/best.pt" ^
  --out "D:/05. Blood/05. Train/results/1_detection/eval0_stained_test"
```

- 결과: `results/1_detection/eval0_stained_test/summary.csv`, `by_source.csv`, `per_image.csv`
- Cellpose는 처음 실행할 때 모델 가중치를 자동으로 내려받습니다.

### (선택) ④ 하한 측정 — 비염색 영상을 가상 염색 없이 그대로 넣기

```bash
python 12_eval_compare.py --prep "D:/05. Blood/04. Prep" ^
  --images "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse/images" ^
  --labels "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse/labels" ^
  --yolo "D:/05. Blood/05. Train/runs/yolo/yolo11s/weights/best.pt" ^
  --classifier "D:/05. Blood/05. Train/runs/cell_classifier/best.pt" ^
  --out "D:/05. Blood/05. Train/results/2_no_staining/eval2_unstained_no_staining"
```

염색 영상으로만 학습한 모델에 비염색을 그대로 넣으면 성능이 얼마나 떨어지는지 보는 기준선입니다 (문서 4-3 "가상 염색 없음"). 이 영상은 RBC 정답만 있어서 WBC 지표는 의미가 없습니다.

## 2. 지표 읽는 법 (`summary.csv`)

| 열 | 뜻 | 좋은 방향 |
| --- | --- | --- |
| `RBC_MAE`, `WBC_MAE` | 이미지 1장당 개수가 평균 몇 개 틀렸나 (메인 지표) | 낮을수록 |
| `RBC_rel_error`, `WBC_rel_error` | 정답 개수 대비 몇 % 틀렸나 | 낮을수록 |
| `*_precision` | 찾은 것 중 맞은 비율 (잘못 찾은 게 적은가) | 높을수록 |
| `*_recall` | 정답 중 찾아낸 비율 (놓친 게 적은가) | 높을수록 |
| `*_F1` | precision과 recall의 균형 | 높을수록 |
| `ms_per_image` | 이미지 1장 처리 시간 | 낮을수록 |

두 방식 모두 **같은 기준**(예측 박스와 정답 박스의 IoU 0.5 이상이면 맞음)으로 채점합니다. Cellpose는 세포 마스크를 감싸는 박스로 바꿔서 비교합니다.

`by_source.csv`에서 출처(BCCD·BCDD·PBC·Raabin-WBC)별 성능도 볼 수 있습니다. 염색 톤이 다른 출처에서 성능 차이가 크면, 가상 염색 결과의 색 차이에도 약할 수 있다는 신호입니다.

## 3. 결과 보내줄 것

학습이 끝나면 아래를 보내주세요. 해석과 다음 단계(가상 염색 학습)를 이어서 진행합니다.

1. ①이 끝날 때 출력된 테스트 mAP50, mAP50-95
2. ②가 끝날 때 출력된 테스트 결과 (정확도, WBC F1)
3. ③의 `summary.csv`, `by_source.csv`

## 4. 자주 나는 문제

| 증상 | 해결 |
| --- | --- |
| `CUDA out of memory` | ①은 `--batch 8`, ②는 `--batch 64` |
| 윈도우에서 데이터 로더 에러 (`BrokenPipe`, `spawn`) | `--workers 0` 추가 |
| `torch.cuda.is_available()`가 `False` | 0번 설치에서 PyTorch를 CUDA 버전으로 다시 설치 |
| `no kernel image is available` / `sm_120 is not compatible` | RTX 50 시리즈인데 PyTorch가 옛 CUDA용. `pip uninstall -y torch torchvision` 후 `cu128`로 다시 설치 |
| Cellpose가 세포를 너무 크게/작게 잡음 (Cellpose 3일 때) | `--diameter` 값을 조절 (기본 64) |
| CycleGAN 학습 시 `Can't pickle` / `BrokenPipe` | `--num_threads 0` (노트북 명령에 이미 들어 있음) |
| CycleGAN 학습이 중간에 끊김 | 같은 명령 끝에 `--continue_train --epoch_count <다음 epoch>` |

## 5. 가상 염색 + 세포 단위 분할 (`run_staining.ipynb`)

```
비염색 → [CycleGAN] → 가상 염색 → [YOLO conf 0.7] → RBC 개수 (평가 ②)
                                    └→ [Cellpose / SAM] → 세포별 윤곽·면적·원형도 (B 방식)
```

| 코드 | 하는 일 |
| --- | --- |
| `20_make_staining_val.py` | (한 번만) dse 84~93번을 가상 염색 학습에서 빼서 val로, 정답 오류 영상 제외 |
| CycleGAN 공식 코드 `train.py` | 학습. `external/cyclegan`에 받음 (junyanz/pytorch-CycleGAN-and-pix2pix, 커밋 `2a7afba`) |
| `21_apply_cyclegan.py` | 학습한 생성기로 비염색 영상 전체를 가상 염색 |
| `12_eval_compare.py` | 가상 염색 영상을 YOLO로 채점 (기존과 같음) |
| `13_cell_masks.py` | B 방식: YOLO 박스 → Cellpose / SAM / 타원 윤곽, 세포별 형태 지표, 정답 마스크로 정확도 |

**데이터 나눔**

| 용도 | 데이터 | 장수 |
| --- | --- | --- |
| 가상 염색 학습 | `3_virtual_staining/rbc_only` (A 비염색 RedTell, B 염색 TXL-PBC) | dse 84~93 제외 |
| epoch 고르기 (val) | `4_eval_unstained/redtell_dse_val` (dse 84~93, 88 제외) | 9장 |
| 최종 채점 (test) | `4_eval_unstained/redtell_dse` (dse 94~186, 97·163 제외) | 91장 |

**CycleGAN 설정**: 공식 기본값(ResNet 9블록 생성기, 70×70 PatchGAN 판별기, LSGAN, cycle 손실 10, identity 0.5, Adam lr 2e-4, batch 1) + 128×128 조각, `--preprocess none`, **50 epoch** (25 + 감소 25, 공식 기본 200은 시간상 줄임), 5 epoch마다 저장.

**epoch 고르는 규칙**: val RBC F1이 가장 높은 epoch. test는 고른 뒤 한 번만 채점.

**B 방식 지표** (`results/4_cell_mask/cellmask_*/summary.csv`, 정답 마스크는 RBC만)

| 열 | 뜻 |
| --- | --- |
| `mask_coverage` | YOLO가 찾은 세포 중 윤곽이 붙은 비율 |
| `seg_precision`, `seg_recall`, `seg_F1` | 예측 윤곽과 정답 윤곽이 IoU 0.5 이상이면 맞음 (세포 단위) |
| `mean_IoU_matched` | 맞은 세포들의 윤곽 정확도 |
| `area_rel_error` | 맞은 세포들의 면적 오차 비율 (크기 측정 정확도) |

`cells.csv`: 세포 1개 = 1줄 (클래스, 신뢰도, 박스, 면적, 둘레, 원형도, 지름, 장·단축, 마스크 출처)

## 6. pix2pix 가상 염색 — 가짜 짝 데이터 (`run_pix2pix.ipynb`)

짝이 필요한 pix2pix(선행연구 Chen 2022·Fanous 2022가 쓴 방식)를 쓰기 위해, 실제 염색 조각을 규칙으로 비염색처럼 바꿔 가짜 짝을 만듭니다.

| 코드 | 하는 일 |
| --- | --- |
| `00_organize_folders.py` | (한 번만) results·virtual_stained 를 단계별·모델별로 정리, 노트북 경로 갱신. 지우는 것 없음 |
| `04. Prep/code/06_make_fake_pairs.py` | 가짜 짝 만들기 → `04. Prep/5_paired_synthetic/rbc_only/{A,B,AB}/{train,val}` |
| CycleGAN 공식 코드 `train.py --model pix2pix` | pix2pix 학습 |
| `22_apply_generator.py` | 학습한 생성기로 가상 염색 (CycleGAN·pix2pix 공통, 128 조각 단위로 변환 후 겹쳐 붙임) |

**pix2pix 설정**: 공식 pix2pix(조건부 GAN + L1 100, U-Net, PatchGAN) + `--input_nc 1`(흑백 입력), `--netG unet_128`, `--norm instance`, 100 epoch(50 + 감소 50), 5 epoch마다 저장.

**가짜 비염색 확인**: TXL 검증 영상을 가짜 비염색으로 바꿔 YOLO로 채점합니다 (실제 비염색 RedTell은 RBC recall 약 0.72).
- 규칙 v1: 252장 기준 recall 0.15. 세포가 선 그림처럼 나오고 겹친 세포가 한 덩어리가 됨 → 사용 안 함 (`5_paired_synthetic/rbc_only/`에 보관)
- 규칙 v2: 입체감(한쪽 어둡고 반대쪽 밝은 가장자리) + 세포 분리 추가 (`rbc_only_v2/`). 시험용 14장 기준 전체 recall 약 0.17이지만 세포가 드문 BCDD 영상은 1.0 → 빽빽한 BCCD 영상이 특히 어려움. 출처별 표로 확인

**고르는 규칙·채점**은 CycleGAN과 같습니다 (val 9장 RBC F1 최고 epoch → test 91장 한 번).

## 7. UNSB 가상 염색 — 메인 기법 (`run_unsb.ipynb`)

| 코드 | 하는 일 |
| --- | --- |
| UNSB 공식 코드 `train.py` | 학습. `external/unsb`에 받음 (cyclomon/UNSB, 커밋 `d1f644f`) |
| `23_apply_unsb.py` | 학습한 UNSB로 가상 염색 (공식 5단계 샘플링 그대로, 128 조각 단위, seed 고정) |

**UNSB 설정**: 공식 horse2zebra 설정(`--lambda_SB 1.0 --lambda_NCE 1.0`, 5단계, tau 0.01) + 128 조각, 50 epoch(25 + 감소 25, 공식 기본 400은 시간상 줄임), 5 epoch마다 저장, `--display_id 0`(visdom 끔), `--num_threads 0`.
공식 코드는 PyTorch 1.9 기준이지만 PyTorch 2.x에서 학습·변환이 되는 것을 확인했습니다 (CPU에서 짧게 시험).

UNSB 결과 (`run_unsb_finish.ipynb`): val에서 epoch 15 · NFE 1 선택. NFE를 줄일수록 가짜 세포가 줄어 precision이 오름 (val 평균 F1 NFE 1: 0.833 → NFE 5: 0.792). test RBC MAE 6.23, F1 0.818.

## 8. 탐지 모델 개선 — A · B · C (`run_detector_ab.ipynb`, `run_control_c.ipynb`)

| 실험 | 탐지 모델 | 입력 | conf (val로 선택) | RBC MAE | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 기준 | 원래 YOLO (진짜 염색으로만 학습) | UNSB 가상 염색 | 0.7 (염색 val) | 6.23 | 0.763 | 0.881 | 0.818 |
| A | 원래 YOLO | UNSB 가상 염색 | 0.8 | 3.52 | 0.815 | 0.844 | 0.829 |
| B | YOLO + 가상 염색 RedTell 82장으로 추가 학습 | UNSB 가상 염색 | 0.7 | 2.48 | 0.939 | 0.969 | 0.954 |
| C (대조) | YOLO + 비염색 RedTell 82장으로 추가 학습 | 비염색 그대로 | 0.8 | **1.90** | 0.962 | 0.958 | **0.960** |

- 추가 학습: `24_finetune_yolo.py` — 원래 YOLO에서 시작, TXL 학습 전체 + RedTell dse 1~83 (65 제외, 82장 × 4번 복사), 30 epoch, lr0 0.001. 정답은 `04. Prep/code/07_make_dse_train.py`로 만듦. B와 C는 입력 영상만 다르고 나머지 조건은 같음
- 진짜 염색 TXL test에서도 성능 유지: F1 원래 0.937 · B 0.933 · C 0.923
- B vs C (test 91장, 장당 오차 짝 비교): C가 0.58개 더 정확 (부트스트랩 95% 신뢰구간 0.15~1.01, Wilcoxon p = 0.004). B는 평균 +1.25개 더 셈 (가상 염색이 만든 가짜 세포)
- 해석: **비염색 정답이 없으면** 가상 염색 + conf 조정(A)이 하한보다 개수 오차를 크게 줄임 (9.85 → 3.52). **정답이 조금이라도 있으면** 비염색으로 바로 추가 학습(C)이 더 정확함. 가상 염색은 사람이 보기 쉬운 영상을 주는 역할은 그대로 남음
- B·C는 원래 계획(탐지 모델은 진짜 염색으로만 학습)을 바꾼 추가 실험이라 보고서에 따로 적음
