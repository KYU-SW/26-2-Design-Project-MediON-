# 05. Train — 검출·계수 모델 학습과 비교 (평가 0)

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
pip install ultralytics cellpose opencv-python
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
  --out "D:/05. Blood/05. Train/results/eval0_stained_test"
```

- 결과: `results/eval0_stained_test/summary.csv`, `by_source.csv`, `per_image.csv`
- Cellpose는 처음 실행할 때 모델 가중치를 자동으로 내려받습니다.

### (선택) ④ 하한 측정 — 비염색 영상을 가상 염색 없이 그대로 넣기

```bash
python 12_eval_compare.py --prep "D:/05. Blood/04. Prep" ^
  --images "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse/images" ^
  --labels "D:/05. Blood/04. Prep/4_eval_unstained/redtell_dse/labels" ^
  --yolo "D:/05. Blood/05. Train/runs/yolo/yolo11s/weights/best.pt" ^
  --classifier "D:/05. Blood/05. Train/runs/cell_classifier/best.pt" ^
  --out "D:/05. Blood/05. Train/results/eval2_unstained_no_staining"
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
