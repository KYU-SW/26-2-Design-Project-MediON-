# 06. Generate — 가상 염색 결과 영상

비염색 영상을 가상 염색 모델로 바꾼 결과입니다. **파일 이름이 원본과 같아서** 원본과 1:1로 비교할 수 있습니다 (예: `dse_100.png`).
영상은 데이터에서 나온 것이라 GitHub에는 올리지 않습니다 (이 README만 올라감).

| 폴더 | 모델·설정 | 비고 |
| --- | --- | --- |
| `unsb/ep15_nfe1/{train,val,test}` | **UNSB epoch 15, 변환 1단계 — 현재 최고 (val로 선택)** | train = YOLO 추가 학습(B)에 쓴 82장 |
| `unsb/ep<숫자>_nfe<1·2·3>/val`, `unsb/ep<숫자>/val` | UNSB 다른 epoch·단계 수 (고르기용) | `ep<숫자>/` = 5단계 |
| `cyclegan/ep15/{val,test}` | CycleGAN 1차 (val로 선택) | 가짜 세포 많음 |
| `cyclegan/ep<숫자>/val` | CycleGAN 다른 epoch | ep35부터 망가짐 |

원본 위치
- test 91장: `04. Prep/4_eval_unstained/redtell_dse/images/`
- val 9장: `04. Prep/4_eval_unstained/redtell_dse_val/images/`
- train 82장: `04. Prep/6_redtell_dse_train/images/`
