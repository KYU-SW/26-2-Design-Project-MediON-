# 04. Prep — 학습용 데이터 (전처리 완료)

학습에 바로 넣을 수 있게 손질한 데이터입니다. **모델 학습은 아직 안 했습니다.**

## 무엇을 할 때 어떤 폴더를 쓰나

```
비염색 영상 → [가상 염색 모델] → 가상 염색 영상 → [검출·계수 모델] → RBC·WBC 개수
                  ↑ 3번으로 학습                        ↑ 1번(+2번)으로 학습        ↑ 4번으로 채점
```

| 하려는 일 | 쓰는 폴더 | 내용 |
| --- | --- | --- |
| 검출·계수 모델 학습 (YOLO) | `1_detection_train/` | 염색 영상 + RBC·WBC 박스 정답 |
| 검출·계수 모델 학습 (셀 기반, Cellpose) | `1_detection_train/` + `2_cell_crops/` | Cellpose 뒤에 붙는 RBC/WBC 분류기 학습용 세포 조각 |
| 가상 염색 모델 학습 (CycleGAN·UNSB) | `3_virtual_staining/rbc_only/` | 비염색 조각(A) + 염색 조각(B). **지금은 이것** |
| 〃 (짝 데이터 받은 뒤) | `3_virtual_staining/all/` | WBC 포함 버전 |
| 최종 채점 | `4_eval_unstained/` | 실제 비염색 영상 93장 + RBC 정답 4,022개 |
| 기록 확인 | `meta/` | 배율 보정 비율, 영상별 출처·분할 목록 |
| 데이터 다시 만들기 | `code/` | 전처리 코드 |

## 폴더 구조

```
04. Prep/
├── README.md
├── code/
│   ├── 01_inspect_txl.py          TXL-PBC 통계 확인
│   ├── 02_measure_redtell_rbc.py  RedTell 세포 크기 측정 (56.5 px)
│   ├── 03_prep_data.py            전처리 본체 (아래 1~4번 폴더와 meta 생성)
│   ├── 04_check_outputs.py        결과 개수 확인 + 미리보기
│   └── 05_reorganize_folders.py   예전 zip 풀어둔 폴더를 이 구조로 정리 (한 번만)
├── 1_detection_train/
│   ├── images/{train,val,test}/   882 / 252 / 126장
│   ├── labels/{train,val,test}/   YOLO 형식, 0 = WBC, 1 = RBC
│   ├── data.yaml                  YOLO 설정 파일
│   └── splits/test_<출처>.txt     출처별 테스트 목록
├── 2_cell_crops/{train,val,test}/{RBC,WBC}/   세포 조각 96×96
├── 3_virtual_staining/
│   ├── rbc_only/  trainA 4,926 · trainB 1,378 · testA 2,779 · testB 536  (128×128)
│   └── all/       trainA 4,300 · trainB 3,730 · testA 837 · testB 1,090  (256×256)
├── 4_eval_unstained/redtell_dse/
│   ├── images/    비염색 영상 (가상 염색 모델 입력)
│   ├── labels/    정답 박스 (YOLO 형식, 1 = RBC)
│   ├── masks/     정답 마스크 (16bit, 셀 기반 평가용)
│   └── counts.csv 영상별 정답 개수
└── meta/
```

3번 폴더의 A는 비염색, B는 염색입니다 (CycleGAN 규칙). 비염색 조각은 흑백이고, 학습 코드가 읽을 때 자동으로 3채널로 바꿉니다.

## 데이터 출처와 나눈 방법

| 데이터 | 어디에 썼나 |
| --- | --- |
| TXL-PBC 학습 분할 | 1번 train, 2번 train, 3번 trainB |
| TXL-PBC 검증 분할 | 1번 val, 2번 val, 3번 testB |
| TXL-PBC 테스트 분할 | 1번 test, 2번 test |
| RedTell anemia 79장 | 3번 trainA |
| RedTell dse 1~93번 | 3번 trainA |
| RedTell dse 94~186번 | 3번 testA, 4번 (채점용, 학습에 안 씀) |

## 전처리 내용

| 처리 | 내용 | 이유 |
| --- | --- | --- |
| 클래스 정리 | WBC, RBC만 남기고 혈소판 제외 | 프로젝트 목표가 RBC·WBC 계수 |
| 배율 맞추기 | 모든 영상의 RBC 크기를 64 px로 통일 | 출처마다 RBC 크기가 56~130 px로 다름 |
| 비염색 밝기 보정 | 배경 밝기 얼룩 제거 | 영상 한쪽이 어두운 불균일 조명 |
| 비염색 채널 | 명시야만 사용, 형광 제외 | 형광은 염료를 써서 비염색이 아님 |
| 빈 조각 제거 | 세포가 거의 없는 비염색 조각 제외 (`rbc_only`) | 배경만 학습하는 것 방지 |
| WBC 조각 제외 | WBC가 걸친 염색 조각 제외 (`rbc_only`) | 비염색에 WBC가 없어 GAN이 없는 WBC를 그려 넣는 것 방지 |

| 출처 | 원래 RBC 크기 | 보정 비율 |
| --- | --- | --- |
| BCCD | 99.5 px | 0.643 |
| BCDD | 59.0 px | 1.085 |
| PBC | 75.0 px | 0.853 |
| Raabin-WBC | 129.7 px | 0.493 |
| RedTell (anemia, dse) | 56.5 px | 1.133 |

## 알려진 한계

- dse는 환자 정보가 없어 영상 번호로만 나눴습니다. 같은 사람 영상이 학습·평가에 섞였을 수 있습니다.
- 일부 비염색 영상(예: dse 101번)은 어두운 얼룩이 다 지워지지 않습니다.
- RedTell은 도말이 아니라 용액 속 세포로 보여, 일반 도말 영상과 차이가 있습니다.
- 실제 비염색 WBC 정답은 없습니다. 짝 데이터를 받으면 추가합니다.
- `3_virtual_staining/rbc_only`의 염색 조각은 대부분 BCCD 출처라 가상 염색 색 톤이 BCCD를 따라갈 수 있습니다.

## 데이터를 다시 만들 때

```bash
pip install opencv-python numpy openpyxl pillow scikit-image
python code/03_prep_data.py --txl    "D:/05. Blood/02. Data/3. TXL-PBC/TXL-PBC" \
                            --anemia "D:/05. Blood/02. Data/6. RedTell/anemia_data" \
                            --dse    "D:/05. Blood/02. Data/6. RedTell/dse_data" \
                            --out    "D:/05. Blood/04. Prep"
python code/04_check_outputs.py "D:/05. Blood/04. Prep"
```

경로는 실제 위치에 맞게 바꿔 주세요. 한글 경로도 읽고 쓸 수 있게 되어 있습니다.

## 총괄 정리 문서 기준 진행 상황

| 단계 | 상태 |
| --- | --- |
| 3-1 검출·계수 모델: 데이터 준비 | 완료 (1, 2번 폴더) |
| 3-2 가상 염색 모델: 데이터 준비 | 완료 (3번 폴더) |
| 3-4 데이터 분할 | 완료 (meta) |
| 4. 평가용 정답 데이터 | 완료 (4번 폴더, RBC만) |
| 3-1 검출 모델 선택·학습 | **다음 할 일** |
| 3-2 가상 염색 모델 학습 (CycleGAN → UNSB) | **다음 할 일** |
| 평가 0 · ① · ② 실행 | 모델 학습 후 |
