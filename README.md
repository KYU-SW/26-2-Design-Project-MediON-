# 26-2 Design Project — MediON

**비염색 혈액 현미경 영상에서 적혈구(RBC)·백혈구(WBC)를 자동으로 찾고 세는 시스템**
건양대학교 의료IT공학과 26-2 설계 프로젝트

```
비염색 영상 → [가상 염색: UNSB] → 염색처럼 보이는 영상 → [탐지: YOLO11s] → RBC·WBC 개수
                                                          └→ [분할: Cellpose / SAM] → 세포별 윤곽·크기
```

## 지금까지 결과 (RedTell 비염색 test 91장, RBC 기준)

| 방법 | 탐지 모델 학습 데이터 | RBC MAE (장당 개수 오차) | F1 |
| --- | --- | --- | --- |
| [참고] 진짜 염색 영상 (TXL-PBC test 126장) | 진짜 염색 | 1.21 | 0.937 |
| 비염색 그대로 (하한) | 진짜 염색 | 9.85 | 0.831 |
| CycleGAN 가상 염색 | 진짜 염색 | 7.77 | 0.768 |
| UNSB 가상 염색 (ep15, 1단계) | 진짜 염색 | 6.23 | 0.818 |
| **A.** UNSB + 신뢰도 기준 0.8 (가상 염색 val로 선택) | 진짜 염색 | 3.52 | 0.829 |
| **B.** UNSB + YOLO 추가 학습 | 진짜 염색 + 가상 염색 RedTell 82장 | 2.48 | 0.954 |
| **C.** 비염색 그대로 + YOLO 추가 학습 (대조) | 진짜 염색 + 비염색 RedTell 82장 | **1.90** | **0.960** |

- **비염색 정답 데이터가 없을 때** (실제 현장에서 흔한 상황): 가상 염색(UNSB) + 신뢰도 기준 조정으로 개수 오차가 9.85 → 3.52로 줄어듦 (A)
- **비염색 정답이 조금이라도 있을 때**: 비염색 영상으로 바로 추가 학습(C)이 가상 염색을 거친 추가 학습(B)보다 개수 오차가 작음 (차이 0.58개, 95% 신뢰구간 0.15~1.01, Wilcoxon p = 0.004). B는 가상 염색이 만든 가짜 세포 때문에 장당 약 1.3개를 더 셈
- 추가 학습한 YOLO도 진짜 염색 영상 성능을 유지함 (TXL test F1 B 0.933 · C 0.923, 원래 0.937)
- 한계: 비염색 WBC 정답이 없어 RBC만 평가, val 9장으로 적음, 염색·비염색 데이터가 서로 다른 영상(짝 없음)

## 폴더

| 폴더 | 내용 | GitHub |
| --- | --- | --- |
| `01. Idea conference/` | 주제 검토 자료 | ○ |
| `02. Project Proposal/` | 프로젝트 총괄 정리, 가상 염색 실험 계획 (docx·pdf) | ○ |
| `03. Data/` | 원본 데이터셋 (TXL-PBC, RedTell 등) | ✕ (라이선스·용량) |
| `04. Prep/` | 전처리 코드와 결과. 사용법은 [`04. Prep/README.md`](04.%20Prep/README.md) | 코드·문서만 |
| `05. Train/` | 학습 — 코드, 실행 노트북, 가중치. 사용법은 [`05. Train/README.md`](05.%20Train/README.md) | 코드·노트북·로그·그래프만 |
| `06. Generate/` | 생성 — 가상 염색한 영상 (모델별). 설명은 [`06. Generate/README.md`](06.%20Generate/README.md) | README만 |
| `07. Evaluate/` | 평가 — 채점 결과 표 (summary·per_image csv, 최종 비교표). 설명은 [`07. Evaluate/README.md`](07.%20Evaluate/README.md) | ○ |

가중치(`.pt`, `.pth`), 데이터에서 나온 영상(가상 염색 결과, 학습 중간 그림), 외부 공식 코드(`external/`)는 올리지 않습니다 (`.gitignore`).

## 실행 순서 (05. Train 노트북)

| 순서 | 노트북 | 내용 |
| --- | --- | --- |
| 1 | `run_train.ipynb` | 탐지 모델 YOLO vs Cellpose 비교, conf 정하기, 하한 |
| 2 | `run_staining.ipynb` | CycleGAN 가상 염색, B 방식 세포 윤곽 |
| 3 | `run_pix2pix.ipynb` | 결과 폴더 정리, 가짜 짝 데이터, pix2pix (보류) |
| 4 | `run_unsb.ipynb` | UNSB 학습 |
| 5 | `run_unsb_finish.ipynb` | UNSB epoch × 변환 단계 수 고르기, 최종 채점 |
| 6 | `run_detector_ab.ipynb` | A (신뢰도 기준) · B (YOLO 추가 학습) |
| 7 | `run_control_c.ipynb` | C (대조: 비염색 그대로 추가 학습) |

평가 규칙: 설정은 val(RedTell dse 84~93, 9장)로만 고르고, test(dse 94~186, 91장)는 방법마다 한 번만 채점. 정답 마스크 오류가 있는 dse 65·88·97·163번은 제외.

## GitHub 올리기

`sync_github.bat` 더블클릭 (노트북은 Ctrl+S로 저장하고 닫은 뒤). 처음 연결 방법은 `GITHUB_SETUP.md`.

## 사용한 공개 데이터·코드

- TXL-PBC (염색 혈액 영상, 박스 정답), RedTell (비염색 적혈구 영상, 마스크 정답)
- Ultralytics YOLO11, Cellpose, SAM 2.1
- CycleGAN·pix2pix 공식 코드 (junyanz/pytorch-CycleGAN-and-pix2pix), UNSB 공식 코드 (cyclomon/UNSB)
