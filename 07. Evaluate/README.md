# 07. Evaluate — 채점 결과

모든 채점은 `05. Train/code/12_eval_compare.py` 하나로 같은 규칙으로 했습니다 (예측 박스와 정답 박스가 IoU 0.5 이상이면 맞음).
폴더마다 `summary.csv`(전체), `per_image.csv`(영상별), `by_source.csv`(출처별)가 있습니다.

| 폴더 | 내용 | 데이터 |
| --- | --- | --- |
| `1_detection/` | 탐지 모델 비교 (YOLO vs Cellpose), conf 정하기 (`val_yolo_conf*` = 고르기, `eval0_*` = test) | 진짜 염색 TXL-PBC |
| `2_no_staining/` | 가상 염색 없이 비염색 그대로 (하한) | RedTell |
| `3_staining/<모델>/` | 가상 염색 모델별. 이름 `ep<숫자>[_nfe<단계>]_<val|test>` | RedTell |
| `4_cell_mask/` | B 방식 세포 윤곽 (Cellpose / SAM / 타원) | RedTell |
| `5_secondary_border/` | 보조 지표: 테두리에 걸친 세포 빼고 채점 | RedTell test |
| `6_detector/` | 탐지 모델 개선 A (`base_*`) · B (`ft_*`) · C (`raw_*`), `selected_conf.json` | RedTell, TXL |
| `final_compare_main.csv`, `final_compare_border6.csv` | 하한 · CycleGAN · UNSB 최종 비교 | RedTell test |
| `6_detector/final_compare.csv`, `compare_B_vs_C.csv` | A · B · C 최종 비교 | RedTell test, TXL test |

규칙: 설정은 `_val`(9장)로만 고르고 `_test`(91장)는 방법마다 한 번만 채점. 고른 값은 `3_staining/unsb/selected.json`, `6_detector/selected_conf.json`.
