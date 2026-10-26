"""
02. RedTell(비염색) RBC 크기 측정
dse_data 정답 마스크로 RBC 박스 크기((가로+세로)/2)의 중앙값을 잽니다.
TXL-PBC와 같은 기준(박스 크기)이라 배율 정규화에 바로 씁니다.
결과를 03_prep_data.py 의 REDTELL_RBC_PX 에 넣습니다. (2026-10-22 측정값: 56.5 px)

참고: 원형 검출(Hough)은 세포 테두리까지 잡아서 약 19% 크게 나옵니다 (anemia 64.6 px, dse 67.2 px).

사용법: python 02_measure_redtell_rbc.py <dse_data 폴더>
"""
import glob
import sys

import numpy as np
from PIL import Image
from skimage import measure

dse = sys.argv[1]
sizes = []
for f in sorted(glob.glob(f"{dse}/annotations/*.tif")):
    lab = np.array(Image.open(f))
    H, W = lab.shape
    for p in measure.regionprops(lab):
        y0, x0, y1, x1 = p.bbox
        if y0 > 0 and x0 > 0 and y1 < H and x1 < W:  # 영상 경계에 잘린 세포 제외
            sizes.append(((x1 - x0) + (y1 - y0)) / 2)
print(f"세포 수: {len(sizes)}, RBC 박스 크기 중앙값: {np.median(sizes):.1f} px, IQR: {np.percentile(sizes, [25, 75]).round(1)}")
