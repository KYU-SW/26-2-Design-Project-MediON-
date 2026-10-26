"""
05. 이미 풀어둔 폴더를 새 구조로 정리 (한 번만 실행)

zip을 하나씩 풀어서 생긴 아래 같은 구조를
  04. Prep/2_결과_A_검출학습/Blood_prep_v3/results/A_detection_train/...
  04. Prep/2_결과_C2a_가상염색_전체_비염색/Blood_prep_v3/results/C_virtual_staining/all/trainA ...
이렇게 바꿉니다.
  04. Prep/1_detection_train/
  04. Prep/2_cell_crops/
  04. Prep/3_virtual_staining/rbc_only/, all/
  04. Prep/4_eval_unstained/
  04. Prep/meta/
(예전 코드·안내 폴더 "1_코드_및_안내"는 건드리지 않습니다. 새 코드 zip으로 바꾸고 직접 지워 주세요.)

파일을 옮기기만 하고 지우지 않습니다. 옮기고 나서 빈 폴더만 정리합니다.
zip 파일은 그대로 두니, 정리가 끝난 걸 확인한 뒤 직접 지우면 됩니다.

사용법: python 05_reorganize_folders.py "D:/05. Blood/04. Prep"
"""
import glob
import os
import shutil
import sys

base = sys.argv[1]
RENAME = {"A_detection_train": "1_detection_train", "B_cell_crops": "2_cell_crops",
          "C_virtual_staining": "3_virtual_staining", "D_eval_unstained": "4_eval_unstained", "meta": "meta"}


def merge_move(src, dst):
    """src 폴더 내용을 dst로 옮김 (dst에 이미 있는 폴더는 합침)"""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s):
            merge_move(s, d)
        elif not os.path.exists(d):
            shutil.move(s, d)


roots = [r for r in glob.glob(os.path.join(base, "**", "Blood_prep_v3"), recursive=True) if os.path.isdir(r)]
if not roots:
    sys.exit("Blood_prep_v3 폴더를 못 찾았어요. 경로를 확인해 주세요.")
for r in roots:
    print("정리 중:", os.path.relpath(r, base))
    res = os.path.join(r, "results")
    if os.path.isdir(res):
        for name in os.listdir(res):
            merge_move(os.path.join(res, name), os.path.join(base, RENAME.get(name, name)))
    # 예전 코드·안내(1_코드_및_안내)는 옮기지 않음. 새 코드 zip으로 교체하고 예전 폴더는 직접 지우면 됨

# 비어 있는 폴더 정리 (안에 파일이 남아 있으면 지우지 않음)
for d in sorted({os.path.dirname(r) for r in roots}, key=len, reverse=True):
    for cur, dirs, files in os.walk(d, topdown=False):
        if not os.listdir(cur):
            os.rmdir(cur)

print("\n완료. 현재 폴더:")
for name in sorted(os.listdir(base)):
    print("  ", name)
# YOLO 설정 파일의 path 를 새 위치로 고침
yaml = os.path.join(base, "1_detection_train", "data.yaml")
if os.path.exists(yaml):
    lines = open(yaml, encoding="utf-8").read().splitlines()
    new_path = os.path.abspath(os.path.join(base, "1_detection_train")).replace("\\", "/")
    lines = [f"path: {new_path}" if l.startswith("path:") else l for l in lines]
    open(yaml, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\ndata.yaml path 수정:", new_path)
