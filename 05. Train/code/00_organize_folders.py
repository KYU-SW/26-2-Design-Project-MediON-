"""
00. 05. Train 결과 폴더 정리 (한 번만 실행, 지우는 것 없음 — 옮기기만)

정리 전 results/ 에 30여 개 폴더가 한 줄로 섞여 있던 것을 단계별로 나눕니다.

  results/
    1_detection/      검출 모델 비교·conf 정하기 (평가 0)         eval0_*, val_yolo_conf*
    2_no_staining/    가상 염색 없이 비염색 그대로 (하한)          eval2_*, stain_none_*
    3_staining/       가상 염색 모델별 결과 (평가 ②)
      cyclegan/         ep15_val, ep15_test ...                    (예전 이름 stain_cyclegan_ep15_val)
      pix2pix/          (앞으로)
    4_cell_mask/      B 방식 세포 윤곽 결과                        cellmask_*
  virtual_stained/
    cyclegan/ep15/{val,test}                                       (예전 이름 cyclegan_ep15)
    pix2pix/...                                                    (앞으로)

노트북(run_train.ipynb, run_staining.ipynb)의 경로도 새 위치로 바꿉니다 (출력은 그대로 둠).
바꾸기 전 노트북은 _backup_before_organize/ 에 복사해 둡니다.

실행 전: Jupyter에서 두 노트북 탭을 닫고, 결과 csv를 엑셀로 열어 두지 마세요 (열린 파일은 못 옮김).

사용법
  python 00_organize_folders.py --train "D:/05. Blood/05. Train"            # 실제로 옮김
  python 00_organize_folders.py --train "D:/05. Blood/05. Train" --dry_run  # 무엇을 옮길지 보기만
"""
import argparse
import json
import os
import re
import shutil


def plan_moves(train):
    R = os.path.join(train, "results")
    V = os.path.join(train, "virtual_stained")
    moves = []
    if os.path.isdir(R):
        for n in sorted(os.listdir(R)):
            src = os.path.join(R, n)
            if not os.path.isdir(src) or n[:2] in ("1_", "2_", "3_", "4_"):
                continue
            m = re.fullmatch(r"stain_cyclegan_ep(\d+)_(val|test)", n)
            if n.startswith(("eval0_", "val_yolo_")):
                dst = os.path.join(R, "1_detection", n)
            elif n.startswith(("eval2_", "stain_none_")):
                dst = os.path.join(R, "2_no_staining", n)
            elif m:
                dst = os.path.join(R, "3_staining", "cyclegan", f"ep{m.group(1)}_{m.group(2)}")
            elif n.startswith("cellmask_"):
                dst = os.path.join(R, "4_cell_mask", n)
            else:
                continue
            moves.append((src, dst))
    if os.path.isdir(V):
        for n in sorted(os.listdir(V)):
            m = re.fullmatch(r"cyclegan_ep(\d+)", n)
            if m:
                moves.append((os.path.join(V, n), os.path.join(V, "cyclegan", f"ep{m.group(1)}")))
    return moves


# 노트북 안 경로 바꾸기: (찾을 패턴, 바꿀 내용). {s} = 원래 쓰인 구분자 (/ 또는 \)
RULES = [
    (r"results([/\\])eval0_", r"results\g<1>1_detection\g<1>eval0_"),
    (r"results([/\\])val_yolo_", r"results\g<1>1_detection\g<1>val_yolo_"),
    (r"results([/\\])eval2_", r"results\g<1>2_no_staining\g<1>eval2_"),
    (r"results([/\\])stain_none_", r"results\g<1>2_no_staining\g<1>stain_none_"),
    (r"results([/\\])stain_cyclegan_ep", r"results\g<1>3_staining\g<1>cyclegan\g<1>ep"),
    (r"results([/\\])cellmask_", r"results\g<1>4_cell_mask\g<1>cellmask_"),
    (r"virtual_stained([/\\])cyclegan_ep", r"virtual_stained\g<1>cyclegan\g<1>ep"),
]


def fix_notebook(path, dry):
    nb = json.load(open(path, encoding="utf-8"))
    n = 0
    for c in nb["cells"]:
        src = "".join(c["source"])
        new = src
        for pat, rep in RULES:
            new = re.sub(pat, rep, new)
        if new != src:
            n += 1
            c["source"] = new.splitlines(True)
    if n and not dry:
        json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="05. Train 폴더")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    moves = plan_moves(a.train)
    ok, fail = 0, []
    for src, dst in moves:
        print(("[보기] " if a.dry_run else "") + os.path.relpath(src, a.train), "→", os.path.relpath(dst, a.train))
        if a.dry_run:
            continue
        if os.path.exists(dst):
            fail.append((src, "옮길 곳에 같은 이름이 이미 있음"))
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            ok += 1
        except OSError as e:
            fail.append((src, str(e)))

    nbs = [p for p in ["run_train.ipynb", "run_staining.ipynb"] if os.path.exists(os.path.join(a.train, p))]
    if not a.dry_run and nbs:
        bk = os.path.join(a.train, "_backup_before_organize")
        os.makedirs(bk, exist_ok=True)
        for p in nbs:
            if not os.path.exists(os.path.join(bk, p)):
                shutil.copy2(os.path.join(a.train, p), os.path.join(bk, p))
    for p in nbs:
        k = fix_notebook(os.path.join(a.train, p), a.dry_run)
        print(f"{'[보기] ' if a.dry_run else ''}{p}: 경로를 바꾼 셀 {k}개")

    if not a.dry_run:
        print(f"\n옮김 {ok}개 / 실패 {len(fail)}개")
        for s, why in fail:
            print("  실패:", os.path.relpath(s, a.train), "-", why)


if __name__ == "__main__":
    main()
