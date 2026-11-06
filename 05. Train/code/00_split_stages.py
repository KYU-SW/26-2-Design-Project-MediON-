"""
00. 단계별 폴더 나누기 (한 번만, 지우는 것 없음 — 옮기기만)

05. Train 에 학습·생성·평가 결과가 다 몰려 있던 것을 단계별 최상위 폴더로 나눕니다.

  정리 전                                   정리 후
  05. Train/virtual_stained/…        →     06. Generate/…        가상 염색으로 만든 영상 (cyclegan/, unsb/ …)
  05. Train/results/…                →     07. Evaluate/…        채점 결과 (1_detection/ … 6_detector/, final_*.csv)
  05. Train/runs, datasets, external, code, run_*.ipynb   그대로 (학습)

노트북(run_*.ipynb)과 05. Train/README.md 안의 경로도 새 위치로 바꿉니다 (출력은 그대로).
바꾸기 전 사본은 05. Train/_backup_before_split/ 에 둡니다.

실행 전: Jupyter 노트북 탭을 모두 닫고, 결과 파일을 엑셀 등으로 열어 두지 마세요.

사용법
  python 00_split_stages.py --root "D:/05. Blood" --dry_run   # 무엇을 옮길지 보기만
  python 00_split_stages.py --root "D:/05. Blood"             # 실제로 옮김
"""
import argparse
import glob
import json
import os
import re
import shutil

MOVES = [("05. Train/virtual_stained", "06. Generate"), ("05. Train/results", "07. Evaluate")]

# (패턴, 바꿀 내용) — 노트북·README 글자 안의 경로
RULES = [
    (r"05\. Train([/\\])results([/\\])", r"07. Evaluate\1"),
    (r"05\. Train([/\\])results\b", "07. Evaluate"),
    (r"05\. Train([/\\])virtual_stained([/\\])", r"06. Generate\1"),
    (r"05\. Train([/\\])virtual_stained\b", "06. Generate"),
    (r"\{T\}/results/", "D:/05. Blood/07. Evaluate/"),
    (r"\{T\}/virtual_stained/", "D:/05. Blood/06. Generate/"),
    (r'T \+ "/results/', '"D:/05. Blood/07. Evaluate/'),
    (r'T \+ "/virtual_stained/', '"D:/05. Blood/06. Generate/'),
    (r"`results/", "`07. Evaluate/"),
    (r"`virtual_stained/", "`06. Generate/"),
]


def rewrite(text):
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    return text


def move_tree(src, dst, dry):
    """dst가 없으면 통째로 이름 바꾸기, 있으면 안의 항목을 하나씩 옮김 (같은 이름이 있으면 건너뜀)"""
    done, skipped = 0, []
    if not os.path.isdir(src):
        return done, skipped
    if not os.path.exists(dst):
        print(("[보기] " if dry else "") + f"{src} → {dst}")
        if not dry:
            shutil.move(src, dst)
        return 1, skipped
    for n in sorted(os.listdir(src)):
        s, d = os.path.join(src, n), os.path.join(dst, n)
        if os.path.exists(d):
            skipped.append(s)
            continue
        print(("[보기] " if dry else "") + f"{s} → {d}")
        if not dry:
            shutil.move(s, d)
        done += 1
    if not dry and not os.listdir(src):
        os.rmdir(src)  # 빈 폴더만 지움
    return done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="프로젝트 최상위 폴더 (D:/05. Blood)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    R = a.root
    T = os.path.join(R, "05. Train")

    total, skipped = 0, []
    for s, d in MOVES:
        n, sk = move_tree(os.path.join(R, s), os.path.join(R, d), a.dry_run)
        total += n
        skipped += sk

    files = sorted(glob.glob(os.path.join(T, "run_*.ipynb"))) + [os.path.join(T, "README.md")]
    if not a.dry_run:
        bk = os.path.join(T, "_backup_before_split")
        os.makedirs(bk, exist_ok=True)
        for f in files:
            if os.path.exists(f) and not os.path.exists(os.path.join(bk, os.path.basename(f))):
                shutil.copy2(f, os.path.join(bk, os.path.basename(f)))
    for f in files:
        if not os.path.exists(f):
            continue
        if f.endswith(".ipynb"):
            nb = json.load(open(f, encoding="utf-8"))
            k = 0
            for c in nb["cells"]:
                s = "".join(c["source"])
                new = rewrite(s)
                if new != s:
                    c["source"] = new.splitlines(True)
                    k += 1
            if k and not a.dry_run:
                json.dump(nb, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        else:
            s = open(f, encoding="utf-8").read()
            new = rewrite(s)
            k = int(new != s)
            if k and not a.dry_run:
                open(f, "w", encoding="utf-8").write(new)
        print(f"{'[보기] ' if a.dry_run else ''}{os.path.basename(f)}: 경로를 바꾼 셀 {k}개")

    print(f"\n{'[보기] ' if a.dry_run else ''}옮김 {total}개, 건너뜀(같은 이름이 이미 있음) {len(skipped)}개")
    for s in skipped:
        print("  건너뜀:", s)


if __name__ == "__main__":
    main()
