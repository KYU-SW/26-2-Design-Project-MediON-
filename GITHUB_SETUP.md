# GitHub 연결 방법 (처음 한 번만)

저장소: https://github.com/KYU-SW/26-2-Design-Project-MediON-

`D:\05. Blood` 폴더를 그대로 저장소로 씁니다. 코드·노트북 안의 경로(`D:/05. Blood/...`)를 고칠 필요가 없습니다.

## 1. Git 설치 확인

Anaconda Prompt 또는 명령 프롬프트에서:

```bash
git --version
```

버전이 안 나오면 [Git for Windows](https://git-scm.com/download/win)를 설치하고 창을 다시 여세요.

## 2. 처음 연결 (한 번만)

```bash
cd /d "D:\05. Blood"
git init -b main
git config core.quotepath false
git config user.name "본인 이름"
git config user.email "GitHub에 쓰는 이메일"
git remote add origin https://github.com/KYU-SW/26-2-Design-Project-MediON-.git
git fetch origin
```

### 2-1. 저장소에 이미 파일이 있는 경우 (팀원이 먼저 올렸거나 README가 있음)

```bash
git pull origin main --allow-unrelated-histories
```

- `main`이 아니라 `master` 브랜치를 쓰는 저장소라면, 이 문서와 `sync_github.bat`의 `main`을 `master`로 바꾸세요.
- 같은 이름의 파일이 있어서 충돌이 나면 멈추고 화면을 캡처해 주세요.

### 2-2. 첫 업로드

```bash
git add -A
git status
```

**여기서 꼭 확인:** 목록에 `03. Data/`, `04. Prep/1_detection_train/` 같은 데이터 폴더나 `.pt` 파일이 **없어야** 합니다. 있으면 멈추고 물어봐 주세요.

```bash
git commit -m "초기 업로드: 전처리·학습 코드, 노트북, 결과"
git push -u origin main
```

처음 push할 때 GitHub 로그인 창이 뜹니다. 브라우저로 로그인하면 다음부터는 안 물어봅니다.

## 3. 이후에는

작업이 끝날 때마다 `D:\05. Blood\sync_github.bat`을 **더블클릭**하면 됩니다.

1. 팀원이 올린 최신 내용을 먼저 받고
2. 내 변경 내용을 커밋하고 (메시지를 물어봄, 그냥 Enter면 날짜로 자동)
3. GitHub에 올립니다

## 4. (선택) 완전 자동으로 올리고 싶다면

윈도우 **작업 스케줄러** → 기본 작업 만들기 → 매일 / 1시간마다 → 프로그램 시작 → `D:\05. Blood\sync_github.bat`

단, 학습 중에 돌면 어중간한 결과가 올라갈 수 있고 커밋 메시지를 물어보는 창이 뜹니다. 팀 저장소라면 수동(더블클릭)을 추천합니다.

## 올라가는 것 / 빠지는 것

| 올라감 | 빠짐 (`.gitignore`) |
| --- | --- |
| 문서 (`01`, `02`, PluMedic pdf) | `03. Data` 원본 데이터 |
| `04. Prep` README, code, meta | `04. Prep` 1~4번 데이터 폴더, `old` |
| `05. Train` README, code, 노트북, results | 모델 가중치 `.pt` `.pth` |
| 학습 곡선·그래프 (`runs`) | `99. Trash`, zip, 캐시 |

데이터를 쓰려는 팀원은 `04. Prep/README.md`의 출처에서 받은 뒤 `04. Prep/code/03_prep_data.py`로 같은 결과를 만들 수 있습니다.
