@echo off
chcp 65001 >nul
REM =========================================================
REM  더블클릭하면 D:\05. Blood 의 변경 내용을 GitHub에 올립니다.
REM  1) 팀원이 올린 최신 내용을 먼저 받고
REM  2) 내 변경 내용을 커밋해서
REM  3) GitHub에 올립니다.
REM  (.gitignore 에 적힌 데이터·모델 파일은 자동으로 빠집니다)
REM =========================================================
cd /d "D:\05. Blood"

echo [1/3] GitHub 최신 내용 받는 중...
git pull --rebase --autostash origin main
if errorlevel 1 (
  echo.
  echo !! 받아오다가 충돌이 났어요. 이 창 내용을 캡처해서 물어봐 주세요.
  pause
  exit /b 1
)

echo [2/3] 변경 내용 정리 중...
git add -A
git diff --cached --quiet
if not errorlevel 1 (
  echo 바뀐 파일이 없어요. 올릴 것 없음.
  pause
  exit /b 0
)
git status --short
set /p MSG="커밋 메시지 (그냥 Enter 누르면 날짜로 자동 입력): "
if "%MSG%"=="" set MSG=update %date% %time:~0,5%
git commit -m "%MSG%"

echo [3/3] GitHub에 올리는 중...
git push origin main
if errorlevel 1 (
  echo.
  echo !! 올리기 실패. 이 창 내용을 캡처해서 물어봐 주세요.
  pause
  exit /b 1
)
echo.
echo 완료!
pause
