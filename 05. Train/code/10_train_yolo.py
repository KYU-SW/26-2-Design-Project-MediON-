"""
10. 박스 기반 검출·계수 모델 (YOLO) 학습
입력 : 04. Prep/1_detection_train (실제 염색 영상, WBC=0 / RBC=1)
출력 : 05. Train/runs/yolo/<이름>/weights/best.pt  (+ 학습 곡선, 검증 점수)

기본 모델은 yolo11s 입니다. TXL-PBC 논문 기준선(YOLOv8s/YOLOv11s, mAP50 약 0.98)과
같은 크기라 결과를 바로 비교할 수 있어서 골랐습니다. --model 로 바꿀 수 있습니다.
  예) --model yolo26s.pt   (Ultralytics 최신, 2026.1)
      --model yolo11m.pt   (더 큰 모델)

색상 증강은 약하게만 줍니다. 가상 염색 결과는 실제 염색과 색 톤이 조금 다를 수 있어서
그 차이에는 버티되, 염색 색 자체를 무시하게 만들지는 않기 위함입니다.

사용법
  python 10_train_yolo.py --prep "D:/05. Blood/04. Prep" --out "D:/05. Blood/05. Train/runs"
  (처음 실행하면 사전학습 가중치를 자동으로 내려받습니다)
"""
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="04. Prep 폴더")
    ap.add_argument("--out", required=True, help="결과 저장 폴더 (예: 05. Train/runs)")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--name", default=None, help="실행 이름 (기본: 모델 이름)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="GPU 메모리가 부족하면 8로")
    ap.add_argument("--device", default="0", help="GPU 번호, CPU면 cpu")
    ap.add_argument("--workers", type=int, default=2, help="윈도우에서 문제 생기면 0")
    ap.add_argument("--fraction", type=float, default=1.0, help="테스트용: 학습 데이터 일부만 사용 (예: 0.05)")
    a = ap.parse_args()

    data = os.path.join(a.prep, "1_detection_train", "data.yaml")
    name = a.name or os.path.splitext(os.path.basename(a.model))[0]
    model = YOLO(a.model)
    model.train(
        data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
        project=os.path.join(a.out, "yolo"), name=name, exist_ok=True, seed=0, patience=30, fraction=a.fraction,
        # 약한 색상 증강
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.2,
        # 세포는 방향이 없으므로 상하 뒤집기도 허용
        flipud=0.5, fliplr=0.5,
    )
    best = os.path.join(a.out, "yolo", name, "weights", "best.pt")
    print("\n학습 완료. 가장 좋은 가중치:", best)
    # 테스트 분할 점수 (YOLO 자체 지표). 두 방식 비교는 12번 스크립트로 합니다.
    m = YOLO(best).val(data=data, split="test", imgsz=a.imgsz, device=a.device, workers=a.workers,
                      project=os.path.join(a.out, "yolo"), name=name + "_test", exist_ok=True)
    print(f"테스트 mAP50: {m.box.map50:.4f}, mAP50-95: {m.box.map:.4f}")


if __name__ == "__main__":
    main()
