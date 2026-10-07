"""Ultralytics YOLO26으로 웹캠 실시간 객체 탐지 (Object Detection).

사용법:
    python detect.py
    python detect.py --model yolo26s.pt --conf 0.4 --camera 0

종료: 영상 창에서 'q' 또는 ESC 키
"""

from ultralytics import YOLO

from common import build_parser, run_webcam


def main():
    args = build_parser("YOLO26 webcam object detection", "yolo26n.pt").parse_args()
    model = YOLO(args.model)

    def infer(frame):
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        return result.plot(), f"Objects: {len(result.boxes)}"

    run_webcam(args.camera, "YOLO26 Detection", infer)


if __name__ == "__main__":
    main()
