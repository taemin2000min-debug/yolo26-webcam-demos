"""Ultralytics YOLO26으로 웹캠 실시간 인스턴스 분할 (Instance Segmentation).

사용법:
    python segment.py
    python segment.py --model yolo26s-seg.pt --conf 0.4 --camera 0

종료: 영상 창에서 'q' 또는 ESC 키
"""

from ultralytics import YOLO

from common import build_parser, run_webcam


def main():
    args = build_parser("YOLO26 webcam instance segmentation", "yolo26n-seg.pt").parse_args()
    model = YOLO(args.model)

    def infer(frame):
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        return result.plot(), f"Masks: {len(result.boxes)}"

    run_webcam(args.camera, "YOLO26 Segmentation", infer)


if __name__ == "__main__":
    main()
