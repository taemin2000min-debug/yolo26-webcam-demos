"""Ultralytics YOLO26으로 웹캠 실시간 객체 추적 (Object Tracking).

객체마다 고유 ID를 부여하고, 최근 이동 경로를 선으로 그린다.

사용법:
    python track.py
    python track.py --tracker botsort.yaml --trail 50

종료: 영상 창에서 'q' 또는 ESC 키
"""

from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from common import build_parser, run_webcam


def main():
    parser = build_parser("YOLO26 webcam object tracking", "yolo26n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="추적 알고리즘 설정 (bytetrack.yaml 또는 botsort.yaml)")
    parser.add_argument("--trail", type=int, default=30,
                        help="그릴 이동 경로 길이 (프레임 수, 0이면 그리지 않음)")
    args = parser.parse_args()

    model = YOLO(args.model)
    trails = defaultdict(lambda: deque(maxlen=max(args.trail, 1)))

    def infer(frame):
        # persist=True: 프레임 간 추적 상태를 유지해 같은 객체에 같은 ID를 부여
        result = model.track(frame, persist=True, tracker=args.tracker,
                             conf=args.conf, imgsz=args.imgsz,
                             device=args.device, verbose=False)[0]
        annotated = result.plot()

        boxes = result.boxes
        if boxes.id is None:
            return annotated, "Tracks: 0"

        ids = boxes.id.int().tolist()
        if args.trail > 0:
            for track_id, (x, y, _, _) in zip(ids, boxes.xywh.tolist()):
                trails[track_id].append((int(x), int(y)))
                points = np.array(trails[track_id], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [points], False, (230, 230, 230), 2)

        return annotated, f"Tracks: {len(ids)}"

    run_webcam(args.camera, "YOLO26 Tracking", infer)


if __name__ == "__main__":
    main()
