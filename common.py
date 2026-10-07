"""detect / segment / track 스크립트가 함께 쓰는 웹캠 루프와 공통 옵션."""

import argparse
import time

import cv2


def build_parser(description, default_model):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=default_model,
                        help=f"모델 가중치 (기본값: {default_model}). 없으면 자동 다운로드")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    parser.add_argument("--imgsz", type=int, default=640, help="추론 이미지 크기")
    parser.add_argument("--device", default=None,
                        help="추론 장치 (예: cpu, 0). 기본값은 자동 선택")
    return parser


def open_camera(index):
    # Windows에서는 DirectShow 백엔드가 웹캠을 더 빨리 연다
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({index})을 열 수 없습니다.")
    return cap


def run_webcam(camera, window, infer):
    """웹캠 프레임마다 infer(frame) -> (annotated, info_text)를 호출해 화면에 표시한다.

    종료: 영상 창에서 'q' 또는 ESC 키
    """
    cap = open_camera(camera)
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다. 종료합니다.")
                break

            annotated, info = infer(frame)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(annotated, f"FPS: {fps:.1f}  {info}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
