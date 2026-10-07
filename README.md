# YOLO26 Webcam Demos

[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)으로 웹캠 영상을 실시간 처리하는 예제 3종입니다.

| 스크립트 | 작업 | 기본 모델 |
|---|---|---|
| `detect.py` | 객체 탐지 (Object Detection) | `yolo26n.pt` |
| `segment.py` | 인스턴스 분할 (Instance Segmentation) | `yolo26n-seg.pt` |
| `track.py` | 객체 추적 (Object Tracking, ID + 이동 경로) | `yolo26n.pt` |

`common.py`에는 세 스크립트가 함께 쓰는 웹캠 루프와 공통 옵션이 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

모델 가중치는 처음 실행할 때 자동으로 다운로드됩니다.

## 실행

```bash
python detect.py
python segment.py
python track.py
```

영상 창에서 `q` 또는 `ESC`를 누르면 종료합니다.

## 공통 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--model` | 모델 가중치 (`n`/`s`/`m`/`l`/`x` 크기 선택) | 스크립트별 상이 |
| `--camera` | 웹캠 인덱스 | `0` |
| `--conf` | 신뢰도 임계값 | `0.25` |
| `--imgsz` | 추론 이미지 크기 | `640` |
| `--device` | 추론 장치 (`cpu`, `0` 등) | 자동 |

`track.py` 전용 옵션:

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--tracker` | `bytetrack.yaml` 또는 `botsort.yaml` | `bytetrack.yaml` |
| `--trail` | 이동 경로 길이 (프레임 수, `0`이면 끔) | `30` |

예시:

```bash
python segment.py --model yolo26s-seg.pt --conf 0.4
python track.py --tracker botsort.yaml --trail 50
```
