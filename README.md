# YOLO26으로 배우는 실시간 웹캠 비전 AI

> **강의노트** — Ultralytics YOLO26으로 웹캠 영상에서 **객체 탐지 → 인스턴스 분할 → 객체 추적**을 차례로 구현하고, 결과물을 GitHub에 올리는 과정까지 다룹니다.

---

## 목차

0. [학습 목표](#0-학습-목표)
1. [환경 준비](#1-환경-준비)
2. [YOLO26 한눈에 보기](#2-yolo26-한눈에-보기)
3. [실습 1 — 객체 탐지 (Detection)](#3-실습-1--객체-탐지-detection)
4. [실습 2 — 인스턴스 분할 (Segmentation)](#4-실습-2--인스턴스-분할-segmentation)
5. [실습 3 — 객체 추적 (Tracking)](#5-실습-3--객체-추적-tracking)
6. [리팩터링 — 공통 코드 분리](#6-리팩터링--공통-코드-분리)
7. [GitHub CLI로 프로젝트 올리기](#7-github-cli로-프로젝트-올리기)
8. [트러블슈팅 노트](#8-트러블슈팅-노트)
9. [정리 & 실습 과제](#9-정리--실습-과제)
10. [부록 — 명령어 & 옵션 요약](#부록--명령어--옵션-요약)

---

## 0. 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

- [ ] `ultralytics` 패키지로 YOLO26 모델을 불러와 추론할 수 있다.
- [ ] OpenCV로 웹캠 프레임을 읽고, 결과를 화면에 실시간으로 그릴 수 있다.
- [ ] **탐지 / 분할 / 추적**이 어떻게 다르고, 코드에서 무엇이 바뀌는지 설명할 수 있다.
- [ ] 겹치는 코드를 공통 모듈로 분리할 수 있다.
- [ ] `gh` CLI로 GitHub 저장소를 만들고 코드를 푸시할 수 있다.

### 최종 프로젝트 구조

```
yolo26-webcam-demos/
├── common.py         # 공통: 웹캠 열기, 프레임 루프, FPS 표시, CLI 옵션
├── detect.py         # 실습 1: 객체 탐지
├── segment.py        # 실습 2: 인스턴스 분할
├── track.py          # 실습 3: 객체 추적
├── requirements.txt  # 의존성
├── .gitignore        # *.pt(모델 가중치), __pycache__ 등 제외
└── README.md         # 이 강의노트
```

---

## 1. 환경 준비

### 1-1. 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`

```text
ultralytics>=8.4.0
opencv-python
lap>=0.5.12  # track.py (ByteTrack/BoT-SORT)
```

| 패키지 | 역할 |
|---|---|
| `ultralytics` | YOLO 모델 로딩·추론·추적. `torch`, `torchvision`도 함께 설치됨 |
| `opencv-python` | 웹캠 캡처(`VideoCapture`), 화면 출력(`imshow`) |
| `lap` | 추적기(ByteTrack/BoT-SORT)가 쓰는 선형 할당(Linear Assignment) 라이브러리 |

> 💡 **YOLO26은 `ultralytics` 8.4 이상**에서 쓸 수 있습니다. 설치 후 버전을 확인해 보세요.
>
> ```bash
> python -c "import ultralytics; print(ultralytics.__version__)"
> ```

### 1-2. ⚠️ 함정: `python`과 `pip`가 서로 다른 Python일 때

실습 중 실제로 겪은 문제입니다. `pip install`은 성공했는데 실행하면 이런 에러가 났습니다.

```
ModuleNotFoundError: No module named 'ultralytics'
```

**원인** — PC에 Python이 여러 개 설치되어 있었습니다.

```bash
$ which python
.../Microsoft/WindowsApps/python          # → Python 3.14 (Microsoft Store)
$ which pip
.../Python/Python312/Scripts/pip          # → Python 3.12
```

`pip`는 3.12에 설치했는데, `python`은 3.14를 실행하고 있었던 것입니다.

**해결 방법**

```bash
# 방법 A: 인터프리터를 직접 지정해서 실행 (Windows py 런처)
py -3.12 detect.py

# 방법 B: 설치할 때부터 "실행할 그 python"의 pip를 쓴다  ← 권장 습관
python -m pip install -r requirements.txt
```

> 📌 **교훈**: `pip install` 대신 **`python -m pip install`** 을 쓰면, 실행할 Python과 설치할 Python이 항상 같습니다. 가상환경(`python -m venv .venv`)을 쓰면 더 확실합니다.

---

## 2. YOLO26 한눈에 보기

**YOLO (You Only Look Once)** 는 이미지를 한 번만 보고(single pass) 객체의 위치와 종류를 동시에 예측하는 실시간 비전 모델 계열입니다. **YOLO26**은 Ultralytics가 공개한 최신 세대입니다.

### 2-1. 하나의 API, 여러 작업(Task)

모델 파일 이름의 **접미사**로 작업이 정해집니다.

| 작업 | 모델 파일 예시 | 출력 | 이번 실습 |
|---|---|---|---|
| 탐지 (Detect) | `yolo26n.pt` | 박스 + 클래스 + 신뢰도 | ✅ 실습 1, 3 |
| 분할 (Segment) | `yolo26n-seg.pt` | 박스 + **픽셀 단위 마스크** | ✅ 실습 2 |
| 자세 추정 (Pose) | `yolo26n-pose.pt` | 사람 관절 키포인트 | 과제 |
| 분류 (Classify) | `yolo26n-cls.pt` | 이미지 전체의 클래스 | — |
| 회전 박스 (OBB) | `yolo26n-obb.pt` | 회전된 박스 | — |

> 추적(Track)은 별도 모델이 아니라, **탐지(또는 분할) 모델 + 추적 알고리즘**의 조합입니다. → [실습 3](#5-실습-3--객체-추적-tracking)

### 2-2. 모델 크기: n / s / m / l / x

```
yolo26n  →  yolo26s  →  yolo26m  →  yolo26l  →  yolo26x
빠름·가벼움 ─────────────────────────────▶ 느림·정확
```

CPU만 있는 노트북이라면 **`n`(nano)** 부터 시작하세요. 이번 실습 환경(`torch 2.14.1+cpu`, GPU 없음)에서도 `n` 모델을 썼습니다.

### 2-3. 기본 사용법 3줄

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")         # 1) 모델 로드 (파일이 없으면 자동 다운로드)
results = model.predict("bus.jpg") # 2) 추론
results[0].show()                  # 3) 결과 시각화
```

---

## 3. 실습 1 — 객체 탐지 (Detection)

> **목표**: 웹캠 영상의 모든 프레임에서 객체를 찾아 박스를 그린다.

### 3-1. 실시간 처리의 기본 구조

웹캠 비전 프로그램은 거의 모두 이 루프를 따릅니다.

```
┌────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
│ 프레임 읽기 │──▶│ 모델 추론 │──▶│ 결과 그리기  │──▶│ 화면 출력 │──┐
│ cap.read() │   │ predict()│   │ result.plot()│   │ imshow() │  │
└────────────┘   └──────────┘   └──────────────┘   └──────────┘  │
      ▲                                                          │
      └──────────────── 'q' / ESC 누를 때까지 반복 ◀──────────────┘
```

### 3-2. 처음 작성한 단일 파일 버전

리팩터링하기 전, 모든 내용을 한 파일에 담았던 첫 버전입니다. **흐름을 이해하기에는 이 버전이 가장 좋습니다.**

```python
import time
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Windows에서는 DirectShow(CAP_DSHOW) 백엔드가 웹캠을 더 빨리 연다
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)          # 실패하면 기본 백엔드로 재시도

prev_time = time.perf_counter()
try:
    while True:
        ok, frame = cap.read()          # ① 프레임 읽기
        if not ok:
            break

        results = model.predict(frame, conf=0.25, verbose=False)  # ② 추론
        annotated = results[0].plot()   # ③ 박스·라벨이 그려진 이미지

        now = time.perf_counter()       # FPS = 1 / (프레임 간 시간)
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO26", annotated) # ④ 화면 출력
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q 또는 ESC
            break
finally:
    cap.release()                       # 예외가 나도 카메라는 반드시 해제
    cv2.destroyAllWindows()
```

### 3-3. 코드 포인트

| 코드 | 의미 |
|---|---|
| `cv2.CAP_DSHOW` | Windows 전용 DirectShow 백엔드. 기본 백엔드보다 웹캠이 빨리 열리는 경우가 많음 |
| `conf=0.25` | 신뢰도 25% 미만인 예측은 버림. 높이면 오탐↓·미탐↑ |
| `verbose=False` | 프레임마다 콘솔에 로그가 찍히지 않게 함 |
| `results[0]` | `predict()`는 **리스트**를 반환. 이미지 1장이면 `[0]` |
| `result.plot()` | 박스·라벨·신뢰도를 그린 **numpy 이미지(BGR)** 반환 |
| `cv2.waitKey(1)` | 1ms 동안 키 입력 대기. **이 호출이 있어야 창이 갱신됨** |
| `try / finally` | 에러로 끝나도 웹캠을 놓아주기 위함 |

### 3-4. `Results` 객체에서 값 꺼내기

그림만 그리는 것이 아니라 **숫자로 된 결과**를 쓰고 싶을 때:

```python
result = model.predict(frame, verbose=False)[0]
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 박스 좌표 (픽셀)
    cls_id = int(box.cls)                  # 클래스 번호
    name = result.names[cls_id]            # 클래스 이름 (예: 'person')
    score = float(box.conf)                # 신뢰도
    print(f"{name} {score:.2f} at ({x1:.0f}, {y1:.0f})")
```

### 3-5. 실행

```bash
python detect.py
python detect.py --model yolo26s.pt --conf 0.4
```

---

## 4. 실습 2 — 인스턴스 분할 (Segmentation)

> **목표**: 박스가 아니라 **객체의 실제 윤곽(픽셀 마스크)** 을 칠한다.

### 4-1. 탐지 vs 분할

```
탐지 (Detection)          분할 (Segmentation)
┌───────────┐              ┌───────────┐
│   ▓▓▓     │              │   ███     │
│  ▓▓▓▓▓    │  박스만       │  █████    │  박스 + 객체 모양 그대로
│   ▓▓▓     │              │   ███     │  (배경 픽셀은 제외)
└───────────┘              └───────────┘
```

### 4-2. 코드에서 바뀌는 것: **모델 파일 한 줄**

```diff
- model = YOLO("yolo26n.pt")
+ model = YOLO("yolo26n-seg.pt")
```

나머지 루프는 **그대로**입니다. `result.plot()`이 마스크까지 알아서 칠해 줍니다. Ultralytics API가 작업과 관계없이 통일되어 있다는 점이 핵심입니다.

### 4-3. 마스크 데이터 직접 쓰기

```python
result = model.predict(frame, verbose=False)[0]
if result.masks is not None:
    masks = result.masks.data   # (N, H, W) 텐서, 객체별 0/1 마스크
    polygons = result.masks.xy  # 객체별 윤곽선 좌표 리스트
```

> 💡 활용 예: 사람 마스크만 남기고 배경 흐리기(가상 배경), 객체 면적 계산 등

### 4-4. 실행

```bash
python segment.py
python segment.py --model yolo26s-seg.pt --conf 0.4
```

---

## 5. 실습 3 — 객체 추적 (Tracking)

> **목표**: 프레임이 바뀌어도 **같은 객체에는 같은 ID**를 붙이고, 이동 경로를 그린다.

### 5-1. 왜 탐지만으로는 부족한가?

탐지는 프레임마다 **독립적**입니다. 1번 프레임의 "person"과 2번 프레임의 "person"이 같은 사람인지 모릅니다. 사람 수 세기, 출입 카운팅, 동선 분석에는 **"누가 누구인지"** 가 필요합니다.

```
프레임 1        프레임 2        프레임 3
[person]  →   [person]  →   [person]      ← 탐지: 매번 새로운 객체
 ID 1     →    ID 1     →    ID 1         ← 추적: 같은 객체로 연결
```

### 5-2. 코드에서 바뀌는 것: `predict()` → `track()`

```diff
- result = model.predict(frame, verbose=False)[0]
+ result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
```

| 인자 | 의미 |
|---|---|
| `persist=True` | **가장 중요!** 프레임 사이에 추적 상태를 유지. 빠뜨리면 매 프레임 ID가 초기화됨 |
| `tracker=` | 추적 알고리즘 선택 (아래 표) |

| 추적기 | 특징 |
|---|---|
| `bytetrack.yaml` | 빠르고 가벼움. 신뢰도가 낮은 박스도 활용해 가려짐(occlusion)에 강함 |
| `botsort.yaml` | 카메라 움직임 보정 등이 추가되어 더 정교하지만 조금 느림 |

### 5-3. 추적 ID와 이동 경로 그리기

```python
from collections import defaultdict, deque

trails = defaultdict(lambda: deque(maxlen=30))  # ID별 최근 30개 위치

result = model.track(frame, persist=True, verbose=False)[0]
annotated = result.plot()

if result.boxes.id is not None:                 # 추적된 객체가 없으면 id는 None
    ids = result.boxes.id.int().tolist()
    for track_id, (x, y, w, h) in zip(ids, result.boxes.xywh.tolist()):
        trails[track_id].append((int(x), int(y)))  # 박스 중심점 기록
        pts = np.array(trails[track_id], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], False, (230, 230, 230), 2)
```

| 기법 | 이유 |
|---|---|
| `deque(maxlen=30)` | 오래된 좌표가 자동으로 빠져서 꼬리 길이가 일정 |
| `defaultdict` | 처음 보는 ID도 따로 초기화할 필요 없음 |
| `boxes.id is None` 체크 | 화면에 아무것도 없을 때 에러 방지 |
| `xywh` | 중심 좌표 `(x, y)`와 너비·높이. 경로는 중심점으로 그림 |

### 5-4. 실행

```bash
python track.py
python track.py --tracker botsort.yaml --trail 50
```

> ⚠️ 처음 `track()`을 호출하면 Ultralytics가 `lap` 패키지가 없다며 자동 설치를 시도합니다. 미리 `requirements.txt`에 넣어 두면 이 과정이 생략됩니다.

---

## 6. 리팩터링 — 공통 코드 분리

### 6-1. 문제: 세 파일이 거의 똑같다

세 스크립트를 나란히 놓고 보면 **달라지는 것은 "추론하고 그리는 부분"뿐**이었습니다.

| 부분 | detect | segment | track |
|---|:---:|:---:|:---:|
| CLI 옵션 (`--model`, `--conf` …) | 같음 | 같음 | 같음 (+2개) |
| 웹캠 열기 / 프레임 루프 / FPS / 종료 키 | 같음 | 같음 | 같음 |
| **추론 + 그리기** | `predict` | `predict` (seg 모델) | `track` + 경로 |

### 6-2. 해결: 바뀌는 부분만 함수로 넘기기

`common.py`의 `run_webcam()`이 루프 전체를 맡고, 각 스크립트는 **`infer(frame)` 함수 하나만** 넘깁니다.

```python
# common.py (핵심만)
def run_webcam(camera, window, infer):
    cap = open_camera(camera)
    while True:
        ok, frame = cap.read()
        annotated, info = infer(frame)   # ← 스크립트마다 다른 부분
        # ... FPS 표시, imshow, 종료 키 처리 ...
```

```python
# detect.py — 이제 이것이 전부
def main():
    args = build_parser("YOLO26 webcam object detection", "yolo26n.pt").parse_args()
    model = YOLO(args.model)

    def infer(frame):
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        return result.plot(), f"Objects: {len(result.boxes)}"

    run_webcam(args.camera, "YOLO26 Detection", infer)
```

> 📌 **설계 포인트**: "변하지 않는 흐름"은 공통 함수에, "변하는 동작"은 **함수(콜백)로 주입**합니다. 새 작업(예: Pose)을 추가할 때도 `infer`만 새로 작성하면 됩니다.

### 6-3. CLI 옵션도 공통화

`build_parser()`가 공통 옵션을 만들어 주고, `track.py`처럼 추가 옵션이 필요한 스크립트는 받은 parser에 더 붙입니다.

```python
parser = build_parser("YOLO26 webcam object tracking", "yolo26n.pt")
parser.add_argument("--tracker", default="bytetrack.yaml")
parser.add_argument("--trail", type=int, default=30)
args = parser.parse_args()
```

### 6-4. 웹캠 없이 검증하기

리팩터링 후에는 Ultralytics에 들어 있는 샘플 이미지(`bus.jpg`)로 세 모델이 정상 동작하는지 먼저 확인했습니다.

```python
import os, cv2, ultralytics
from ultralytics import YOLO

img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
print(len(YOLO("yolo26n.pt").predict(img, verbose=False)[0].boxes))      # 탐지
print(YOLO("yolo26n-seg.pt").predict(img, verbose=False)[0].masks is not None)  # 분할
```

| 검증 항목 | 결과 |
|---|---|
| 탐지 | 객체 5개 |
| 분할 | 마스크 5개 |
| 추적 (같은 이미지 2회) | ID `[1, 2, 3, 4, 5]` |
| 각 스크립트 `--help` | 정상 |

---

## 7. GitHub CLI로 프로젝트 올리기

### 7-1. 준비: `.gitignore`

모델 가중치(`*.pt`)는 용량이 크고 자동 다운로드되므로 **저장소에 올리지 않습니다.**

```gitignore
__pycache__/
*.pyc
*.pt
runs/
.venv/
venv/
```

### 7-2. GitHub CLI 로그인 확인

```bash
gh auth status
```

> ⚠️ Windows에서 Git Bash가 `gh`를 못 찾는 경우가 있었습니다(`gh: command not found`). 설치 경로 `C:\Program Files\GitHub CLI\gh.exe`를 PowerShell에서 직접 실행하거나, PATH에 추가하세요.

### 7-3. 커밋 → 저장소 생성 → 푸시

```bash
git branch -M main                       # 기본 브랜치 이름을 main으로
git add .gitignore README.md requirements.txt common.py detect.py segment.py track.py
git commit -m "Add YOLO26 webcam demos: detection, segmentation, tracking"

gh repo create yolo26-webcam-demos --public \
  --description "Real-time webcam detection, segmentation and tracking with Ultralytics YOLO26" \
  --source . --remote origin --push
```

| `gh repo create` 옵션 | 의미 |
|---|---|
| `--public` / `--private` | 공개 범위 |
| `--source .` | 현재 폴더(로컬 저장소)를 원본으로 사용 |
| `--remote origin` | 원격 이름을 `origin`으로 등록 |
| `--push` | 만들자마자 바로 푸시 |

결과: **https://github.com/taemin2000min-debug/yolo26-webcam-demos**

> 💡 `git add .` 대신 **파일을 하나씩 지정**하면, 모델 파일이나 캐시가 실수로 올라가는 것을 한 번 더 막을 수 있습니다.

---

## 8. 트러블슈팅 노트

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | `pip`와 `python`이 서로 다른 인터프리터 | `python -m pip install ...` 또는 `py -3.12 script.py` |
| `웹캠(0)을 열 수 없습니다` | 카메라 번호가 다르거나 다른 앱이 사용 중 | `--camera 1` 시도, Zoom/Teams 등 종료, Windows 카메라 권한 확인 |
| 창이 회색이거나 멈춤 | `cv2.waitKey()` 누락 | 루프 안에서 반드시 `waitKey(1)` 호출 |
| FPS가 너무 낮음 | CPU 추론 + 큰 모델 | `n` 모델 사용, `--imgsz 320`, GPU가 있으면 `--device 0` |
| 추적 ID가 계속 바뀜 | `persist=True` 누락 | `model.track(..., persist=True)` |
| `requirements: lap not found` | 추적기 의존성 누락 | `pip install lap` (requirements.txt에 포함됨) |
| `gh: command not found` (Git Bash) | PATH에 GitHub CLI 없음 | PowerShell에서 실행 또는 PATH 추가 |

---

## 9. 정리 & 실습 과제

### 9-1. 핵심 요약

```
탐지   : YOLO("yolo26n.pt").predict(frame)
분할   : YOLO("yolo26n-seg.pt").predict(frame)          ← 모델 파일만 교체
추적   : YOLO("yolo26n.pt").track(frame, persist=True)  ← 메서드만 교체
시각화 : result.plot()                                   ← 세 작업 모두 동일
```

1. Ultralytics는 **작업이 달라도 API가 같다** — 모델 파일과 메서드만 바꾸면 된다.
2. 실시간 처리는 **읽기 → 추론 → 그리기 → 출력** 루프다.
3. 추적은 **탐지 + 프레임 간 연결**이며, `persist=True`가 핵심이다.
4. 겹치는 코드는 공통 모듈로 빼고, **바뀌는 부분만 함수로 주입**한다.
5. 설치는 **`python -m pip`** 로, 업로드 전에는 **`.gitignore`** 부터 확인한다.

### 9-2. 실습 과제

| 난이도 | 과제 | 힌트 |
|:---:|---|---|
| ⭐ | 사람(`person`)만 탐지하기 | `predict(..., classes=[0])` |
| ⭐ | 모델 크기(`n`→`s`→`m`)를 바꿔 가며 FPS·정확도 비교표 만들기 | `--model yolo26s.pt` |
| ⭐⭐ | 자세 추정 스크립트 `pose.py` 추가하기 | `yolo26n-pose.pt` + `common.run_webcam` 재사용 |
| ⭐⭐ | 탐지 결과를 동영상 파일로 저장하기 | `cv2.VideoWriter` |
| ⭐⭐⭐ | 화면 가운데 가상의 선을 긋고, 지나간 사람 수 세기 | `track.py`의 `trails`로 이전/현재 위치 비교 |
| ⭐⭐⭐ | 분할 마스크로 사람만 남기고 배경 흐리기 | `result.masks.data`, `cv2.GaussianBlur` |

---

## 부록 — 명령어 & 옵션 요약

### 실행

```bash
pip install -r requirements.txt   # (권장) python -m pip install -r requirements.txt

python detect.py     # 객체 탐지
python segment.py    # 인스턴스 분할
python track.py      # 객체 추적
```

영상 창에서 `q` 또는 `ESC`를 누르면 종료합니다. 모델 가중치는 처음 실행할 때 자동으로 다운로드됩니다.

### 공통 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--model` | 모델 가중치 (`n`/`s`/`m`/`l`/`x` 크기 선택) | 스크립트별 상이 |
| `--camera` | 웹캠 인덱스 | `0` |
| `--conf` | 신뢰도 임계값 | `0.25` |
| `--imgsz` | 추론 이미지 크기 | `640` |
| `--device` | 추론 장치 (`cpu`, `0` 등) | 자동 |

### `track.py` 전용 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--tracker` | `bytetrack.yaml` 또는 `botsort.yaml` | `bytetrack.yaml` |
| `--trail` | 이동 경로 길이 (프레임 수, `0`이면 끔) | `30` |

### 참고 자료

- Ultralytics YOLO26 문서: https://docs.ultralytics.com/models/yolo26/
- Ultralytics 추적(Track) 모드: https://docs.ultralytics.com/modes/track/
- GitHub CLI 매뉴얼: https://cli.github.com/manual/
