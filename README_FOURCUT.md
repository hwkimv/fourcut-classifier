# 네컷 포토부스 합성 이미지 생성기

프레임 PNG의 투명 슬롯(4개)에 일반 사진을 자동으로 합성하여 네컷 이미지를 생성하는 Python 스크립트입니다.

## 주요 기능

✅ 투명 PNG 네컷 프레임(frame.png) 로드  
✅ 투명 슬롯 자동 탐지 또는 하드코딩 좌표 사용  
✅ 일반 사진 폴더에서 랜덤으로 4장 선택  
✅ Center-crop + resize로 슬롯 크기에 자동 맞춤  
✅ 프레임 레이어를 최상단에 덮어씌워 합성  
✅ 자동 파일명(fourcut_0001.png 형식) 저장  

## 설치 방법

### 1. 필수 패키지 설치

```bash
pip install Pillow numpy
```

또는 requirements.txt가 있다면:

```bash
pip install -r requirements.txt
```

### 2. 프로젝트 구조 확인

```
4cut_classifier/
├── generate_fourcut.py      # 메인 스크립트
├── frame.png                 # 네컷 프레임 (투명 슬롯 4개 포함)
├── data/
│   ├── raw/
│   │   └── normal/          # 일반 사진 폴더 (소스 이미지)
│   └── train/
│       └── fourcut/         # 생성된 네컷 이미지 저장
└── README_FOURCUT.md        # 이 파일
```

## 사용 방법

### 기본 실행 (하드코딩된 슬롯 사용)

```bash
python generate_fourcut.py
```

기본 옵션:
- `--source-dir data/raw/normal`
- `--frame-path frame.png`
- `--output-dir data/train/fourcut`
- `--num-images 100`
- `--random-seed 42`
- `--slot-mode hardcoded`

### 커스텀 옵션으로 실행

```bash
python generate_fourcut.py \
  --source-dir data/raw/normal \
  --frame-path my_frame.png \
  --output-dir output/fourcut \
  --num-images 50 \
  --random-seed 123 \
  --slot-mode auto
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source-dir` | 합성에 사용할 일반 사진 폴더 | `data/raw/normal` |
| `--frame-path` | 네컷 프레임 PNG 경로 | `frame.png` |
| `--output-dir` | 결과 저장 폴더 | `data/train/fourcut` |
| `--num-images` | 생성할 네컷 이미지 개수 | `100` |
| `--random-seed` | 랜덤 시드 고정 값 | `42` |
| `--slot-mode` | 슬롯 탐지 방식 (`hardcoded` 또는 `auto`) | `hardcoded` |

## 슬롯 탐지 방식

### 방식 A: 하드코딩 (권장) - `--slot-mode hardcoded`

코드 상단의 `SLOTS_HARDCODED` 변수를 실제 프레임에 맞게 수정:

```python
SLOTS_HARDCODED: List[Tuple[int, int, int, int]] = [
    # (x, y, width, height) - 위에서 아래 순서
    (100, 80, 800, 800),      # 1번째 칸
    (100, 930, 800, 800),     # 2번째 칸
    (100, 1780, 800, 800),    # 3번째 칸
    (100, 2630, 800, 800),    # 4번째 칸
]
```

**슬롯 좌표 확인 방법:**
1. 프레임 PNG를 이미지 편집기(Photoshop, GIMP 등)로 열기
2. 각 투명 슬롯의 좌측 상단 좌표(x, y)와 크기(w, h) 확인
3. 코드의 `SLOTS_HARDCODED` 값을 수정

### 방식 B: 자동 탐지 - `--slot-mode auto`

PNG의 alpha 채널에서 `alpha == 0`인 투명 영역을 자동으로 탐지합니다.

⚠️ **주의사항:**
- 프레임 디자인에 따라 정확도가 달라질 수 있습니다
- 슬롯이 정확히 4개 직사각형이어야 합니다
- 복잡한 디자인은 하드코딩 방식 권장

## 프레임 준비하기

### 프레임 PNG 요구사항

1. **RGBA 모드** PNG 파일
2. **투명 슬롯 4개** (알파 채널 = 0)
3. **세로형 배치** (위에서 아래로 4개 슬롯)
4. **권장 해상도:** 1000x4000 이상

### 예시 프레임 구조

```
┌──────────────┐
│   프레임     │
│ ┌──────────┐ │  ← 슬롯 1 (투명)
│ └──────────┘ │
│   꾸미기     │
│ ┌──────────┐ │  ← 슬롯 2 (투명)
│ └──────────┘ │
│   프레임     │
│ ┌──────────┐ │  ← 슬롯 3 (투명)
│ └──────────┘ │
│   장식       │
│ ┌──────────┐ │  ← 슬롯 4 (투명)
│ └──────────┘ │
└──────────────┘
```

## 소스 이미지 준비

### 소스 폴더 구성

`data/raw/normal/` 폴더에 일반 사진(jpg/png)을 넣어주세요:

```
data/raw/normal/
├── photo_001.jpg
├── photo_002.jpg
├── photo_003.png
├── ...
└── photo_100.jpg
```

### 이미지 요구사항

- **형식:** JPG, JPEG, PNG
- **최소 크기:** 슬롯보다 큰 이미지 권장
- **자동 처리:** RGB 변환, center-crop, resize, upscale 모두 자동

## 실행 예시

### 1. 테스트 실행 (10장만 생성)

```bash
python generate_fourcut.py --num-images 10
```

### 2. 대량 생성 (500장)

```bash
python generate_fourcut.py --num-images 500
```

### 3. 자동 슬롯 탐지 모드

```bash
python generate_fourcut.py --slot-mode auto --num-images 20
```

### 4. 커스텀 경로 지정

```bash
python generate_fourcut.py \
  --source-dir /path/to/photos \
  --frame-path /path/to/frame.png \
  --output-dir /path/to/output
```

## 출력 결과

### 생성 로그 예시

```
네컷 포토부스 합성 이미지 생성을 시작합니다...
  - 소스 디렉토리: data/raw/normal
  - 프레임 경로: frame.png
  - 출력 디렉토리: data/train/fourcut
  - 생성할 이미지 개수: 100
  - 슬롯 탐지 방식: hardcoded

프레임 이미지 로드 완료: 1000x4000
하드코딩된 슬롯 사용: 4개

소스 이미지 로딩 중...
사용 가능한 소스 이미지: 250개

네컷 이미지 생성 시작...

[1/100] fourcut_0001.png 생성 완료
[2/100] fourcut_0002.png 생성 완료
[3/100] fourcut_0003.png 생성 완료
...
[100/100] fourcut_0100.png 생성 완료

✓ 네컷 이미지 생성 완료!
  총 100개 이미지가 data/train/fourcut에 저장되었습니다.
```

### 출력 파일

```
data/train/fourcut/
├── fourcut_0001.png
├── fourcut_0002.png
├── fourcut_0003.png
├── ...
└── fourcut_0100.png
```

## 문제 해결

### 1. "프레임 이미지를 찾을 수 없습니다"

**원인:** `frame.png` 파일이 없음

**해결:**
```bash
# 프레임 경로 확인
ls frame.png

# 또는 경로 직접 지정
python generate_fourcut.py --frame-path /path/to/frame.png
```

### 2. "source_dir가 존재하지 않습니다"

**원인:** 소스 이미지 폴더가 없음

**해결:**
```bash
# 폴더 생성
mkdir -p data/raw/normal

# 이미지 복사
cp /path/to/photos/*.jpg data/raw/normal/
```

### 3. "슬롯 자동 탐지 실패"

**원인:** 프레임 디자인이 복잡하거나 슬롯이 정확히 4개가 아님

**해결:**
```bash
# 하드코딩 모드로 전환
python generate_fourcut.py --slot-mode hardcoded
```

코드에서 `SLOTS_HARDCODED` 값을 수정

### 4. 이미지 크기가 이상함

**원인:** 슬롯 좌표가 프레임과 맞지 않음

**해결:**
1. 이미지 편집기로 프레임 열기
2. 슬롯 좌표 정확히 측정
3. `SLOTS_HARDCODED` 값 수정

## 코드 구조

### 주요 함수

```python
load_images(source_dir)
# 소스 디렉토리에서 jpg/png 파일 목록 로드

detect_slots(frame)
# alpha=0 투명 영역에서 슬롯 4개 자동 탐지

crop_to_slot(image, target_size)
# 이미지를 슬롯 비율에 맞게 center-crop + resize

compose_fourcut(frame, slot_images, slots)
# 프레임과 4장 이미지를 합성

save_image(image, output_dir, index)
# fourcut_0001.png 형식으로 저장

main()
# 전체 프로세스 실행
```

## 라이센스

이 스크립트는 교육 목적으로 제작되었습니다.

## 문의

- 프로젝트 문제: GitHub Issues
- 기술 지원: 프로젝트 저장소 참조

