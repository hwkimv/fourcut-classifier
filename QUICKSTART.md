# 네컷 포토부스 합성 스크립트 - 빠른 시작 가이드

## 📋 체크리스트

프로젝트 실행 전 확인사항:

- [ ] Python 3.7 이상 설치
- [ ] Pillow, numpy 패키지 설치
- [ ] `frame.png` 파일 준비 (투명 슬롯 4개 포함)
- [ ] `data/raw/normal/` 폴더에 일반 사진 준비

## 🚀 3분 안에 시작하기

### 1단계: 패키지 설치

```bash
pip install Pillow numpy
```

### 2단계: 폴더 구조 생성

```bash
mkdir -p data/raw/normal
mkdir -p data/train/fourcut
```

### 3단계: 파일 준비

1. **프레임 파일** → 프로젝트 루트에 `frame.png` 배치
2. **소스 이미지** → `data/raw/normal/` 폴더에 jpg/png 이미지 복사

### 4단계: 슬롯 좌표 설정

`generate_fourcut.py` 파일을 열고 상단의 `SLOTS_HARDCODED` 값을 실제 프레임에 맞게 수정:

```python
SLOTS_HARDCODED = [
    (100, 80, 800, 800),      # 슬롯 1: x, y, width, height
    (100, 930, 800, 800),     # 슬롯 2
    (100, 1780, 800, 800),    # 슬롯 3
    (100, 2630, 800, 800),    # 슬롯 4
]
```

### 5단계: 실행!

#### Windows:
```cmd
run_generate.bat
```

또는:
```cmd
python generate_fourcut.py --num-images 10
```

#### Linux/Mac:
```bash
bash run_generate.sh
```

또는:
```bash
python generate_fourcut.py --num-images 10
```

## 📂 프로젝트 구조

```
4cut_classifier/
├── generate_fourcut.py          # ⭐ 메인 스크립트
├── frame.png                     # 네컷 프레임 (준비 필요)
├── run_generate.bat              # Windows 빠른 실행
├── run_generate.sh               # Linux/Mac 빠른 실행
├── README_FOURCUT.md             # 상세 사용 설명서
├── QUICKSTART.md                 # 이 파일
├── requirements.txt              # Python 패키지 목록
├── data/
│   ├── raw/
│   │   └── normal/              # 📸 소스 이미지 (준비 필요)
│   └── train/
│       └── fourcut/             # 🎨 생성된 네컷 이미지
└── src/                         # 분류 모델 관련 코드
```

## 🎯 자주 사용하는 명령어

### 기본 실행 (100장 생성)
```bash
python generate_fourcut.py
```

### 테스트 (10장만)
```bash
python generate_fourcut.py --num-images 10
```

### 대량 생성 (500장)
```bash
python generate_fourcut.py --num-images 500
```

### 자동 슬롯 탐지 모드
```bash
python generate_fourcut.py --slot-mode auto
```

### 커스텀 경로
```bash
python generate_fourcut.py \
  --source-dir my_photos \
  --frame-path my_frame.png \
  --output-dir output
```

### 도움말 보기
```bash
python generate_fourcut.py --help
```

## 🔧 슬롯 좌표 찾는 방법

### 방법 1: 이미지 편집기 사용 (권장)

1. Photoshop, GIMP, Paint.NET 등으로 `frame.png` 열기
2. 선택 도구로 각 투명 슬롯 영역 선택
3. 속성 패널에서 좌표(x, y)와 크기(w, h) 확인
4. 코드의 `SLOTS_HARDCODED`에 입력

### 방법 2: 자동 탐지 테스트

```bash
python generate_fourcut.py --slot-mode auto --num-images 1
```

콘솔 출력에서 탐지된 좌표를 확인하고 `SLOTS_HARDCODED`에 복사

## ❓ 문제 해결

### "프레임 이미지를 찾을 수 없습니다"

```bash
# 현재 위치 확인
pwd

# frame.png 확인
ls frame.png

# 경로 지정
python generate_fourcut.py --frame-path /full/path/to/frame.png
```

### "source_dir가 존재하지 않습니다"

```bash
# 폴더 생성
mkdir -p data/raw/normal

# 이미지 복사
cp ~/Pictures/*.jpg data/raw/normal/
```

### 생성된 이미지 위치가 이상함

→ `SLOTS_HARDCODED` 좌표를 다시 확인

### 슬롯 자동 탐지 실패

→ `--slot-mode hardcoded`로 전환 (기본값)

## 📊 실행 예시

```bash
$ python generate_fourcut.py --num-images 10

네컷 포토부스 합성 이미지 생성을 시작합니다...
  - 소스 디렉토리: data/raw/normal
  - 프레임 경로: frame.png
  - 출력 디렉토리: data/train/fourcut
  - 생성할 이미지 개수: 10
  - 슬롯 탐지 방식: hardcoded

프레임 이미지 로드 완료: 1000x4000
하드코딩된 슬롯 사용: 4개

소스 이미지 로딩 중...
사용 가능한 소스 이미지: 150개

네컷 이미지 생성 시작...

[1/10] fourcut_0001.png 생성 완료
[2/10] fourcut_0002.png 생성 완료
...
[10/10] fourcut_0010.png 생성 완료

✓ 네컷 이미지 생성 완료!
  총 10개 이미지가 data/train/fourcut에 저장되었습니다.
```

## 📚 더 알아보기

- **상세 매뉴얼:** `README_FOURCUT.md` 참조
- **코드 구조:** `generate_fourcut.py` 주석 참조
- **옵션 전체 목록:** `python generate_fourcut.py --help`

## 💡 팁

1. **처음 실행:** `--num-images 1`로 테스트 후 대량 생성
2. **슬롯 좌표:** 하드코딩 방식이 자동 탐지보다 안정적
3. **소스 이미지:** 최소 20장 이상 준비 (중복 방지)
4. **랜덤 시드:** 재현 가능한 결과를 위해 `--random-seed` 고정

## 🎓 학습 목적

이 스크립트는 다음을 학습하기 위해 만들어졌습니다:

- PIL/Pillow를 활용한 이미지 처리
- 알파 채널 기반 이미지 합성
- argparse를 통한 CLI 도구 개발
- 파일 입출력 및 경로 관리

---

**문제가 있나요?** `README_FOURCUT.md`의 "문제 해결" 섹션을 확인하세요!

