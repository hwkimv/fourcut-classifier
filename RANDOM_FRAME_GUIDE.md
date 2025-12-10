# 🎨 랜덤 프레임 선택 기능 사용 가이드

## ✅ 업데이트 완료!

`generate_fourcut.py`가 이제 **여러 프레임 중에서 랜덤으로 선택**하는 기능을 지원합니다!

## 📂 프레임 폴더 준비

프레임들을 다음 경로에 넣어주세요:

```
data/raw/fourcut_frame/
├── photo-booth-frame-cute.png
├── photo-booth-frame-pink-cute.png
├── photo-booth-frame-transparent_20px.png
└── ... (더 많은 프레임)
```

## 🚀 사용 방법

### 1. 기본 실행 (랜덤 프레임 선택)

```bash
python generate_fourcut.py --num-images 100
```

- 기본적으로 `data/raw/fourcut_frame/` 폴더에서 랜덤 선택
- 매 이미지마다 다른 프레임이 사용될 수 있음

### 2. 커스텀 프레임 폴더 지정

```bash
python generate_fourcut.py --frame-dir my_frames --num-images 100
```

### 3. 단일 프레임만 사용 (기존 방식)

```bash
python generate_fourcut.py --frame-path photo-booth-frame-transparent_20px.png --num-images 100
```

- `--frame-path` 옵션을 사용하면 지정한 프레임만 사용

## 📊 실행 예시

```bash
$ python generate_fourcut.py --num-images 5

네컷 포토부스 합성 이미지 생성을 시작합니다...
  - 소스 디렉토리: data/raw/normal
  - 출력 디렉토리: data/train/fourcut
  - 생성할 이미지 개수: 5
  - 슬롯 탐지 방식: hardcoded
  - 프레임 모드: 랜덤 선택
  - 프레임 디렉토리: data/raw/fourcut_frame

사용 가능한 프레임: 3개
  - photo-booth-frame-cute.png
  - photo-booth-frame-pink-cute.png
  - photo-booth-frame-transparent_20px.png

하드코딩된 슬롯 사용: 4개

소스 이미지 로딩 중...
사용 가능한 소스 이미지: 4930개

네컷 이미지 생성 시작...

[1/5] fourcut_0001.png 생성 완료 (프레임: photo-booth-frame-transparent_20px.png)
[2/5] fourcut_0002.png 생성 완료 (프레임: photo-booth-frame-cute.png)
[3/5] fourcut_0003.png 생성 완료 (프레임: photo-booth-frame-cute.png)
[4/5] fourcut_0004.png 생성 완료 (프레임: photo-booth-frame-transparent_20px.png)
[5/5] fourcut_0005.png 생성 완료 (프레임: photo-booth-frame-cute.png)

✓ 네컷 이미지 생성 완료!
  총 5개 이미지가 data/train/fourcut에 저장되었습니다.
  3개의 프레임이 랜덤으로 사용되었습니다.
```

## 🎯 주요 기능

### ✨ 새로운 기능

1. **랜덤 프레임 선택** (기본값)
   - `data/raw/fourcut_frame/` 폴더의 모든 PNG 파일 중 랜덤 선택
   - 매 이미지마다 다른 프레임 사용 가능
   - 다양한 스타일의 네컷 이미지 자동 생성

2. **프레임 폴더 지정**
   - `--frame-dir` 옵션으로 커스텀 폴더 사용

3. **로그에 프레임 이름 표시**
   - 어떤 프레임이 사용되었는지 확인 가능
   - 예: `[1/5] fourcut_0001.png 생성 완료 (프레임: photo-booth-frame-cute.png)`

### 📌 기존 기능 (유지)

- 단일 프레임 사용: `--frame-path` 옵션
- 자동/수동 슬롯 탐지: `--slot-mode` 옵션
- 랜덤 시드 고정: `--random-seed` 옵션

## 🔧 옵션 전체 목록

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source-dir` | 소스 이미지 폴더 | `data/raw/normal` |
| `--frame-dir` | 프레임 폴더 (랜덤 선택) | `data/raw/fourcut_frame` |
| `--frame-path` | 단일 프레임 경로 | `None` (랜덤 모드) |
| `--output-dir` | 출력 폴더 | `data/train/fourcut` |
| `--num-images` | 생성할 이미지 개수 | `100` |
| `--random-seed` | 랜덤 시드 | `42` |
| `--slot-mode` | 슬롯 탐지 방식 | `hardcoded` |

## ⚠️ 주의사항

1. **`--frame-path`와 `--frame-dir`는 동시 사용 불가**
   ```bash
   # ❌ 에러
   python generate_fourcut.py --frame-path a.png --frame-dir frames/
   
   # ✅ 정상
   python generate_fourcut.py --frame-path a.png
   # 또는
   python generate_fourcut.py --frame-dir frames/
   ```

2. **프레임 파일 형식**
   - PNG 파일만 인식 (`.png`, `.PNG`)
   - RGBA 모드 권장 (투명도 포함)

3. **슬롯 좌표**
   - 현재는 하드코딩된 슬롯 사용
   - 모든 프레임이 동일한 슬롯 크기/위치를 가져야 함
   - 프레임마다 슬롯이 다르면 `--slot-mode auto` 사용 권장

## 💡 사용 팁

### 1. 대량 생성 (다양한 프레임)

```bash
# 500장 생성, 3개 프레임 랜덤 사용
python generate_fourcut.py --num-images 500
```

### 2. 특정 프레임만 사용

```bash
# 귀여운 프레임만 100장
python generate_fourcut.py \
  --frame-path data/raw/fourcut_frame/photo-booth-frame-cute.png \
  --num-images 100
```

### 3. 프레임별 슬롯 자동 탐지

```bash
# 프레임마다 슬롯 위치가 다른 경우
python generate_fourcut.py --slot-mode auto --num-images 100
```

⚠️ 단, auto 모드는 프레임 디자인에 따라 정확도가 달라질 수 있습니다.

### 4. 재현 가능한 랜덤 생성

```bash
# 동일한 시드로 같은 결과 재현
python generate_fourcut.py --random-seed 123 --num-images 50
```

## 📁 프로젝트 구조

```
4cut_classifier/
├── generate_fourcut.py          # ⭐ 메인 스크립트 (업데이트됨!)
├── data/
│   └── raw/
│       ├── fourcut_frame/       # 🎨 프레임 폴더 (여기에 프레임 추가)
│       │   ├── photo-booth-frame-cute.png
│       │   ├── photo-booth-frame-pink-cute.png
│       │   └── photo-booth-frame-transparent_20px.png
│       └── normal/              # 📸 소스 이미지 폴더
└── ...
```

## 🎓 활용 예시

### 학습 데이터 다양화

```bash
# 1000장 생성 - 여러 프레임 스타일 포함
python generate_fourcut.py --num-images 1000
```

→ 다양한 프레임으로 학습하면 분류기가 프레임 종류에 상관없이 네컷을 인식할 수 있습니다.

### 프레임별 테스트 데이터

```bash
# 프레임 A로 100장
python generate_fourcut.py --frame-path frameA.png --num-images 100 --output-dir test/frameA

# 프레임 B로 100장
python generate_fourcut.py --frame-path frameB.png --num-images 100 --output-dir test/frameB
```

## 🆕 업데이트 내역

### v2.0 (2024-12-11)

**새로운 기능:**
- ✅ 프레임 폴더에서 랜덤 선택 기능 추가
- ✅ `--frame-dir` 옵션 추가 (기본: `data/raw/fourcut_frame`)
- ✅ `load_frames()` 함수 추가
- ✅ 로그에 사용된 프레임 이름 표시
- ✅ 최종 요약에 프레임 개수 표시

**호환성:**
- ✅ 기존 `--frame-path` 옵션 유지 (하위 호환성)
- ✅ 단일/랜덤 프레임 모드 자동 전환

---

**이제 프레임을 마음껏 추가하고 다양한 네컷 이미지를 생성하세요!** 🎉

