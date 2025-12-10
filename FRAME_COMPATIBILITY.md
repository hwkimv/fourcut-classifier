# ✅ photo-booth-frame-transparent_20px.png 호환성 확인 완료!

## 📊 프레임 분석 결과

- **프레임 파일:** `photo-booth-frame-transparent_20px.png`
- **프레임 크기:** 900 x 2700 픽셀
- **이미지 모드:** RGBA (투명도 포함)
- **투명 픽셀:** 2,099,200 개
- **슬롯 개수:** 4개 (자동 탐지 성공 ✅)

## 🎯 자동 탐지된 슬롯 좌표

`generate_fourcut.py`에 이미 적용되었습니다:

```python
SLOTS_HARDCODED = [
    (40, 40, 820, 640),       # 1번째 칸
    (40, 700, 820, 640),      # 2번째 칸
    (40, 1360, 820, 640),     # 3번째 칸
    (40, 2020, 820, 640),     # 4번째 칸
]
```

### 슬롯 상세 정보

| 슬롯 | X 좌표 | Y 좌표 | 너비 | 높이 | 비율 |
|------|--------|--------|------|------|------|
| 1번 | 40 | 40 | 820 | 640 | 1.28:1 |
| 2번 | 40 | 700 | 820 | 640 | 1.28:1 |
| 3번 | 40 | 1360 | 820 | 640 | 1.28:1 |
| 4번 | 40 | 2020 | 820 | 640 | 1.28:1 |

- **슬롯 비율:** 820:640 = 1.28:1 (가로로 조금 넓은 직사각형)
- **슬롯 간격:** 60px (프레임 장식 영역)

## ✅ 테스트 결과

```bash
$ python generate_fourcut.py --num-images 1

네컷 포토부스 합성 이미지 생성을 시작합니다...
  - 프레임 경로: photo-booth-frame-transparent_20px.png
  - 출력 디렉토리: data/train/fourcut

프레임 이미지 로드 완료: 900x2700
하드코딩된 슬롯 사용: 4개
사용 가능한 소스 이미지: 4개

[1/1] fourcut_0001.png 생성 완료

✓ 네컷 이미지 생성 완료!
```

**생성된 파일:** `data/train/fourcut/fourcut_0001.png` (2.3MB)

## 🚀 바로 사용 가능!

이제 프레임과 완벽하게 호환됩니다:

### 1. 기본 실행 (100장 생성)
```bash
python generate_fourcut.py
```

### 2. 대량 생성 (500장)
```bash
python generate_fourcut.py --num-images 500
```

### 3. 테스트 (10장)
```bash
python generate_fourcut.py --num-images 10
```

## 🎨 작동 방식

1. **프레임 로드:** `photo-booth-frame-transparent_20px.png` (900x2700)
2. **슬롯 좌표:** 하드코딩된 정확한 좌표 사용
3. **소스 선택:** `data/raw/normal/`에서 랜덤 4장
4. **이미지 처리:**
   - RGB 변환
   - Center-crop (비율 1.28:1 맞춤)
   - 820x640으로 리사이즈
5. **합성:** 슬롯에 붙인 후 프레임 레이어 덮어쓰기
6. **저장:** `data/train/fourcut/fourcut_XXXX.png`

## 📝 변경 사항

### `generate_fourcut.py` 업데이트 내역

1. **SLOTS_HARDCODED** 좌표를 실제 프레임에 맞게 수정
   - 기존: 예시 좌표 (100, 80, 800, 800) 등
   - 신규: 자동 탐지된 정확한 좌표 (40, 40, 820, 640) 등

2. **기본 프레임 경로** 변경
   - 기존: `frame.png`
   - 신규: `photo-booth-frame-transparent_20px.png`

## 🔍 슬롯 좌표 다시 확인하는 방법

프레임이 바뀌면 다시 탐지할 수 있습니다:

```bash
python detect_slots.py
```

또는 자동 탐지 모드로 실행:

```bash
python generate_fourcut.py --slot-mode auto --num-images 1
```

## 💡 추천 설정

### 최적의 소스 이미지

- **최소 크기:** 820x640 이상
- **권장 크기:** 1024x768 이상
- **비율:** 상관없음 (자동 center-crop)
- **형식:** JPG, PNG

### 대량 생성 시

```bash
# 500장 생성 (학습 데이터용)
python generate_fourcut.py --num-images 500

# 100장 생성 (테스트 데이터용) + 다른 시드
python generate_fourcut.py --num-images 100 --random-seed 123
```

## ✨ 결론

**완벽하게 호환됩니다!** 🎉

- ✅ 프레임 크기 인식 성공
- ✅ 4개 슬롯 자동 탐지 성공
- ✅ 정확한 좌표로 업데이트 완료
- ✅ 테스트 이미지 생성 성공

이제 `python generate_fourcut.py`만 실행하면 프레임에 딱 맞는 네컷 이미지가 자동으로 생성됩니다!

---

**생성된 파일들:**
- `generate_fourcut.py` (업데이트됨)
- `detect_slots.py` (슬롯 탐지 유틸리티)
- `data/train/fourcut/fourcut_0001.png` (테스트 결과)

