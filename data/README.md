# 데이터 폴더 구조

## 📁 폴더 설명

이 폴더는 네컷사진 판별 모델의 학습 및 테스트에 사용되는 이미지 데이터를 저장합니다.

```
data/
├── train/              # 학습용 데이터
│   ├── fourcut/       # 네컷 사진 (label=1)
│   └── normal/        # 일반 사진 (label=0)
│
└── test/               # 테스트용 데이터
    ├── fourcut/       # 네컷 사진 (label=1)
    └── normal/        # 일반 사진 (label=0)
```

## 📌 데이터 준비

### 네컷 사진 생성기 🎨

**문제:** 공개된 네컷 사진 데이터셋이 존재하지 않음

**해결:** 자동 네컷 사진 생성기 개발 (`generate_fourcut.py`)

네컷 사진을 직접 생성하여 학습 데이터를 확보했습니다:

```bash
# 프로젝트 루트에서 실행
python generate_fourcut.py \
  --source-dir data/raw/normal \
  --output-dir data/train/fourcut \
  --num-images 10000
```

**생성 원리:**
1. 투명 네컷 프레임 로드 (4개 슬롯)
2. 일반 사진 4장 랜덤 선택
3. 각 사진을 슬롯에 맞게 crop & resize
4. 프레임에 합성하여 네컷 완성

**장점:**
- 무제한 데이터 생성 가능
- 다양한 프레임 스타일 사용 가능
- 실제 네컷과 유사한 품질

### 학습 데이터 (train/)

**최소 요구사항:**
- `fourcut/`: 네컷 사진 500장 이상
- `normal/`: 일반 사진 500장 이상

**권장 사항:**
- 두 클래스의 개수를 비슷하게 유지 (1:1 비율)
- 다양한 종류의 이미지 포함
- 고품질 이미지 사용

### 테스트 데이터 (test/)

**최소 요구사항:**
- `fourcut/`: 네컷 사진 20장 이상
- `normal/`: 일반 사진 20장 이상

**주의사항:**
- 학습 데이터와 겹치지 않는 이미지 사용
- 실제 성능 평가를 위해 다양한 이미지 포함

## 🖼️ 지원 이미지 형식

- `.jpg`, `.jpeg`
- `.png`

## ⚠️ 주의사항

1. **네컷 사진 (fourcut/)**
   - 포토부스에서 찍은 격자 형태의 사진
   - 보통 2×2 또는 4컷 구조

2. **일반 사진 (normal/)**
   - 네컷이 아닌 모든 일반 사진
   - 풍경, 인물, 음식, 사물 등
   - 콜라주나 격자 형태 이미지는 제외 권장

## 📊 현재 데이터 분포 확인

프로젝트 루트에서 다음 명령어로 데이터 분포를 확인할 수 있습니다:

```python
# Python 스크립트로 확인 (예정)
python check_data.py
```

또는 직접 확인:

```bash
# Windows (PowerShell)
(Get-ChildItem data/train/fourcut).Count
(Get-ChildItem data/train/normal).Count

# Linux/Mac
ls data/train/fourcut | wc -l
ls data/train/normal | wc -l
```

## 💡 데이터 불균형 해결

학습 시 자동으로 데이터 밸런싱이 적용됩니다:

```bash
# 언더샘플링 (기본값)
python src/train.py --balance-method undersample

# 오버샘플링
python src/train.py --balance-method oversample

# 밸런싱 비활성화
python src/train.py --no-balance
```

---

**데이터 준비가 완료되면 학습을 시작하세요!**

