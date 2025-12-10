# fourcut-classifier

네컷사진을 판별하기 위한 단층 퍼셉트론(SLP) 기반 머신러닝 모델 구현 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 numpy를 기반으로 단층 퍼셉트론(Single Layer Perceptron)을 직접 구현하여 네컷사진을 판별하는 이진 분류 모델을 제공합니다.

### 주요 기능

- **이미지 전처리**: 리사이즈, 그레이스케일 변환, 정규화, 벡터화
- **SLP 모델**: numpy 기반 단층 퍼셉트론 구현
- **학습 스크립트**: 명령줄 인터페이스를 통한 모델 학습
- **평가 스크립트**: 학습된 모델로 새로운 이미지 판별
- **Jupyter 노트북**: 예제 코드 및 시각화

## 프로젝트 구조

```
fourcut-classifier/
├── data/                      # 데이터 디렉토리
│   ├── raw/                   # 원본 이미지
│   │   ├── fourcut/           # 네컷사진 (label=1)
│   │   └── non_fourcut/       # 일반 사진 (label=0)
│   └── processed/             # 전처리된 데이터 및 모델
│       └── model_weights.npz  # 학습된 모델 가중치
├── src/                       # 소스 코드
│   ├── __init__.py            # 패키지 초기화
│   ├── preprocessing.py       # 이미지 전처리 모듈
│   ├── model.py               # SLP 모델 구현
│   ├── train.py               # 학습 스크립트
│   └── evaluate.py            # 평가 스크립트
├── notebooks/                 # Jupyter 노트북
│   └── example_usage.ipynb    # 사용 예제 노트북
├── requirements.txt           # 의존성 패키지
└── README.md                  # 프로젝트 문서
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/fourcut-classifier.git
cd fourcut-classifier
```

### 2. 가상 환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 준비

학습을 위한 이미지 데이터를 준비합니다:

```bash
# 네컷사진 이미지를 data/raw/fourcut/ 폴더에 추가
# 일반 사진 이미지를 data/raw/non_fourcut/ 폴더에 추가
```

지원 형식: `.jpg`, `.jpeg`, `.png`

### 2. 모델 학습

기본 설정으로 학습:

```bash
python src/train.py
```

커스텀 설정으로 학습:

```bash
python src/train.py \
    --data-dir data/raw \
    --image-size 64 \
    --learning-rate 0.01 \
    --epochs 100 \
    --test-ratio 0.2 \
    --save-path data/processed/model_weights.npz
```

#### 학습 파라미터

- `--data-dir`: 데이터 디렉토리 경로 (기본값: `data/raw`)
- `--image-size`: 이미지 리사이즈 크기 (기본값: `64`)
- `--learning-rate`: 학습률 (기본값: `0.01`)
- `--epochs`: 학습 에포크 수 (기본값: `100`)
- `--test-ratio`: 테스트 세트 비율 (기본값: `0.2`)
- `--random-state`: 난수 시드 (기본값: `42`)
- `--save-path`: 모델 가중치 저장 경로 (기본값: `data/processed/model_weights.npz`)

### 3. 모델 평가

단일 이미지 예측:

```bash
python src/evaluate.py --image-path path/to/image.jpg
```

디렉토리 내 모든 이미지 예측:

```bash
python src/evaluate.py --image-dir path/to/images/
```

#### 평가 파라미터

- `--model-path`: 모델 가중치 경로 (기본값: `data/processed/model_weights.npz`)
- `--image-path`: 예측할 단일 이미지 경로
- `--image-dir`: 예측할 이미지 디렉토리 경로
- `--image-size`: 이미지 리사이즈 크기 (기본값: `64`, 학습 시와 동일해야 함)

### 4. Jupyter 노트북 사용

예제 노트북을 실행하여 전체 과정을 확인할 수 있습니다:

```bash
jupyter notebook notebooks/example_usage.ipynb
```

## 모델 구조

### 단층 퍼셉트론 (Single Layer Perceptron)

```
입력 레이어 (64x64=4096) → 가중치 → 시그모이드 활성화 → 출력 (0 or 1)
```

- **입력**: 64x64 그레이스케일 이미지를 1D 벡터로 변환 (4096 차원)
- **활성화 함수**: Sigmoid
- **손실 함수**: Binary Cross-Entropy
- **최적화**: Gradient Descent

## 전처리 파이프라인

1. **이미지 로드**: PIL을 사용하여 이미지 로드
2. **리사이즈**: 64x64 픽셀로 크기 조정
3. **그레이스케일 변환**: RGB → 그레이스케일
4. **정규화**: 픽셀 값을 0-1 범위로 변환
5. **벡터화**: 2D 배열을 1D 벡터로 변환

## 코드 예제

### Python 스크립트에서 사용

```python
from src.preprocessing import ImagePreprocessor
from src.model import SingleLayerPerceptron
import numpy as np

# 전처리기 초기화
preprocessor = ImagePreprocessor(
    target_size=(64, 64),
    grayscale=True,
    normalize=True
)

# 이미지 전처리
image_vector = preprocessor.preprocess('path/to/image.jpg')

# 모델 초기화 및 학습
model = SingleLayerPerceptron(
    input_size=64*64,
    learning_rate=0.01,
    epochs=100
)

# 학습 (X: 특징, y: 레이블)
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 모델 저장
model.save_weights('model_weights.npz')

# 모델 로드
model.load_weights('model_weights.npz')
```

## 성능 향상 팁

1. **데이터 증강**: 회전, 반전, 밝기 조정 등으로 데이터 다양성 증가
2. **학습률 조정**: 학습률을 조정하여 수렴 속도 개선
3. **에포크 수 증가**: 더 많은 에포크로 학습하여 성능 향상
4. **이미지 크기 조정**: 더 큰 이미지 크기로 더 많은 정보 활용
5. **충분한 데이터**: 각 클래스당 최소 100개 이상의 이미지 권장

## 기술 스택

- **Python 3.8+**
- **NumPy**: 수치 연산 및 배열 처리
- **Pillow (PIL)**: 이미지 로드 및 전처리
- **Matplotlib**: 데이터 시각화
- **scikit-learn**: 데이터 분할 및 평가 메트릭
- **Jupyter**: 대화형 노트북

## 라이센스

MIT License

## 기여

이슈 및 풀 리퀘스트는 언제나 환영합니다!

## 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 등록해주세요.
