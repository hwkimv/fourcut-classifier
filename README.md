# 네컷사진 판별 SLP 프로젝트

단층 퍼셉트론(SLP)을 numpy만 사용해 직접 구현하여 네컷사진(fourcut) 여부를 이진 분류하는 학습용 템플릿입니다.

## 프로젝트 구조
```
./
├─ src/
│  ├─ preprocessing.py   # 이미지 전처리 (리사이즈, 그레이스케일, 정규화, flatten)
│  ├─ model.py           # SLP 모델 클래스 (sigmoid, MSE, 경사하강법)
│  ├─ train.py           # 학습 스크립트 (로그, 가중치/편향 저장)
│  ├─ evaluate.py        # 전체 테스트셋 정확도 평가
│  ├─ test_one.py        # 단일 이미지 판별
│  └─ utils.py           # 로깅/저장/불러오기 유틸
├─ data/
│  ├─ train/
│  │  ├─ fourcut/
│  │  └─ normal/         # (하위 호환: non_fourcut/ 도 허용)
│  └─ test/
│     ├─ fourcut/
│     └─ normal/         # (하위 호환: non_fourcut/ 도 허용)
├─ model/                # 학습 후 model.npy, bias.npy 저장
├─ logs/                 # train_log.txt 저장
└─ main.ipynb (선택)
```

## 환경
- Python 3.8+
- NumPy, Pillow, Matplotlib (간단 시각화 용도)
- (선택) Jupyter Notebook

설치:
```bash
pip install -r requirements.txt
```

## 데이터 준비
- 학습용 이미지를 `data/train/fourcut/`, `data/train/normal/` 에 넣습니다.
- 테스트 이미지를 `data/test/fourcut/`, `data/test/normal/` 에 넣습니다.
- 허용 확장자: `.jpg`, `.jpeg`, `.png`

## 학습 방법 (train.py)
```bash
python src/train.py --data-dir data/train --image-size 64 --learning-rate 0.01 --epochs 100 --test-ratio 0.2 --random-state 42 --model-dir model --log-path logs/train_log.txt
```
학습이 완료되면 다음 산출물이 생성됩니다:
- `model/model.npy`  (가중치 W)
- `model/bias.npy`   (편향 b)
- `logs/train_log.txt` (에폭별 loss 기록, 탭 구분: epoch\tloss)

## 단일 이미지 테스트 (test_one.py)
```bash
python src/test_one.py path/to/image.jpg --model-dir model --image-size 64
```
콘솔에 "네컷/일반"과 예측 확률이 출력됩니다.

## 성능 평가 (evaluate.py)
```bash
python src/evaluate.py --data-dir data/test --model-dir model --image-size 64
```
전체 테스트셋의 정확도를 출력합니다.

## 모델/학습 개요
- 모델: 입력(64x64 => 4096) → 선형결합(Wx+b) → Sigmoid → 출력확률
- 손실: 평균제곱오차(MSE)
- 최적화: 경사하강법(수동 구현)
- 전처리: PIL 기반 로드 → 리사이즈(64x64) → 그레이스케일 → [0,1] 정규화 → 1D flatten

## 참고
- 디렉토리 명칭은 `normal`을 권장하나, 기존 데이터가 `non_fourcut`인 경우도 처리하도록 스크립트에서 하위 호환을 제공합니다.
