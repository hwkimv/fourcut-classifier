# 단층 퍼셉트론(SLP) 모델
# 네컷 사진인지 일반 사진인지 구분하는 AI 모델

import numpy as np


class SingleLayerPerceptron:
    """
    단층 퍼셉트론: 가장 간단한 인공지능 모델
    - 사진을 보고 네컷인지 일반 사진인지 판단합니다
    """

    SIGMOID_CLIP_RANGE = 500  # 계산 오류 방지용 숫자

    def __init__(self, input_size, learning_rate=0.01, epochs=100, random_state=42):
        # 모델 설정값 저장
        self.input_size = input_size        # 입력 데이터 크기
        self.learning_rate = learning_rate  # 학습 속도 (너무 크면 불안정, 너무 작으면 느림)
        self.epochs = epochs                # 학습 반복 횟수
        self.random_state = random_state    # 랜덤 시드 (재현성 위해)

        # 가중치 초기화 (모델의 뇌 역할)
        np.random.seed(self.random_state)
        self.weights = np.random.randn(input_size) * 0.01  # 작은 랜덤값으로 시작
        self.bias = 0.0  # 편향 (기본 성향)

        # 학습 기록 저장
        self.loss_history = []      # 손실 기록
        self.accuracy_history = []  # 정확도 기록

    def activation(self, z):
        """
        시그모이드 함수: 어떤 숫자든 0~1 사이로 변환
        - 0.5보다 크면 네컷, 작으면 일반으로 판단
        """
        return 1 / (1 + np.exp(-np.clip(z, -self.SIGMOID_CLIP_RANGE, self.SIGMOID_CLIP_RANGE)))

    def predict_proba(self, X):
        """
        확률 예측: 네컷일 확률을 0~1 사이 숫자로 계산
        """
        z = np.dot(X, self.weights) + self.bias  # 가중치와 입력을 곱해서 더함
        return self.activation(z)  # 0~1 사이로 변환

    def predict(self, X, threshold=0.5):
        """
        클래스 예측: 네컷(1) 또는 일반(0)으로 최종 결정
        - threshold: 기준값 (기본 0.5)
        """
        proba = self.predict_proba(X)  # 확률 계산
        return (proba >= threshold).astype(int)  # 기준값보다 크면 1, 아니면 0

    def compute_loss(self, y_true, y_pred_proba):
        """
        손실 계산: 모델이 얼마나 틀렸는지 측정
        - 숫자가 작을수록 잘 맞춤
        """
        return float(np.mean((y_true - y_pred_proba) ** 2))

    def compute_accuracy(self, y_true, y_pred):
        """
        정확도 계산: 전체 중 몇 개를 맞췄는지
        - 1.0 = 100% 정확, 0.5 = 50% 정확
        """
        return np.mean(y_true == y_pred)

    def fit(self, X, y, verbose=True):
        """
        모델 학습: 데이터를 보고 패턴을 배움
        """
        n_samples = X.shape[0]  # 데이터 개수

        for epoch in range(self.epochs):
            # 1단계: 현재 가중치로 예측
            y_pred_proba = self.predict_proba(X)

            # 2단계: 손실과 정확도 계산
            loss = self.compute_loss(y, y_pred_proba)
            self.loss_history.append(loss)

            y_pred = (y_pred_proba >= 0.5).astype(int)
            accuracy = self.compute_accuracy(y, y_pred)
            self.accuracy_history.append(accuracy)

            # 3단계: 오차 계산 (얼마나 틀렸는지)
            error = (y_pred_proba - y) * y_pred_proba * (1 - y_pred_proba)
            dw = np.dot(X.T, error) / n_samples  # 가중치 변화량
            db = np.mean(error)  # 편향 변화량

            # 4단계: 가중치 업데이트 (조금씩 수정)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 진행 상황 출력 (10번마다)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    def evaluate(self, X, y):
        """
        모델 평가: 테스트 데이터로 성능 측정
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        loss = self.compute_loss(y, y_pred_proba)
        accuracy = self.compute_accuracy(y, y_pred)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

