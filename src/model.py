"""
단층 퍼셉트론(Single Layer Perceptron) 모델
numpy 기반으로 구현된 간단한 SLP 분류기입니다.
- 시그모이드 활성화
- MSE 손실
- 경사하강법(가중치/편향 업데이트)
"""

import numpy as np


class SingleLayerPerceptron:
    """
    단층 퍼셉트론 분류기
    
    Args:
        input_size (int): 입력 특징의 개수
        learning_rate (float): 학습률
        epochs (int): 학습 에포크 수
        random_state (int): 난수 시드
    """
    
    # 시그모이드 함수 오버플로우 방지를 위한 클리핑 범위
    SIGMOID_CLIP_RANGE = 500
    
    def __init__(self, input_size, learning_rate=0.01, epochs=100, random_state=42):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        
        # 가중치와 편향 초기화
        np.random.seed(self.random_state)
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        
        # 학습 이력
        self.loss_history = []
        self.accuracy_history = []
    
    def activation(self, z):
        """
        시그모이드 활성화 함수
        
        Args:
            z (numpy.ndarray): 입력 값
            
        Returns:
            numpy.ndarray: 활성화된 값 (0-1 사이)
        """
        return 1 / (1 + np.exp(-np.clip(z, -self.SIGMOID_CLIP_RANGE, self.SIGMOID_CLIP_RANGE)))
    
    def predict_proba(self, X):
        """
        확률 예측
        
        Args:
            X (numpy.ndarray): 입력 데이터 (n_samples, n_features)
            
        Returns:
            numpy.ndarray: 예측 확률 (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
    
    def predict(self, X, threshold=0.5):
        """
        클래스 예측
        
        Args:
            X (numpy.ndarray): 입력 데이터 (n_samples, n_features)
            threshold (float): 분류 임계값
            
        Returns:
            numpy.ndarray: 예측된 클래스 (0 또는 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def compute_loss(self, y_true, y_pred_proba):
        """MSE 손실 계산"""
        return float(np.mean((y_true - y_pred_proba) ** 2))

    def compute_accuracy(self, y_true, y_pred):
        """
        정확도 계산
        
        Args:
            y_true (numpy.ndarray): 실제 레이블
            y_pred (numpy.ndarray): 예측 레이블
            
        Returns:
            float: 정확도 (0-1)
        """
        return np.mean(y_true == y_pred)
    
    def fit(self, X, y, verbose=True):
        """
        모델 학습
        
        Args:
            X (numpy.ndarray): 학습 데이터 (n_samples, n_features)
            y (numpy.ndarray): 레이블 (n_samples,)
            verbose (bool): 학습 과정 출력 여부
        """
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred_proba = self.predict_proba(X)
            
            # 손실 계산
            loss = self.compute_loss(y, y_pred_proba)
            self.loss_history.append(loss)

            # 정확도 계산
            y_pred = (y_pred_proba >= 0.5).astype(int)
            accuracy = self.compute_accuracy(y, y_pred)
            self.accuracy_history.append(accuracy)
            
            # Gradient 계산 (MSE 기준): (y_hat - y) * sigma'(z)
            error = (y_pred_proba - y) * y_pred_proba * (1 - y_pred_proba)
            dw = np.dot(X.T, error) / n_samples
            db = np.mean(error)
            
            # 가중치 업데이트
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 진행 상황 출력
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    def evaluate(self, X, y):
        """
        모델 평가
        
        Args:
            X (numpy.ndarray): 테스트 데이터 (n_samples, n_features)
            y (numpy.ndarray): 레이블 (n_samples,)
            
        Returns:
            dict: 평가 메트릭 (loss, accuracy)
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        loss = self.compute_loss(y, y_pred_proba)
        accuracy = self.compute_accuracy(y, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
