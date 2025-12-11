"""
모델 학습 스크립트
네컷사진 판별 모델을 학습합니다.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# src 모듈을 임포트하기 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import ImagePreprocessor
from src.model import SingleLayerPerceptron
from src.utils import save_weights_numpy, write_epoch_log, ensure_dir


def load_dataset(data_dir, preprocessor):
    """
    데이터셋 로드 및 전처리

    Args:
        data_dir (str): 데이터 디렉토리 경로
        preprocessor (ImagePreprocessor): 전처리기 인스턴스

    Returns:
        tuple: (X, y) - 특징과 레이블
    """
    fourcut_dir = os.path.join(data_dir, 'fourcut')
    normal_dir = os.path.join(data_dir, 'normal')
    # 하위 호환: 기존 non_fourcut 폴더도 지원
    non_fourcut_dir = os.path.join(data_dir, 'non_fourcut')

    X = []
    y = []

    # 네컷사진 로드 (label=1)
    if os.path.exists(fourcut_dir):
        fourcut_images = [os.path.join(fourcut_dir, f)
                          for f in os.listdir(fourcut_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_path in fourcut_images:
            try:
                vector = preprocessor.preprocess(img_path)
                X.append(vector)
                y.append(1)
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")

    # 일반 사진 로드 (label=0)
    # 네컷사진 로드 (label=1)
    if os.path.exists(fourcut_dir):
        all_files = os.listdir(fourcut_dir)
        fourcut_images = [os.path.join(fourcut_dir, f)
                          for f in all_files
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        print(f"네컷 이미지 파일 {len(fourcut_images)}개 발견")
        for idx, img_path in enumerate(fourcut_images, 1):
            try:
                vector = preprocessor.preprocess(img_path)
                X.append(vector)
                y.append(1)
                if idx % 500 == 0:
                    print(f"  네컷 이미지 로드 중... {idx}/{len(fourcut_images)}")
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
        print(f"네컷 이미지 로드 완료: {len([_y for _y in y if _y == 1])}개")

    # 일반 사진 로드 (label=0)
    neg_dirs = [d for d in [normal_dir, non_fourcut_dir] if os.path.exists(d)]
    for neg_dir in neg_dirs:
        all_files = os.listdir(neg_dir)
        neg_images = [os.path.join(neg_dir, f)
                      for f in all_files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        print(f"일반 이미지 파일 {len(neg_images)}개 발견 (from {neg_dir})")
        for idx, img_path in enumerate(neg_images, 1):
            try:
                vector = preprocessor.preprocess(img_path)
                X.append(vector)
                y.append(0)
                if idx % 100 == 0:
                    print(f"  일반 이미지 로드 중... {idx}/{len(neg_images)}")
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
        print(f"일반 이미지 로드 완료: 총 {len([_y for _y in y if _y == 0])}개")

    if len(X) == 0:
        raise ValueError(
            f"데이터를 찾을 수 없습니다. {fourcut_dir} 및 {normal_dir} (또는 {non_fourcut_dir}) 폴더에 이미지를 추가해주세요.")

    return np.array(X), np.array(y)


def split_dataset(X, y, test_ratio=0.2, random_state=42):
    """
    데이터셋을 학습/테스트 세트로 분할

    Args:
        X (numpy.ndarray): 특징
        y (numpy.ndarray): 레이블
        test_ratio (float): 테스트 세트 비율
        random_state (int): 난수 시드

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    test_size = int(n_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def main(args):
    """메인 학습 함수"""

    print("=" * 60)
    print("네컷사진 판별 모델 학습")
    print("=" * 60)

    # 데이터 디렉토리 설정
    data_dir = args.data_dir

    # 전처리기 초기화
    print(f"\n전처리기 초기화 (이미지 크기: {args.image_size}x{args.image_size})")
    preprocessor = ImagePreprocessor(
        target_size=(args.image_size, args.image_size),
        grayscale=True,
        normalize=True
    )

    # 데이터 로드
    print("\n데이터 로드 중...")
    try:
        X, y = load_dataset(data_dir, preprocessor)
        print(f"총 샘플 수: {len(X)}")
        print(f"네컷사진: {np.sum(y == 1)}개")
        print(f"일반 사진: {np.sum(y == 0)}개")
        print(f"특징 차원: {X.shape[1]}")
    except ValueError as e:
        print(f"\n오류: {e}")
        return

    # 데이터 분할
    print("\n데이터 분할 중...")
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=args.test_ratio, random_state=args.random_state
    )
    print(f"학습 세트: {len(X_train)}개")
    print(f"테스트 세트: {len(X_test)}개")

    # 모델 초기화
    print(f"\n모델 초기화 (학습률: {args.learning_rate}, 에포크: {args.epochs})")
    model = SingleLayerPerceptron(
        input_size=X.shape[1],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        random_state=args.random_state
    )

    # 학습 및 로그
    print("\n모델 학습 시작...")
    print("-" * 60)
    ensure_dir(args.log_path)
    # 로그 헤더
    write_epoch_log(args.log_path, epoch=0, loss=float('nan'))  # 포맷 고정용 (옵션)

    for epoch in range(model.epochs):
        # 한 에폭 수동 학습: model.fit은 전체 에폭을 돌리므로, 여기선 직접 구현
        # forward
        y_pred_proba = model.predict_proba(X_train)
        # loss (MSE)
        loss = model.compute_loss(y_train, y_pred_proba)
        model.loss_history.append(loss)
        # grad
        error = (y_pred_proba - y_train) * y_pred_proba * (1 - y_pred_proba)
        dw = np.dot(X_train.T, error) / len(X_train)
        db = np.mean(error)
        # update
        model.weights -= model.learning_rate * dw
        model.bias -= model.learning_rate * db
        # 로그 출력 및 저장
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == model.epochs:
            print(f"Epoch {epoch + 1}/{model.epochs} - Loss: {loss:.6f}")
        write_epoch_log(args.log_path, epoch=epoch + 1, loss=loss)

    # 평가
    print("\n" + "=" * 60)
    print("모델 평가")
    print("=" * 60)

    train_metrics = model.evaluate(X_train, y_train)
    print(f"\n[학습 세트]")
    print(f"손실: {train_metrics['loss']:.4f}")
    print(f"정확도: {train_metrics['accuracy']:.4f}")

    test_metrics = model.evaluate(X_test, y_test)
    print(f"\n[테스트 세트]")
    print(f"손실: {test_metrics['loss']:.4f}")
    print(f"정확도: {test_metrics['accuracy']:.4f}")

    # 모델 저장 (W, b를 개별 npy로)
    print(f"\n모델 저장 중: {args.model_dir}")
    save_weights_numpy(model.weights, model.bias, model_dir=args.model_dir)

    print("\n학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="네컷사진 판별 모델 학습")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="데이터 디렉토리 경로 (기본값: data/train)"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="이미지 리사이즈 크기 (기본값: 64)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="학습률 (기본값: 0.01)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에포크 수 (기본값: 100)"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="테스트 세트 비율 (기본값: 0.2)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="난수 시드 (기본값: 42)"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="모델 저장 디렉토리 (model.npy, bias.npy 생성)"
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/train_log.txt",
        help="학습 로그 파일 경로 (기본값: logs/train_log.txt)"
    )

    args = parser.parse_args()
    main(args)
