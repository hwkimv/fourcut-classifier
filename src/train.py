# 모델 학습 프로그램
# 네컷 사진과 일반 사진을 구분하는 AI를 학습시킴

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# 프로젝트 폴더를 Python이 찾을 수 있게 추가
sys.path.append(str(Path(__file__).parent.parent))
project_root = Path(__file__).parent.parent
os.chdir(project_root)

from src.preprocessing import ImagePreprocessor
from src.model import SingleLayerPerceptron
from src.utils import save_weights_numpy, write_epoch_log, ensure_dir


def load_dataset(data_dir, preprocessor):
    """
    데이터 불러오기
    - fourcut 폴더: 네컷 사진들
    - normal 폴더: 일반 사진들
    """
    fourcut_dir = os.path.join(data_dir, 'fourcut')
    normal_dir = os.path.join(data_dir, 'normal')

    X = []  # 이미지 데이터
    y = []  # 정답 레이블 (1=네컷, 0=일반)

    # 네컷사진 불러오기
    if os.path.exists(fourcut_dir):
        all_files = os.listdir(fourcut_dir)
        fourcut_images = [os.path.join(fourcut_dir, f)
                          for f in all_files
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        print(f"네컷 이미지 파일 {len(fourcut_images)}개 발견")

        for idx, img_path in enumerate(fourcut_images, 1):
            try:
                vector = preprocessor.preprocess(img_path)  # 이미지를 숫자로 변환
                X.append(vector)
                y.append(1)  # 네컷 = 1
                if idx % 500 == 0:
                    print(f"  네컷 이미지 로드 중... {idx}/{len(fourcut_images)}")
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
        print(f"네컷 이미지 로드 완료: {len([_y for _y in y if _y == 1])}개")

    # 일반 사진 불러오기
    neg_dirs = [d for d in [normal_dir] if os.path.exists(d)]
    for neg_dir in neg_dirs:
        all_files = os.listdir(neg_dir)
        neg_images = [os.path.join(neg_dir, f)
                      for f in all_files
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        print(f"일반 이미지 파일 {len(neg_images)}개 발견 (from {neg_dir})")

        for idx, img_path in enumerate(neg_images, 1):
            try:
                vector = preprocessor.preprocess(img_path)  # 이미지를 숫자로 변환
                X.append(vector)
                y.append(0)  # 일반 = 0
                if idx % 100 == 0:
                    print(f"  일반 이미지 로드 중... {idx}/{len(neg_images)}")
            except Exception as e:
                print(f"이미지 로드 실패: {img_path} - {e}")
        print(f"일반 이미지 로드 완료: 총 {len([_y for _y in y if _y == 0])}개")

    # 데이터가 없으면 오류
    if len(X) == 0:
        raise ValueError(
            f"[오류: 데이터를 찾을 수 없습니다. {fourcut_dir} 및 {normal_dir} 폴더에 이미지를 추가해주세요.]")

    return np.array(X), np.array(y)


def split_dataset(X, y, test_ratio=0.2, val_ratio=0.15, random_state=42):
    """
    데이터를 3개로 나누기
    - 학습용 (Train): 65%
    - 검증용 (Validation): 15%
    - 테스트용 (Test): 20%
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)  # 랜덤 섞기

    # 데이터 개수 계산
    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)
    train_size = n_samples - test_size - val_size

    # 인덱스로 분할
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_dataset(X, y, method='undersample', random_state=42):
    """
    데이터 균형 맞추기
    - 네컷과 일반 사진 개수를 비슷하게 만듦
    - undersample: 많은 쪽을 줄임
    - oversample: 적은 쪽을 늘림 (복사)
    """
    np.random.seed(random_state)

    # 각 클래스 개수 확인
    idx_0 = np.where(y == 0)[0]  # 일반 사진 인덱스
    idx_1 = np.where(y == 1)[0]  # 네컷 사진 인덱스
    n_0 = len(idx_0)
    n_1 = len(idx_1)

    print(f"\n📊 데이터 밸런싱 ({method}):")
    print(f"  원본 - 일반: {n_0}개, 네컷: {n_1}개 (비율 {n_1/n_0:.2f}:1)")

    if method == 'undersample':
        # 많은 쪽을 적은 쪽만큼 줄이기
        min_samples = min(n_0, n_1)
        if n_0 > n_1:
            idx_0_sampled = np.random.choice(idx_0, min_samples, replace=False)
            idx_balanced = np.concatenate([idx_0_sampled, idx_1])
        else:
            idx_1_sampled = np.random.choice(idx_1, min_samples, replace=False)
            idx_balanced = np.concatenate([idx_0, idx_1_sampled])

    elif method == 'oversample':
        # 적은 쪽을 많은 쪽만큼 늘리기 (복사해서)
        max_samples = max(n_0, n_1)
        if n_0 < n_1:
            idx_0_sampled = np.random.choice(idx_0, max_samples, replace=True)
            idx_balanced = np.concatenate([idx_0_sampled, idx_1])
        else:
            idx_1_sampled = np.random.choice(idx_1, max_samples, replace=True)
            idx_balanced = np.concatenate([idx_0, idx_1_sampled])
    else:
        raise ValueError(f"Unknown method: {method}")

    # 섞기
    np.random.shuffle(idx_balanced)

    X_balanced = X[idx_balanced]
    y_balanced = y[idx_balanced]

    n_0_new = np.sum(y_balanced == 0)
    n_1_new = np.sum(y_balanced == 1)
    print(f"  밸런싱 후 - 일반: {n_0_new}개, 네컷: {n_1_new}개 (비율 {n_1_new/n_0_new:.2f}:1)")
    print(f"  총 샘플 수: {len(y_balanced)}개")

    return X_balanced, y_balanced


def main(args):
    """메인 학습 함수"""

    print("=" * 60)
    print("네컷사진 판별 모델 학습")
    print("=" * 60)

    # 데이터 경로
    data_dir = args.data_dir

    # 이미지 전처리기 준비
    print(f"\n전처리기 초기화 (이미지 크기: {args.image_size}x{args.image_size})")
    preprocessor = ImagePreprocessor(
        target_size=(args.image_size, args.image_size),
        grayscale=True,
        normalize=True
    )

    # 데이터 불러오기
    print("\n데이터 로드 중...")
    try:
        X, y = load_dataset(data_dir, preprocessor)
        print(f"총 샘플 수: {len(X)}")
        print(f"네컷사진: {np.sum(y == 1)}개")
        print(f"일반 사진: {np.sum(y == 0)}개")
        print(f"특징 차원: {X.shape[1]}")

        # 데이터 불균형 확인
        n_fourcut = np.sum(y == 1)
        n_normal = np.sum(y == 0)
        if n_normal > 0:
            ratio = n_fourcut / n_normal
            if ratio > 3 or ratio < 0.33:
                print(f"\n⚠️  데이터 불균형 감지! 비율: {ratio:.2f}:1")
                if not args.no_balance:
                    print(f"   → 자동 밸런싱 활성화 (방법: {args.balance_method})")

    except ValueError as e:
        print(f"\n오류: {e}")
        return

    # 데이터 균형 맞추기 (옵션)
    if not args.no_balance:
        X, y = balance_dataset(X, y, method=args.balance_method, random_state=args.random_state)

    # 데이터 나누기 (학습/검증/테스트)
    print("\n데이터 분할 중...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state
    )
    print(f"학습 세트: {len(X_train)}개 (일반: {np.sum(y_train==0)}, 네컷: {np.sum(y_train==1)})")
    print(f"검증 세트: {len(X_val)}개 (일반: {np.sum(y_val==0)}, 네컷: {np.sum(y_val==1)})")
    print(f"테스트 세트: {len(X_test)}개 (일반: {np.sum(y_test==0)}, 네컷: {np.sum(y_test==1)})")

    # AI 모델 준비
    print(f"\n모델 초기화 (학습률: {args.learning_rate}, 에포크: {args.epochs})")
    model = SingleLayerPerceptron(
        input_size=X.shape[1],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        random_state=args.random_state
    )

    # 학습 시작
    print("\n📚 모델 학습 시작...")
    if args.early_stopping:
        print(f"⚡ 조기 종료(Early Stopping) 활성화 (인내: {args.patience}회)")
    if args.l2_reg > 0:
        print(f"🔧 L2 정규화 활성화 (람다: {args.l2_reg})")
    print("-" * 60)
    ensure_dir(args.log_path)

    # 조기 종료를 위한 변수
    best_val_loss = float('inf')  # 가장 좋은 검증 손실
    best_val_accuracy = 0.0  # 가장 좋은 검증 정확도
    patience_counter = 0  # 개선 안된 횟수
    best_weights = None  # 가장 좋은 가중치
    best_bias = None  # 가장 좋은 편향
    best_epoch = 0  # 가장 좋았던 에포크

    # 학습 기록
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # 에포크 반복 (학습 반복)
    for epoch in range(model.epochs):
        # 1단계: 학습 데이터로 예측
        y_pred_proba = model.predict_proba(X_train)

        # 2단계: 손실 계산 (얼마나 틀렸는지)
        train_loss = model.compute_loss(y_train, y_pred_proba)
        if args.l2_reg > 0:
            # L2 정규화 추가 (과적합 방지)
            l2_penalty = args.l2_reg * np.sum(model.weights ** 2)
            train_loss += l2_penalty

        # 3단계: 학습 정확도 계산
        y_pred_train = (y_pred_proba >= 0.5).astype(int)
        train_accuracy = model.compute_accuracy(y_train, y_pred_train)

        # 4단계: 기울기 계산 (가중치를 어떻게 바꿀지)
        error = (y_pred_proba - y_train) * y_pred_proba * (1 - y_pred_proba)
        dw = np.dot(X_train.T, error) / len(X_train)
        if args.l2_reg > 0:
            dw += 2 * args.l2_reg * model.weights
        db = np.mean(error)

        # 5단계: 가중치 업데이트 (조금씩 수정)
        model.weights -= model.learning_rate * dw
        model.bias -= model.learning_rate * db

        # 6단계: 검증 데이터로 성능 확인
        val_pred_proba = model.predict_proba(X_val)
        val_loss = model.compute_loss(y_val, val_pred_proba)
        y_pred_val = (val_pred_proba >= 0.5).astype(int)
        val_accuracy = model.compute_accuracy(y_val, y_pred_val)

        # 기록 저장
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

        # 조기 종료 확인
        if args.early_stopping:
            # 검증 손실이 개선되면
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                patience_counter = 0
                best_weights = model.weights.copy()
                best_bias = model.bias

                if (epoch + 1) % 10 == 0 or epoch < 5:
                    print(f"✅ 에포크 {epoch + 1}: 검증 손실 개선! {val_loss:.6f} (정확도: {val_accuracy:.4f})")
            else:
                patience_counter += 1

            # 너무 오래 개선 안되면 중단
            if patience_counter >= args.patience:
                print(f"\n⏹️  조기 종료: 에포크 {epoch + 1}에서 학습 중단")
                print(f"   💡 검증 손실이 {args.patience} 에포크 동안 개선되지 않았습니다.")
                print(f"   🏆 최적 에포크: {best_epoch}")
                print(f"   📉 최적 검증 손실: {best_val_loss:.6f}")
                print(f"   🎯 최적 검증 정확도: {best_val_accuracy:.4f}")
                # 최적 가중치로 되돌림
                model.weights = best_weights
                model.bias = best_bias
                break

        # 진행 상황 출력 (10번마다)
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == model.epochs:
            print(f"에포크 {epoch + 1}/{model.epochs} - "
                  f"손실: {train_loss:.6f}, 정확도: {train_accuracy:.4f} | "
                  f"검증 손실: {val_loss:.6f}, 검증 정확도: {val_accuracy:.4f}")

        # 로그 파일에 기록
        write_epoch_log(args.log_path, epoch=epoch + 1, loss=train_loss)

    # 조기 종료 사용 시 최종 정보
    if args.early_stopping and best_weights is not None:
        print(f"\n🏆 최적 모델 사용 (에포크 {best_epoch})")
        print(f"   📉 검증 손실: {best_val_loss:.6f}")
        print(f"   🎯 검증 정확도: {best_val_accuracy:.4f}")

    # 최종 평가
    print("\n" + "=" * 60)
    print("모델 평가")
    print("=" * 60)

    # 학습 세트 평가
    train_metrics = model.evaluate(X_train, y_train)
    print(f"\n[학습 세트]")
    print(f"손실: {train_metrics['loss']:.4f}")
    print(f"정확도: {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")

    # 검증 세트 평가
    val_metrics = model.evaluate(X_val, y_val)
    print(f"\n[검증 세트]")
    print(f"손실: {val_metrics['loss']:.4f}")
    print(f"정확도: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")

    # 테스트 세트 평가 (진짜 성능)
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\n[테스트 세트]")
    print(f"손실: {test_metrics['loss']:.4f}")
    print(f"정확도: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")

    # 과적합/과소적합 확인
    overfitting_gap = train_metrics['accuracy'] - test_metrics['accuracy']
    if overfitting_gap > 0.1:
        print(f"\n⚠️  과적합 가능성 감지!")
        print(f"   학습-테스트 정확도 차이: {overfitting_gap*100:.2f}%")
        print(f"   💡 권장: L2 정규화 증가 또는 데이터 증강")
    elif overfitting_gap < -0.05:
        print(f"\n⚠️  과소적합 가능성 감지!")
        print(f"   학습-테스트 정확도 차이: {overfitting_gap*100:.2f}%")
        print(f"   💡 권장: 에포크 증가 또는 학습률 조정")
    else:
        print(f"\n✅ 모델이 잘 일반화되었습니다!")
        print(f"   학습-테스트 정확도 차이: {overfitting_gap*100:.2f}%")

    # 모델 저장
    print(f"\n💾 모델 저장 중: {args.model_dir}")
    save_weights_numpy(model.weights, model.bias, model_dir=args.model_dir)

    print("\n✅ 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    # 명령줄 옵션 설정
    parser = argparse.ArgumentParser(description="네컷사진 판별 모델 학습")

    parser.add_argument("--data-dir", type=str, default="data/train",
                       help="학습 데이터 폴더 (기본값: data/train)")

    parser.add_argument("--image-size", type=int, default=128,
                       help="이미지 크기 (기본값: 128)")

    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="학습 속도 (기본값: 0.001)")

    parser.add_argument("--epochs", type=int, default=2000,
                       help="학습 반복 횟수 (기본값: 2000)")

    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="테스트 데이터 비율 (기본값: 0.2)")

    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="검증 데이터 비율 (기본값: 0.15)")

    parser.add_argument("--random-state", type=int, default=42,
                       help="랜덤 시드 (기본값: 42)")

    parser.add_argument("--model-dir", type=str, default="model",
                       help="모델 저장 폴더 (기본값: model)")

    parser.add_argument("--log-path", type=str, default="logs/train_log.txt",
                       help="학습 로그 파일 (기본값: logs/train_log.txt)")

    parser.add_argument("--balance-method", type=str, default="undersample",
                       choices=["undersample", "oversample"],
                       help="데이터 균형 방법 (기본값: undersample)")

    parser.add_argument("--no-balance", action="store_true",
                       help="데이터 균형 맞추기 끄기")

    parser.add_argument("--early-stopping", action="store_true",
                       help="조기 종료 활성화")

    parser.add_argument("--patience", type=int, default=50,
                       help="조기 종료 인내 횟수 (기본값: 50)")

    parser.add_argument("--min-delta", type=float, default=0.0001,
                       help="조기 종료 최소 개선폭 (기본값: 0.0001)")

    parser.add_argument("--l2-reg", type=float, default=0.001,
                       help="L2 정규화 계수 (기본값: 0.001)")

    args = parser.parse_args()
    main(args)

