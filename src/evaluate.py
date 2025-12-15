# 전체 테스트셋 평가 프로그램
# 여러 이미지를 한번에 테스트해서 정확도를 측정

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# 프로젝트 폴더를 Python이 찾을 수 있게 추가
sys.path.append(str(Path(__file__).parent.parent))
project_root = Path(__file__).parent.parent
os.chdir(project_root)

from src.preprocessing import ImagePreprocessor
from src.model import SingleLayerPerceptron
from src.utils import load_weights_numpy


def collect_paths(root):
    """
    테스트 폴더에서 모든 이미지 경로를 수집
    - fourcut 폴더: 네컷 사진 (라벨 1)
    - normal 폴더: 일반 사진 (라벨 0)
    """
    fourcut = Path(root) / "fourcut"
    normal = Path(root) / "normal"
    non_fourcut = Path(root) / "non_fourcut"  # 예전 이름도 지원

    paths = []  # 이미지 경로 리스트
    labels = []  # 정답 리스트

    # 네컷 사진 수집 (라벨 1)
    for p in sorted(fourcut.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p))
            labels.append(1)

    # 일반 사진 수집 (라벨 0)
    neg_dirs = [normal, non_fourcut]
    for nd in neg_dirs:
        if nd.exists():
            for p in sorted(nd.glob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(str(p))
                    labels.append(0)

    if not paths:
        raise ValueError(f"테스트 이미지가 없습니다: {root}")

    return paths, np.array(labels)


def main():
    # 명령줄 옵션 설정
    parser = argparse.ArgumentParser(description="테스트셋 정확도 평가")
    parser.add_argument("--data-dir", type=str, default="data/test",
                       help="테스트 폴더 경로")
    parser.add_argument("--model-dir", type=str, default="model",
                       help="모델 폴더 경로")
    parser.add_argument("--image-size", type=int, default=128,
                       help="이미지 크기")
    parser.add_argument("--threshold", type=float, default=0.45,
                       help="분류 기준값 (기본: 0.45)")
    parser.add_argument("--find-best-threshold", action="store_true",
                       help="최적 임계값 자동으로 찾기")
    args = parser.parse_args()

    # 이미지 전처리기 준비
    pre = ImagePreprocessor(target_size=(args.image_size, args.image_size),
                           grayscale=True,
                           normalize=True)

    # 테스트 이미지 불러오기
    try:
        paths, labels = collect_paths(args.data_dir)
    except ValueError as e:
        print(f"오류: {e}")
        return

    # 이미지를 숫자 데이터로 변환
    X = pre.preprocess_batch(paths)

    # 저장된 모델 불러오기
    model = SingleLayerPerceptron(input_size=X.shape[1])
    W, b = load_weights_numpy(args.model_dir)
    model.weights = W
    model.bias = float(b)

    # 최적 임계값 찾기 모드
    if args.find_best_threshold:
        print("\n🔍 최적 임계값 탐색 중...\n")
        print(f"{'임계값':<10} {'전체 정확도':<12} {'네컷 정확도':<12} {'일반 정확도':<12}")
        print("-" * 60)

        best_threshold = 0.5
        best_acc = 0.0
        best_balanced_acc = 0.0

        # 0.1부터 0.95까지 0.05씩 증가하며 테스트
        for thresh in np.arange(0.1, 1.0, 0.05):
            preds_test = model.predict(X, threshold=thresh)
            acc = float(np.mean(preds_test == labels))

            # 네컷과 일반 각각의 정확도 계산
            fourcut_mask = labels == 1
            normal_mask = labels == 0
            fourcut_acc = float(np.mean(preds_test[fourcut_mask] == labels[fourcut_mask])) if np.any(fourcut_mask) else 0
            normal_acc = float(np.mean(preds_test[normal_mask] == labels[normal_mask])) if np.any(normal_mask) else 0

            # 균형 정확도 (네컷 정확도 + 일반 정확도의 평균)
            balanced_acc = (fourcut_acc + normal_acc) / 2

            print(f"{thresh:<10.2f} {acc:<12.4f} {fourcut_acc:<12.4f} {normal_acc:<12.4f}")

            # 가장 좋은 임계값 저장
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                best_threshold = thresh
                best_acc = acc

        print("-" * 60)
        print(f"\n🏆 최적 임계값: {best_threshold:.2f}")
        print(f"   전체 정확도: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"   균형 정확도: {best_balanced_acc:.4f} ({best_balanced_acc*100:.2f}%)")
        print(f"\n💡 권장: 이 임계값으로 재평가하려면 --threshold {best_threshold:.2f} 옵션을 사용하세요.\n")
        return

    # 일반 평가 모드
    if args.threshold != 0.5:
        print(f"\n⚙️  사용자 지정 임계값: {args.threshold}")
        print(f"   💡 임계값이 낮을수록 네컷으로 분류하기 쉬워집니다.\n")

    # 예측 수행
    preds = model.predict(X, threshold=args.threshold)
    acc = float(np.mean(preds == labels))

    # 네컷과 일반 각각의 정확도 계산
    fourcut_mask = labels == 1
    normal_mask = labels == 0
    fourcut_acc = float(np.mean(preds[fourcut_mask] == labels[fourcut_mask])) if np.any(fourcut_mask) else 0
    normal_acc = float(np.mean(preds[normal_mask] == labels[normal_mask])) if np.any(normal_mask) else 0

    # 혼동 행렬 계산
    tp = np.sum((preds == 1) & (labels == 1))  # 네컷을 네컷으로 맞춤
    fp = np.sum((preds == 1) & (labels == 0))  # 일반을 네컷으로 틀림
    tn = np.sum((preds == 0) & (labels == 0))  # 일반을 일반으로 맞춤
    fn = np.sum((preds == 0) & (labels == 1))  # 네컷을 일반으로 틀림

    # 결과 출력
    print("=" * 60)
    print("테스트셋 평가 결과")
    print("=" * 60)
    print(f"분류 임계값: {args.threshold}")
    print(f"전체 샘플 수: {len(labels)}")
    print(f"  - 네컷 이미지: {np.sum(labels == 1)}개")
    print(f"  - 일반 이미지: {np.sum(labels == 0)}개")
    print()
    print(f"전체 정확도: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  - 네컷 정확도: {fourcut_acc:.4f} ({fourcut_acc*100:.2f}%)")
    print(f"  - 일반 정확도: {normal_acc:.4f} ({normal_acc*100:.2f}%)")
    print()
    print("혼동 행렬 (Confusion Matrix):")
    print(f"                예측: 일반    예측: 네컷")
    print(f"실제: 일반        {tn:>4}        {fp:>4}  ← 일반을 네컷으로 오분류: {fp}개")
    print(f"실제: 네컷        {fn:>4}        {tp:>4}  ← 네컷을 일반으로 오분류: {fn}개")
    print()

    # 오분류 분석
    if fp > 0:
        print(f"⚠️  문제: 일반 이미지 {np.sum(labels == 0)}개 중 {fp}개를 네컷으로 잘못 분류!")
        print(f"   → 일반 이미지 오분류 비율: {fp/np.sum(labels == 0)*100:.1f}%")

    if fn > 0:
        print(f"⚠️  문제: 네컷 이미지 {np.sum(labels == 1)}개 중 {fn}개를 일반으로 잘못 분류!")
        print(f"   → 네컷 이미지 오분류 비율: {fn/np.sum(labels == 1)*100:.1f}%")
        print(f"   💡 해결방안: 최적 임계값을 찾으려면 --find-best-threshold 옵션을 사용하세요.")

    print("=" * 60)


if __name__ == "__main__":
    main()

