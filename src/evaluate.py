"""
전체 테스트셋 정확도 평가 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import ImagePreprocessor
from src.model import SingleLayerPerceptron
from src.utils import load_weights_numpy


def collect_paths(root: str):
    fourcut = Path(root) / "fourcut"
    normal = Path(root) / "normal"
    # 하위 호환: non_fourcut 폴더도 허용
    non_fourcut = Path(root) / "non_fourcut"

    paths = []
    labels = []

    for p in sorted(fourcut.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(p))
            labels.append(1)

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
    parser = argparse.ArgumentParser(description="테스트셋 정확도 평가")
    parser.add_argument("--data-dir", type=str, default="data/test", help="테스트셋 루트 경로")
    parser.add_argument("--model-dir", type=str, default="model", help="모델 디렉토리 (model.npy, bias.npy)")
    parser.add_argument("--image-size", type=int, default=64, help="전처리 이미지 크기")
    args = parser.parse_args()

    pre = ImagePreprocessor(target_size=(args.image_size, args.image_size), grayscale=True, normalize=True)

    try:
        paths, labels = collect_paths(args.data_dir)
    except ValueError as e:
        print(f"오류: {e}")
        return

    X = pre.preprocess_batch(paths)

    model = SingleLayerPerceptron(input_size=X.shape[1])
    W, b = load_weights_numpy(args.model_dir)
    model.weights = W
    model.bias = float(b)

    preds = model.predict(X)
    acc = float(np.mean(preds == labels))

    print("=" * 60)
    print("테스트셋 평가 결과")
    print(f"샘플 수: {len(labels)}")
    print(f"정확도: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
