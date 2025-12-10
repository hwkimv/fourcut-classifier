"""단일 이미지 판별 스크립트
저장된 SLP 모델(W, b)을 로드하여 단일 이미지가 네컷인지 여부를 출력합니다.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# src 임포트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import ImagePreprocessor
from src.model import SingleLayerPerceptron
from src.utils import load_weights_numpy


def main():
    parser = argparse.ArgumentParser(description="단일 이미지 네컷/일반 판별")
    parser.add_argument("image_path", type=str, help="입력 이미지 경로")
    parser.add_argument("--model-dir", type=str, default="model", help="모델 디렉토리 (model.npy, bias.npy)")
    parser.add_argument("--image-size", type=int, default=64, help="전처리 이미지 크기 (정사각) ")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {args.image_path}")

    # 전처리기
    pre = ImagePreprocessor(target_size=(args.image_size, args.image_size), grayscale=True, normalize=True)
    x = pre.preprocess(args.image_path).reshape(1, -1)

    # 모델 구성 및 가중치 로드
    input_size = x.shape[1]
    model = SingleLayerPerceptron(input_size=input_size)
    W, b = load_weights_numpy(args.model_dir)
    model.weights = W
    model.bias = float(b)

    proba = float(model.predict_proba(x)[0])
    pred = int(proba >= 0.5)

    print(f"이미지: {args.image_path}")
    print(f"예측: {'네컷' if pred == 1 else '일반'} (확률={proba:.4f})")


if __name__ == "__main__":
    main()
