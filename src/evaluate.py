"""
모델 평가 스크립트
학습된 모델로 이미지를 판별합니다.
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


def load_model(model_path, input_size):
    """
    저장된 모델 가중치 로드
    
    Args:
        model_path (str): 모델 가중치 경로
        input_size (int): 입력 특징 크기
        
    Returns:
        SingleLayerPerceptron: 로드된 모델
    """
    model = SingleLayerPerceptron(input_size=input_size)
    model.load_weights(model_path)
    return model


def predict_image(image_path, model, preprocessor):
    """
    단일 이미지 예측
    
    Args:
        image_path (str): 이미지 경로
        model (SingleLayerPerceptron): 학습된 모델
        preprocessor (ImagePreprocessor): 전처리기
        
    Returns:
        tuple: (예측 클래스, 예측 확률)
    """
    # 이미지 전처리
    X = preprocessor.preprocess(image_path).reshape(1, -1)
    
    # 예측
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    
    return pred, proba


def predict_batch(image_paths, model, preprocessor):
    """
    여러 이미지 배치 예측
    
    Args:
        image_paths (list): 이미지 경로 리스트
        model (SingleLayerPerceptron): 학습된 모델
        preprocessor (ImagePreprocessor): 전처리기
        
    Returns:
        tuple: (예측 클래스 배열, 예측 확률 배열)
    """
    # 이미지 전처리
    X = preprocessor.preprocess_batch(image_paths)
    
    # 예측
    probas = model.predict_proba(X)
    preds = model.predict(X)
    
    return preds, probas


def main(args):
    """메인 평가 함수"""
    
    print("=" * 60)
    print("네컷사진 판별 모델 평가")
    print("=" * 60)
    
    # 전처리기 초기화
    print(f"\n전처리기 초기화 (이미지 크기: {args.image_size}x{args.image_size})")
    preprocessor = ImagePreprocessor(
        target_size=(args.image_size, args.image_size),
        grayscale=True,
        normalize=True
    )
    
    # 입력 크기 계산
    input_size = args.image_size * args.image_size
    
    # 모델 로드
    print(f"\n모델 로드 중: {args.model_path}")
    try:
        model = load_model(args.model_path, input_size)
    except FileNotFoundError:
        print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model_path}")
        print("먼저 train.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    # 단일 이미지 예측
    if args.image_path:
        print(f"\n이미지 예측: {args.image_path}")
        try:
            pred, proba = predict_image(args.image_path, model, preprocessor)
            
            print("-" * 60)
            print(f"예측 결과: {'네컷사진' if pred == 1 else '일반 사진'}")
            print(f"네컷사진 확률: {proba:.4f}")
            print(f"일반 사진 확률: {1 - proba:.4f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"오류: {e}")
    
    # 디렉토리 내 모든 이미지 예측
    elif args.image_dir:
        print(f"\n디렉토리 예측: {args.image_dir}")
        
        # 이미지 파일 수집
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_paths:
            print(f"오류: {args.image_dir}에서 이미지 파일을 찾을 수 없습니다.")
            return
        
        print(f"총 {len(image_paths)}개의 이미지 발견")
        print("-" * 60)
        
        # 배치 예측
        try:
            preds, probas = predict_batch(image_paths, model, preprocessor)
            
            # 결과 출력
            for img_path, pred, proba in zip(image_paths, preds, probas):
                filename = os.path.basename(img_path)
                label = "네컷사진" if pred == 1 else "일반 사진"
                print(f"{filename:40s} -> {label:10s} (확률: {proba:.4f})")
            
            # 통계
            print("-" * 60)
            print(f"네컷사진: {np.sum(preds == 1)}개")
            print(f"일반 사진: {np.sum(preds == 0)}개")
            
        except Exception as e:
            print(f"오류: {e}")
    
    else:
        print("\n오류: --image-path 또는 --image-dir 중 하나를 지정해주세요.")
        return
    
    print("\n평가 완료!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="네컷사진 판별 모델 평가")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="data/processed/model_weights.npz",
        help="모델 가중치 경로 (기본값: data/processed/model_weights.npz)"
    )
    
    parser.add_argument(
        "--image-path",
        type=str,
        help="예측할 단일 이미지 경로"
    )
    
    parser.add_argument(
        "--image-dir",
        type=str,
        help="예측할 이미지 디렉토리 경로"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="이미지 리사이즈 크기 (기본값: 64)"
    )
    
    args = parser.parse_args()
    main(args)
