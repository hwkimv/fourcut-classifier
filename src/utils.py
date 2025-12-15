# 유틸리티 함수 모음
# 모델 저장/불러오기, 로그 기록 등 도우미 함수들

import os
from pathlib import Path
from typing import Tuple
import numpy as np


def ensure_dir(path):
    """
    폴더가 없으면 자동으로 만들어줌
    """
    p = Path(path)
    if p.suffix:  # 파일 경로면 상위 폴더만 생성
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)


def save_weights_numpy(weights, bias, model_dir="model"):
    """
    학습된 모델(가중치와 편향)을 파일로 저장
    - model.npy: 가중치 저장
    - bias.npy: 편향 저장
    """
    model_dir_path = Path(model_dir)
    model_path = model_dir_path / "model.npy"
    bias_path = model_dir_path / "bias.npy"

    # 폴더가 없으면 생성
    ensure_dir(model_path)

    # 파일로 저장
    np.save(model_path, weights)
    np.save(bias_path, np.array(bias))

    return str(model_path), str(bias_path)


def load_weights_numpy(model_dir="model"):
    """
    저장된 모델(가중치와 편향)을 불러오기
    """
    model_dir_path = Path(model_dir)
    model_path = model_dir_path / "model.npy"
    bias_path = model_dir_path / "bias.npy"

    # 파일이 없으면 오류 메시지
    if not model_path.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {model_path}")
    if not bias_path.exists():
        raise FileNotFoundError(f"편향 파일을 찾을 수 없습니다: {bias_path}")

    # 파일 불러오기
    W = np.load(model_path)
    b = np.load(bias_path)

    return W, b


def append_log(log_path, text):
    """
    로그 파일에 한 줄 추가
    """
    ensure_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def write_epoch_log(log_path, epoch, loss):
    """
    학습 과정을 로그 파일에 기록
    - 에포크 번호와 손실값을 저장
    """
    append_log(log_path, f"{epoch}\t{loss:.6f}")

