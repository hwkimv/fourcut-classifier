"""유틸리티 함수 모음
- 모델 가중치/편향 저장 및 로드 (numpy)
- 학습 로그 기록
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def ensure_dir(path: str | os.PathLike) -> None:
    """경로의 상위 디렉토리를 생성합니다."""
    p = Path(path)
    if p.suffix:  # 파일 경로로 간주
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)


def save_weights_numpy(weights: np.ndarray, bias: float | np.ndarray, model_dir: str = "model") -> Tuple[str, str]:
    """가중치(W)와 편향(b)을 각각 npy 파일로 저장합니다.

    Returns: (model_path, bias_path)
    """
    model_dir_path = Path(model_dir)
    model_path = model_dir_path / "model.npy"
    bias_path = model_dir_path / "bias.npy"
    ensure_dir(model_path)
    np.save(model_path, weights)
    np.save(bias_path, np.array(bias))
    return str(model_path), str(bias_path)


def load_weights_numpy(model_dir: str = "model") -> Tuple[np.ndarray, np.ndarray]:
    """npy 파일에서 가중치(W)와 편향(b)을 로드합니다."""
    model_dir_path = Path(model_dir)
    model_path = model_dir_path / "model.npy"
    bias_path = model_dir_path / "bias.npy"
    if not model_path.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {model_path}")
    if not bias_path.exists():
        raise FileNotFoundError(f"편향 파일을 찾을 수 없습니다: {bias_path}")
    W = np.load(model_path)
    b = np.load(bias_path)
    return W, b


def append_log(log_path: str, text: str) -> None:
    """로그 파일에 문자열을 한 줄 추가합니다."""
    ensure_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def write_epoch_log(log_path: str, epoch: int, loss: float) -> None:
    """에폭별 손실을 train_log.txt에 기록합니다."""
    append_log(log_path, f"{epoch}\t{loss:.6f}")

