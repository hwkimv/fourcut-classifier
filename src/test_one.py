# 단일 이미지 테스트 프로그램
# 사진 한 장을 넣으면 네컷인지 일반 사진인지 알려줌

import argparse
import os
import sys
import random
from pathlib import Path
import numpy as np
from PIL import Image

# 프로젝트 폴더를 Python이 찾을 수 있게 추가
sys.path.append(str(Path(__file__).parent.parent))
project_root = Path(__file__).parent.parent
os.chdir(project_root)

from src.preprocessing import ImagePreprocessor
from src.utils import load_weights_numpy


def show_image_info(image_path):
    """
    이미지 정보를 화면에 보여주고 이미지를 열어줌
    """
    try:
        img = Image.open(image_path)
        print(f"\n{'='*60}")
        print(f"테스트 이미지 정보:")
        print(f"{'='*60}")
        print(f"경로: {image_path}")
        print(f"파일명: {os.path.basename(image_path)}")
        print(f"크기: {img.size} (가로 x 세로)")
        print(f"모드: {img.mode}")
        print(f"파일 크기: {os.path.getsize(image_path) / 1024:.2f} KB")
        print(f"{'='*60}\n")

        # 이미지 창으로 보여주기
        img.show()
        print("이미지를 표시했습니다. 확인 후 창을 닫아주세요.\n")

    except Exception as e:
        print(f"이미지 정보 표시 중 오류: {e}\n")


def get_random_image_from_folder(folder_path):
    """
    폴더에서 랜덤으로 이미지 하나를 골라줌
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")

    # 이미지 파일만 필터링
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]

    if not images:
        raise FileNotFoundError(f"폴더에 이미지가 없습니다: {folder_path}")

    # 랜덤으로 하나 선택
    selected = random.choice(images)
    return os.path.join(folder_path, selected)


def get_random_image_from_test_folder(test_root="data/test"):
    """
    test 폴더 안의 fourcut, normal 등 모든 폴더에서 랜덤으로 이미지 선택
    """
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"테스트 폴더를 찾을 수 없습니다: {test_root}")

    # 모든 서브폴더에서 이미지 수집
    all_images = []
    subfolders = []

    for subfolder in ["fourcut", "normal", "non_fourcut"]:
        subfolder_path = os.path.join(test_root, subfolder)
        if os.path.exists(subfolder_path):
            images = [os.path.join(subfolder_path, f)
                     for f in os.listdir(subfolder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
            if images:
                all_images.extend(images)
                subfolders.append(subfolder)

    if not all_images:
        raise FileNotFoundError(f"테스트 폴더에 이미지가 없습니다: {test_root}")

    # 랜덤으로 하나 선택
    selected = random.choice(all_images)
    selected_folder = os.path.basename(os.path.dirname(selected))

    return selected, selected_folder, len(all_images)


def main():
    # 명령줄 옵션 설정
    parser = argparse.ArgumentParser(description="단일 이미지 네컷/일반 판별")
    parser.add_argument("image_path", type=str, nargs='?',
                        default="data/test",
                        help="이미지 경로 (기본: data/test 폴더에서 랜덤)")
    parser.add_argument("--random", "-r", action="store_true", default=True,
                        help="랜덤으로 이미지 선택 (기본: True)")
    parser.add_argument("--no-random", action="store_true",
                        help="랜덤 선택 끄기")
    parser.add_argument("--show-image", "-s", action="store_true", default=True,
                        help="테스트 전 이미지 미리보기 (기본: True)")
    parser.add_argument("--no-show", action="store_true",
                        help="이미지 미리보기 끄기")
    parser.add_argument("--model-dir", type=str, default="model",
                        help="모델 폴더 경로 (기본: model)")
    parser.add_argument("--image-size", type=int, default=128,
                        help="이미지 크기 (기본: 128x128)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="분류 기준값 (기본: 0.45)")
    args = parser.parse_args()

    # 옵션 처리
    if args.no_show:
        args.show_image = False
    if args.no_random:
        args.random = False

    # 랜덤 모드일 때 이미지 선택
    if args.random:
        if args.image_path == "data/test":
            # 기본값: test 폴더에서 랜덤 선택
            try:
                args.image_path, selected_folder, total_count = get_random_image_from_test_folder(args.image_path)
                print(f"\n📁 테스트 폴더: data/test (총 {total_count}개 이미지)")
                print(f"📂 선택된 서브폴더: {selected_folder}")
                print(f"🎲 랜덤 선택된 이미지: {os.path.basename(args.image_path)}\n")
            except FileNotFoundError as e:
                print(f"오류: {e}")
                return
        elif os.path.isfile(args.image_path):
            # 파일이 직접 지정된 경우
            pass
        elif os.path.isdir(args.image_path) and os.path.basename(args.image_path) == "test":
            # test 폴더가 지정된 경우
            try:
                args.image_path, selected_folder, total_count = get_random_image_from_test_folder(args.image_path)
                print(f"\n📁 테스트 폴더: {args.image_path} (총 {total_count}개 이미지)")
                print(f"📂 선택된 서브폴더: {selected_folder}")
                print(f"🎲 랜덤 선택된 이미지: {os.path.basename(args.image_path)}\n")
            except FileNotFoundError as e:
                print(f"오류: {e}")
                return
        elif os.path.isdir(args.image_path):
            # 특정 폴더가 지정된 경우
            folder_path = args.image_path
            try:
                args.image_path = get_random_image_from_folder(folder_path)
                print(f"\n📁 폴더: {folder_path}")
                print(f"🎲 랜덤 선택된 이미지: {os.path.basename(args.image_path)}\n")
            except FileNotFoundError as e:
                print(f"오류: {e}")
                return
        else:
            # 경로가 존재하지 않으면 기본 폴더에서 선택
            try:
                args.image_path, selected_folder, total_count = get_random_image_from_test_folder("data/test")
                print(f"\n📁 테스트 폴더: data/test (총 {total_count}개 이미지)")
                print(f"📂 선택된 서브폴더: {selected_folder}")
                print(f"🎲 랜덤 선택된 이미지: {os.path.basename(args.image_path)}\n")
            except FileNotFoundError as e:
                print(f"오류: {e}")
                return

    # 파일 존재 여부 확인
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {args.image_path}")

    # 이미지 미리보기
    if args.show_image:
        show_image_info(args.image_path)

    # 이미지 전처리 (AI가 이해할 수 있는 형태로 변환)
    pre = ImagePreprocessor(target_size=(args.image_size, args.image_size),
                           grayscale=True,
                           normalize=True)
    x = pre.preprocess(args.image_path)

    # 저장된 모델 불러오기
    W, b = load_weights_numpy(args.model_dir)

    # 예측 수행
    x = x.astype(np.float64)
    z = np.sum(x * W) + float(b)  # 가중치 곱하기
    proba = float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))  # 0~1 확률로 변환
    pred = int(proba >= args.threshold)  # 기준값 넘으면 네컷(1), 아니면 일반(0)

    # 결과 출력
    print(f"{'='*60}")
    print(f"🤖 예측 결과")
    print(f"{'='*60}")
    if args.threshold != 0.5:
        print(f"⚙️  분류 임계값: {args.threshold}")
    print(f"이미지: {args.image_path}")
    print(f"예측 클래스: {'🖼️  네컷' if pred == 1 else '📷 일반'}")
    print(f"네컷일 확률: {proba:.4f}")
    print(f"일반일 확률: {1 - proba:.4f}")
    print(f"신뢰도: {abs(proba - 0.5) * 200:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

