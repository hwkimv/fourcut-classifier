# 이미지 전처리 모듈
# 사진을 AI가 이해할 수 있는 숫자 데이터로 변환

import numpy as np
from PIL import Image
import shutil
from pathlib import Path
import random


class ImagePreprocessor:
    """
    이미지 전처리기
    - 사진을 불러와서 AI가 사용할 수 있는 형태로 바꿔줌
    """

    def __init__(self, target_size=(64, 64), grayscale=True, normalize=True):
        self.target_size = target_size  # 이미지 크기 (가로, 세로)
        self.grayscale = grayscale      # 흑백으로 변환할지 여부
        self.normalize = normalize      # 0~1 사이로 정규화할지 여부

    def load_image(self, image_path):
        """
        이미지 파일 불러오기
        """
        return Image.open(image_path)

    def resize(self, image):
        """
        이미지 크기 조정 (모든 사진을 같은 크기로 만듦)
        """
        # Pillow 버전에 따라 다른 방식 사용
        if hasattr(Image, 'Resampling'):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS
        return image.resize(self.target_size, resample)

    def to_grayscale(self, image):
        """
        컬러 이미지를 흑백으로 변환
        - 흑백이 더 간단하고 빠름
        """
        return image.convert('L')

    def to_array(self, image):
        """
        이미지를 숫자 배열로 변환
        - 예: 64x64 이미지 → 4096개의 숫자
        """
        return np.array(image)

    def normalize_array(self, array):
        """
        숫자 범위를 0~1로 변환
        - 원래: 0~255 (픽셀값)
        - 변환 후: 0.0~1.0 (AI가 학습하기 좋음)
        """
        return array.astype(np.float32) / 255.0

    def vectorize(self, array):
        """
        2D 배열을 1D로 펴기
        - 예: [[1,2], [3,4]] → [1,2,3,4]
        """
        return array.flatten()

    def preprocess(self, image_path):
        """
        전체 전처리 과정 실행
        1. 이미지 불러오기
        2. 크기 조정
        3. 흑백 변환
        4. 숫자 배열로 변환
        5. 정규화
        6. 1D 벡터로 변환
        """
        # 1. 이미지 불러오기
        image = self.load_image(image_path)

        # 2. 크기 조정
        image = self.resize(image)

        # 3. 흑백 변환
        if self.grayscale:
            image = self.to_grayscale(image)

        # 4. 배열로 변환
        array = self.to_array(image)

        # 5. 정규화
        if self.normalize:
            array = self.normalize_array(array)

        # 6. 1D 벡터로 변환
        vector = self.vectorize(array)

        return vector

    def preprocess_batch(self, image_paths):
        """
        여러 이미지를 한번에 전처리
        """
        vectors = []
        for path in image_paths:
            vector = self.preprocess(path)
            vectors.append(vector)
        return np.array(vectors)


def split_dataset(
    source_fourcut="data/train/fourcut",
    source_normal="data/train/normal",
    output_dir="data/split",
    train_ratio=0.65,
    val_ratio=0.15,
    test_ratio=0.20,
    random_seed=42
):
    """
    데이터를 학습/검증/테스트 폴더로 분할

    사용 예:
        from src.preprocessing import split_dataset
        split_dataset()
    """

    random.seed(random_seed)

    # 비율 검증
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"비율의 합이 1.0이 아닙니다: {total}")

    print("=" * 60)
    print("데이터셋 분할 시작")
    print("=" * 60)
    print(f"학습 데이터: {train_ratio*100:.1f}%")
    print(f"검증 데이터: {val_ratio*100:.1f}%")
    print(f"테스트 데이터: {test_ratio*100:.1f}%")
    print()

    # 출력 폴더 구조 생성
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for category in ['fourcut', 'normal']:
            folder = output_path / split / category
            folder.mkdir(parents=True, exist_ok=True)

    # 각 카테고리별로 처리
    categories = [
        ('fourcut', source_fourcut),
        ('normal', source_normal)
    ]

    for category_name, source_dir in categories:
        source_path = Path(source_dir)

        if not source_path.exists():
            print(f"⚠️  경고: {source_dir} 폴더가 없습니다. 건너뜁니다.")
            continue

        # 이미지 파일 목록 가져오기
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_files = [
            f for f in source_path.iterdir()
            if f.is_file() and f.suffix in image_extensions
        ]

        if not image_files:
            print(f"⚠️  경고: {source_dir}에 이미지가 없습니다.")
            continue

        # 랜덤 섞기
        random.shuffle(image_files)

        total_count = len(image_files)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)

        # 데이터 분할
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        print(f"\n📁 {category_name} 폴더 처리 중...")
        print(f"   전체: {total_count}개")
        print(f"   학습: {len(train_files)}개")
        print(f"   검증: {len(val_files)}개")
        print(f"   테스트: {len(test_files)}개")

        # 파일 복사
        splits = [
            ('train', train_files),
            ('val', val_files),
            ('test', test_files)
        ]

        for split_name, files in splits:
            dest_folder = output_path / split_name / category_name

            for file in files:
                dest_file = dest_folder / file.name
                shutil.copy2(file, dest_file)

        print(f"   ✓ 복사 완료")

    print("\n" + "=" * 60)
    print("데이터셋 분할 완료!")
    print("=" * 60)
    print(f"\n결과 폴더: {output_dir}")
    print("\n폴더 구조:")
    print(f"{output_dir}/")
    print("├── train/ (학습 데이터)")
    print("│   ├── fourcut/")
    print("│   └── normal/")
    print("├── val/ (검증 데이터)")
    print("│   ├── fourcut/")
    print("│   └── normal/")
    print("└── test/ (테스트 데이터)")
    print("    ├── fourcut/")
    print("    └── normal/")
    print()


if __name__ == "__main__":
    # 직접 실행 시 데이터 분할 수행
    split_dataset(
        source_fourcut="data/train/fourcut",
        source_normal="data/train/normal",
        output_dir="data/split",
        train_ratio=0.65,
        val_ratio=0.15,
        test_ratio=0.20,
        random_seed=42
    )

