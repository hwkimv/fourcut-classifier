# 이미지 전처리 모듈
# 사진을 AI가 이해할 수 있는 숫자 데이터로 변환

import numpy as np
from PIL import Image


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

