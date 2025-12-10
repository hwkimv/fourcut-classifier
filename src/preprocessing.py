"""
이미지 전처리 모듈
네컷사진 판별을 위한 이미지 전처리 기능을 제공합니다.
"""

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    이미지 전처리를 위한 클래스
    
    Args:
        target_size (tuple): 리사이즈할 목표 크기 (width, height)
        grayscale (bool): 그레이스케일 변환 여부
        normalize (bool): 정규화 여부 (0-1 범위로 변환)
    """
    
    def __init__(self, target_size=(64, 64), grayscale=True, normalize=True):
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
    
    def load_image(self, image_path):
        """
        이미지 파일을 로드합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            PIL.Image: 로드된 이미지
        """
        return Image.open(image_path)
    
    def resize(self, image):
        """
        이미지를 지정된 크기로 리사이즈합니다.
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            PIL.Image: 리사이즈된 이미지
        """
        # Pillow 10.0.0+ uses Image.Resampling.LANCZOS
        # Older versions use Image.LANCZOS
        if hasattr(Image, 'Resampling'):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS
        return image.resize(self.target_size, resample)
    
    def to_grayscale(self, image):
        """
        이미지를 그레이스케일로 변환합니다.
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            PIL.Image: 그레이스케일 이미지
        """
        return image.convert('L')
    
    def to_array(self, image):
        """
        PIL 이미지를 numpy 배열로 변환합니다.
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            numpy.ndarray: 이미지 배열
        """
        return np.array(image)
    
    def normalize_array(self, array):
        """
        배열을 0-1 범위로 정규화합니다.
        
        Args:
            array (numpy.ndarray): 입력 배열
            
        Returns:
            numpy.ndarray: 정규화된 배열
        """
        return array.astype(np.float32) / 255.0
    
    def vectorize(self, array):
        """
        2D 배열을 1D 벡터로 변환합니다.
        
        Args:
            array (numpy.ndarray): 입력 배열
            
        Returns:
            numpy.ndarray: 1D 벡터
        """
        return array.flatten()
    
    def preprocess(self, image_path):
        """
        이미지를 전처리하여 벡터로 변환합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            numpy.ndarray: 전처리된 이미지 벡터
        """
        # 이미지 로드
        image = self.load_image(image_path)
        
        # 리사이즈
        image = self.resize(image)
        
        # 그레이스케일 변환
        if self.grayscale:
            image = self.to_grayscale(image)
        
        # numpy 배열로 변환
        array = self.to_array(image)
        
        # 정규화
        if self.normalize:
            array = self.normalize_array(array)
        
        # 벡터화
        vector = self.vectorize(array)
        
        return vector
    
    def preprocess_batch(self, image_paths):
        """
        여러 이미지를 배치로 전처리합니다.
        
        Args:
            image_paths (list): 이미지 파일 경로 리스트
            
        Returns:
            numpy.ndarray: 전처리된 이미지 벡터들의 배열 (n_samples, n_features)
        """
        vectors = []
        for path in image_paths:
            vector = self.preprocess(path)
            vectors.append(vector)
        
        return np.array(vectors)
