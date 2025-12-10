#!/bin/bash
# 네컷 포토부스 합성 이미지 생성 - 빠른 시작 스크립트

echo "======================================"
echo "  네컷 포토부스 합성 이미지 생성기"
echo "======================================"
echo ""

# 1. 필수 폴더 생성
echo "[1/4] 필수 폴더 생성 중..."
mkdir -p data/raw/normal
mkdir -p data/train/fourcut
echo "✓ 폴더 생성 완료"
echo ""

# 2. 소스 이미지 확인
echo "[2/4] 소스 이미지 확인 중..."
image_count=$(ls data/raw/normal/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
if [ $image_count -eq 0 ]; then
    echo "⚠️  경고: data/raw/normal/ 폴더에 이미지가 없습니다!"
    echo "   jpg 또는 png 이미지를 추가해주세요."
    echo ""
else
    echo "✓ 사용 가능한 이미지: ${image_count}개"
    echo ""
fi

# 3. 프레임 이미지 확인
echo "[3/4] 프레임 이미지 확인 중..."
if [ ! -f "frame.png" ]; then
    echo "⚠️  경고: frame.png 파일이 없습니다!"
    echo "   투명 슬롯이 있는 네컷 프레임 PNG를 준비해주세요."
    echo ""
else
    echo "✓ frame.png 파일 발견"
    echo ""
fi

# 4. 스크립트 실행
echo "[4/4] 네컷 이미지 생성 시작..."
echo ""
echo "실행 명령: python generate_fourcut.py --num-images 10"
echo ""

# 실제 Python 스크립트 실행
python generate_fourcut.py --num-images 10

echo ""
echo "======================================"
echo "  완료!"
echo "======================================"

