# 네컷 사진 자동 생성 프로그램
# 일반 사진 4장을 골라서 네컷 프레임에 합성해줌

import argparse
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

# Pillow 버전 호환성
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


# 네컷 프레임 슬롯 위치 (900x2700 프레임 기준)
# (x좌표, y좌표, 가로, 세로) 형식
SLOTS_HARDCODED = [
    (40, 40, 820, 640),       # 1번째 칸
    (40, 700, 820, 640),      # 2번째 칸
    (40, 1360, 820, 640),     # 3번째 칸
    (40, 2020, 820, 640),     # 4번째 칸
]


def load_images(source_dir):
    """
    폴더에서 일반 사진들을 불러옴
    """
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {source_dir}")

    # jpg, png 파일만 찾기
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = [
        p for p in source_path.iterdir()
        if p.is_file() and p.suffix in exts
    ]

    if not image_paths:
        raise ValueError(f"폴더에 이미지가 없습니다: {source_dir}")

    return image_paths


def load_frames(frame_dir):
    """
    폴더에서 네컷 프레임들을 불러옴
    """
    frame_path = Path(frame_dir)
    if not frame_path.exists() or not frame_path.is_dir():
        raise FileNotFoundError(f"프레임 폴더를 찾을 수 없습니다: {frame_dir}")

    # png 파일만 찾기
    exts = {".png", ".PNG"}
    frame_paths = [
        p for p in frame_path.iterdir()
        if p.is_file() and p.suffix in exts
    ]

    if not frame_paths:
        raise ValueError(f"폴더에 프레임 이미지가 없습니다: {frame_dir}")

    return frame_paths


def detect_slots(frame):
    """
    프레임의 투명한 부분(슬롯)을 자동으로 찾기
    - 알파 채널(투명도)을 분석해서 4개의 칸을 찾음
    """

    Returns:
        [(x, y, w, h), ...] 형태의 슬롯 리스트 (위→아래 순서)
    """
    frame = frame.convert("RGBA")
    alpha = frame.split()[3]  # A 채널
    w, h = frame.size

    alpha_np = np.array(alpha)
    # 투명(슬롯)인 영역: alpha == 0
    mask = (alpha_np == 0).astype(np.uint8)

    # 투명 픽셀이 아예 없다면 예외
    if not mask.any():
        raise ValueError("alpha=0 투명 슬롯 영역을 찾을 수 없습니다. 하드코딩된 SLOTS를 사용하세요.")

    # 전체 투명 영역의 y 방향 프로파일 (각 행에 투명 픽셀 몇 개 있는지)
    row_counts = mask.sum(axis=1)  # shape: (h,)

    # '슬롯'이 있는 y 구간(행)들을 연속된 덩어리로 그룹화
    slots_y_ranges: List[Tuple[int, int]] = []
    in_region = False
    start_y = 0

    threshold = int(0.1 * w)  # 한 행에 전체 폭의 10% 이상 투명 픽셀이 있으면 슬롯의 일부라고 가정

    for y in range(h):
        if row_counts[y] > threshold and not in_region:
            # 새 슬롯 시작
            in_region = True
            start_y = y
        elif row_counts[y] <= threshold and in_region:
            # 슬롯 끝
            in_region = False
            end_y = y - 1
            slots_y_ranges.append((start_y, end_y))

    # 마지막이 열린 채로 끝났다면 마감
    if in_region:
        slots_y_ranges.append((start_y, h - 1))

    # 슬롯이 네 개가 아닐 수 있으므로 필터/정렬
    # 매우 얇은 영역(노이즈) 제거
    slots_y_ranges = [
        (sy, ey) for (sy, ey) in slots_y_ranges
        if (ey - sy) > 10
    ]

    if len(slots_y_ranges) < 4:
        raise ValueError(
            f"감지된 슬롯 영역이 4개 미만입니다. 감지된 개수: {len(slots_y_ranges)}. "
            "하드코딩된 SLOTS_HARDCODED를 사용하거나 --slot-mode hardcoded로 실행하세요."
        )

    # 위→아래 순으로 정렬
    slots_y_ranges.sort(key=lambda r: r[0])

    # 상위 4개만 사용
    slots_y_ranges = slots_y_ranges[:4]

    # 각 y 구간에 대해 x 방향 bounding box 계산
    slots: List[Tuple[int, int, int, int]] = []
    for (sy, ey) in slots_y_ranges:
        # 이 y 범위 내에서만 다시 마스크를 봄
        submask = mask[sy:ey + 1, :]  # shape: (ey-sy+1, w)

        col_counts = submask.sum(axis=0)
        # 슬롯이 차지하는 x 구간 찾기
        in_x_region = False
        start_x = 0
        threshold_x = int(0.1 * (ey - sy + 1))  # 열 기준으로도 최소 길이 조건

        x_ranges: List[Tuple[int, int]] = []
        for x in range(w):
            if col_counts[x] > threshold_x and not in_x_region:
                in_x_region = True
                start_x = x
            elif col_counts[x] <= threshold_x and in_x_region:
                in_x_region = False
                end_x = x - 1
                x_ranges.append((start_x, end_x))

        if in_x_region:
            x_ranges.append((start_x, w - 1))

        # 가장 넓은 x 구간을 슬롯으로 선택
        if not x_ranges:
            # x방향으로는 연속된 슬롯 영역이 없다고 판단
            continue

        x_ranges.sort(key=lambda r: (r[1] - r[0]), reverse=True)
        sx, ex = x_ranges[0]

        slot_w = ex - sx + 1
        slot_h = ey - sy + 1
        slots.append((sx, sy, slot_w, slot_h))

    if len(slots) != 4:
        raise ValueError(
            f"자동 슬롯 탐지 결과 슬롯 개수가 4개가 아닙니다: {len(slots)}개. "
            "하드코딩된 SLOTS_HARDCODED를 사용하거나 프레임 디자인/감지 로직을 조정하세요."
        )

    # y 좌표 기준으로 다시 정렬 (위에서 아래 순)
    slots.sort(key=lambda s: s[1])

    return slots


def crop_to_slot(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    이미지를 슬롯 비율에 맞게 center-crop 후, 슬롯 크기로 리사이즈합니다.

    - 모든 이미지는 RGB로 변환합니다.
    - 이미지가 작더라도 Pillow의 resize가 자동으로 upscaling 합니다.

    Args:
        image: 원본 이미지
        target_size: (width, height) 슬롯 크기

    Returns:
        슬롯 크기에 맞게 잘라서 리사이즈된 이미지
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    target_w, target_h = target_size
    w, h = image.size

    target_ratio = target_w / target_h
    src_ratio = w / h

    # 비율 비교 후 center crop
    if src_ratio > target_ratio:
        # 원본이 더 가로로 긴 경우 -> 좌우를 잘라냄
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        right = left + new_w
        top = 0
        bottom = h
    else:
        # 원본이 더 세로로 긴 경우 -> 위아래를 잘라냄
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        bottom = top + new_h
        left = 0
        right = w

    image_cropped = image.crop((left, top, right, bottom))
    image_resized = image_cropped.resize((target_w, target_h), LANCZOS)
    return image_resized


def compose_fourcut(
    frame: Image.Image,
    slot_images: List[Image.Image],
    slots: List[Tuple[int, int, int, int]],
) -> Image.Image:
    """
    프레임 이미지와 슬롯 이미지를 합성해 하나의 네컷 이미지를 생성합니다.

    Args:
        frame: 네컷 프레임 PNG (알파 채널 포함 권장)
        slot_images: 슬롯에 들어갈 4장의 이미지
        slots: (x, y, w, h) 슬롯 좌표 리스트 (길이 4)

    Returns:
        합성된 최종 네컷 이미지 (RGBA)
    """
    if len(slot_images) != 4 or len(slots) != 4:
        raise ValueError("slot_images와 slots는 반드시 길이가 4여야 합니다.")

    frame = frame.convert("RGBA")
    canvas = Image.new("RGBA", frame.size, (0, 0, 0, 0))

    # 각 슬롯 위치에 사진 붙이기
    for img, (x, y, w, h) in zip(slot_images, slots):
        prepared = crop_to_slot(img, (w, h))
        canvas.paste(prepared, (x, y))

    # 최상단에 프레임 덮어쓰기 (알파 채널을 mask로 사용)
    canvas.paste(frame, (0, 0), frame)
    return canvas


def save_image(image: Image.Image, output_dir: str, index: int) -> Path:
    """
    합성된 네컷 이미지를 저장합니다.

    파일명 형식: fourcut_0001.png

    Args:
        image: 저장할 이미지
        output_dir: 출력 디렉토리
        index: 1부터 시작하는 인덱스

    Returns:
        저장된 파일 경로
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"fourcut_{index:04d}.png"
    save_path = output_path / filename

    image.save(save_path, format="PNG")
    return save_path


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="네컷 포토부스 합성 이미지 자동 생성 스크립트"
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw/normal",
        help="합성에 사용할 일반 사진 폴더 (기본값: data/raw/normal)",
    )

    parser.add_argument(
        "--frame-path",
        type=str,
        default=None,
        help="네컷 프레임 PNG 경로 (단일 파일). --frame-dir와 함께 사용 불가",
    )

    parser.add_argument(
        "--frame-dir",
        type=str,
        default="data/raw/fourcut_frame",
        help="네컷 프레임 PNG 폴더 (랜덤 선택). 기본값: data/raw/fourcut_frame",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train/fourcut",
        help="결과 저장 폴더 (기본값: data/train/fourcut)",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="생성할 네컷 이미지 개수 (기본값: 100)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="랜덤 시드 고정 값 (기본값: 42)",
    )

    parser.add_argument(
        "--slot-mode",
        type=str,
        choices=["hardcoded", "auto"],
        default="hardcoded",
        help=(
            "슬롯 탐지 방식 선택: "
            "'hardcoded'는 코드 상단 SLOTS_HARDCODED 사용, "
            "'auto'는 alpha=0 자동 탐지 사용 (기본값: hardcoded)"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """메인 실행 함수"""
    args = parse_args()

    # 랜덤 시드 고정
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    print(f"네컷 포토부스 합성 이미지 생성을 시작합니다...")
    print(f"  - 소스 디렉토리: {args.source_dir}")
    print(f"  - 출력 디렉토리: {args.output_dir}")
    print(f"  - 생성할 이미지 개수: {args.num_images}")
    print(f"  - 슬롯 탐지 방식: {args.slot_mode}")

    # 프레임 로드 방식 결정
    frame_paths = None
    single_frame = None

    if args.frame_path and args.frame_dir:
        raise ValueError("--frame-path와 --frame-dir은 동시에 사용할 수 없습니다.")

    if args.frame_path:
        # 단일 프레임 모드
        frame_path = Path(args.frame_path)
        if not frame_path.exists():
            raise FileNotFoundError(f"프레임 이미지를 찾을 수 없습니다: {frame_path}")
        single_frame = Image.open(frame_path).convert("RGBA")
        print(f"  - 프레임 모드: 단일 프레임")
        print(f"  - 프레임 경로: {args.frame_path}")
        print(f"\n프레임 이미지 로드 완료: {single_frame.size[0]}x{single_frame.size[1]}")
    else:
        # 랜덤 프레임 모드 (기본)
        print(f"  - 프레임 모드: 랜덤 선택")
        print(f"  - 프레임 디렉토리: {args.frame_dir}")
        frame_paths = load_frames(args.frame_dir)
        print(f"\n사용 가능한 프레임: {len(frame_paths)}개")
        for fp in frame_paths:
            print(f"  - {fp.name}")

    # 슬롯 좌표 결정 (첫 번째 프레임으로 탐지 또는 하드코딩)
    if args.slot_mode == "auto":
        print("\n투명 슬롯 자동 탐지 중...")
        test_frame = single_frame if single_frame else Image.open(frame_paths[0]).convert("RGBA")
        try:
            slots = detect_slots(test_frame)
            print(f"슬롯 자동 탐지 완료: {len(slots)}개")
            for i, (x, y, w, h) in enumerate(slots, 1):
                print(f"  슬롯 {i}: x={x}, y={y}, w={w}, h={h}")
        except Exception as e:
            print(f"경고: 자동 슬롯 탐지 실패 - {e}")
            print("하드코딩된 슬롯 좌표를 사용합니다.")
            slots = SLOTS_HARDCODED
    else:
        slots = SLOTS_HARDCODED
        print(f"\n하드코딩된 슬롯 사용: {len(slots)}개")

    if len(slots) != 4:
        raise ValueError(
            f"슬롯 좌표 개수가 4개가 아닙니다: {len(slots)}개. "
            "SLOTS_HARDCODED 또는 detect_slots 로직을 확인하세요."
        )

    # 소스 이미지 목록 로드
    print(f"\n소스 이미지 로딩 중...")
    image_paths = load_images(args.source_dir)
    print(f"사용 가능한 소스 이미지: {len(image_paths)}개")

    if len(image_paths) < 4:
        print(f"경고: 소스 이미지가 4개 미만입니다. 중복 사용됩니다.")

    # num_images 개수만큼 네컷 생성
    print(f"\n네컷 이미지 생성 시작...\n")
    total = args.num_images

    for i in range(1, total + 1):
        # 프레임 선택 (랜덤 모드일 경우 매번 선택)
        if frame_paths:
            current_frame_path = random.choice(frame_paths)
            frame = Image.open(current_frame_path).convert("RGBA")
            frame_name = current_frame_path.name
        else:
            frame = single_frame
            frame_name = Path(args.frame_path).name

        # 4장 랜덤 선택 (중복 허용)
        selected_paths = random.choices(image_paths, k=4)

        slot_images: List[Image.Image] = []
        for p in selected_paths:
            try:
                img = Image.open(p)
                slot_images.append(img)
            except Exception as e:
                print(f"  경고: 이미지 로드 실패 ({p.name}): {e}")
                # 실패 시 첫 번째 이미지로 대체
                fallback_img = Image.open(image_paths[0])
                slot_images.append(fallback_img)

        # 합성
        composed = compose_fourcut(frame, slot_images, slots)

        # 저장
        save_path = save_image(composed, args.output_dir, i)

        # 프레임 정보 포함한 로그 출력
        print(f"[{i}/{total}] {save_path.name} 생성 완료 (프레임: {frame_name})")

    print(f"\n✓ 네컷 이미지 생성 완료!")
    print(f"  총 {total}개 이미지가 {args.output_dir}에 저장되었습니다.")
    if frame_paths:
        print(f"  {len(frame_paths)}개의 프레임이 랜덤으로 사용되었습니다.")


if __name__ == "__main__":
    main()

