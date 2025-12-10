"""프레임 슬롯 자동 탐지 테스트 스크립트"""
from PIL import Image
import numpy as np

frame_path = "photo-booth-frame-transparent_20px.png"
frame = Image.open(frame_path).convert("RGBA")
print(f"프레임 크기: {frame.size[0]}x{frame.size[1]}")

alpha = frame.split()[3]
w, h = frame.size
alpha_np = np.array(alpha)

# 투명 영역 마스크
mask = (alpha_np == 0).astype(np.uint8)
print(f"투명 픽셀 수: {mask.sum()}")

# y 방향 프로파일
row_counts = mask.sum(axis=1)

slots_y_ranges = []
in_region = False
start_y = 0
threshold = int(0.1 * w)

for y in range(h):
    if row_counts[y] > threshold and not in_region:
        in_region = True
        start_y = y
    elif row_counts[y] <= threshold and in_region:
        in_region = False
        end_y = y - 1
        slots_y_ranges.append((start_y, end_y))

if in_region:
    slots_y_ranges.append((start_y, h - 1))

# 필터링 (너무 얇은 영역 제거)
slots_y_ranges = [(sy, ey) for (sy, ey) in slots_y_ranges if (ey - sy) > 10]

print(f"\n감지된 Y 구간 개수: {len(slots_y_ranges)}")

# 각 Y 구간에 대해 X 방향 bounding box 계산
slots = []
for idx, (sy, ey) in enumerate(slots_y_ranges[:4], 1):
    submask = mask[sy:ey + 1, :]
    col_counts = submask.sum(axis=0)

    in_x_region = False
    start_x = 0
    threshold_x = int(0.1 * (ey - sy + 1))

    x_ranges = []
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

    if not x_ranges:
        continue

    x_ranges.sort(key=lambda r: (r[1] - r[0]), reverse=True)
    sx, ex = x_ranges[0]

    slot_w = ex - sx + 1
    slot_h = ey - sy + 1
    slots.append((sx, sy, slot_w, slot_h))

    print(f"슬롯 {idx}: x={sx}, y={sy}, w={slot_w}, h={slot_h}")

print(f"\n총 감지된 슬롯: {len(slots)}개")

if len(slots) == 4:
    print("\n✅ 4개 슬롯 감지 성공!")
    print("\ngenerate_fourcut.py에 붙여넣을 코드:")
    print("\nSLOTS_HARDCODED = [")
    for i, (x, y, w, h) in enumerate(slots, 1):
        print(f"    ({x}, {y}, {w}, {h}),  # {i}번째 칸")
    print("]")
else:
    print(f"\n⚠️ 경고: {len(slots)}개 슬롯 감지됨 (4개 필요)")

