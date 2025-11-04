"""
YOLO-based intelligent frame selection for traffic videos.
Detects Vietnamese traffic signs, ranks frames by relevance.
"""

import cv2
from ultralytics import YOLO
import torch
import time
import os
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

# --- Configuration Constants ---

# Stage 1: Super-fast Filtering
BLUR_THRESHOLD = 100.0  # Threshold to remove blurry frames (higher = clearer)
YOLO_CONFIDENCE = 0.3   # Minimum confidence for YOLO detection

# Stage 2: Ranking
TOP_K_FRAMES = 3  # Number of "best" frames we want to select

# Vietnamese Traffic Sign Classes (52 classes)
VIETNAMESE_SIGN_CLASSES = {
    0: 'Đường người đi bộ cắt ngang',
    1: 'Đường giao nhau (ngã ba bên phải)',
    2: 'Cấm đi ngược chiều',
    3: 'Phải đi vòng sang bên phải',
    4: 'Giao nhau với đường đồng cấp',
    5: 'Giao nhau với đường không ưu tiên',
    6: 'Chỗ ngoặt nguy hiểm vòng bên trái',
    7: 'Cấm rẽ trái',
    8: 'Bến xe buýt',
    9: 'Nơi giao nhau chạy theo vòng xuyến',
    10: 'Cấm dừng và đỗ xe',
    11: 'Chỗ quay xe',
    12: 'Biển gộp làn đường theo phương tiện',
    13: 'Đi chậm',
    14: 'Cấm xe tải',
    15: 'Đường bị thu hẹp về phía phải',
    16: 'Giới hạn chiều cao',
    17: 'Cấm quay đầu',
    18: 'Cấm ô tô khách và ô tô tải',
    19: 'Cấm rẽ phải và quay đầu',
    20: 'Cấm ô tô',
    21: 'Đường bị thu hẹp về phía trái',
    22: 'Gồ giảm tốc phía trước',
    23: 'Cấm xe hai và ba bánh',
    24: 'Kiểm tra',
    25: 'Chỉ dành cho xe máy*',
    26: 'Chướng ngoại vật phía trước',
    27: 'Trẻ em',
    28: 'Xe tải và xe công*',
    29: 'Cấm mô tô và xe máy',
    30: 'Chỉ dành cho xe tải*',
    31: 'Đường có camera giám sát',
    32: 'Cấm rẽ phải',
    33: 'Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải',
    34: 'Cấm xe sơ-mi rơ-moóc',
    35: 'Cấm rẽ trái và phải',
    36: 'Cấm đi thẳng và rẽ phải',
    37: 'Đường giao nhau (ngã ba bên trái)',
    38: 'Giới hạn tốc độ (50km/h)',
    39: 'Giới hạn tốc độ (60km/h)',
    40: 'Giới hạn tốc độ (80km/h)',
    41: 'Giới hạn tốc độ (40km/h)',
    42: 'Các xe chỉ được rẽ trái',
    43: 'Chiều cao tĩnh không thực tế',
    44: 'Nguy hiểm khác',
    45: 'Đường một chiều',
    46: 'Cấm đỗ xe',
    47: 'Cấm ô tô quay đầu xe (được rẽ trái)',
    48: 'Giao nhau với đường sắt có rào chắn',
    49: 'Cấm rẽ trái và quay đầu xe',
    50: 'Chỗ ngoặt nguy hiểm vòng bên phải',
    51: 'Chú ý chướng ngại vật – vòng tránh sang bên phải'
}


def find_best_frames_with_context(
    video_path: str,
    yolo_model_path: str,
    top_k: int = TOP_K_FRAMES,
    device: str = 'cuda',
    confidence: float = YOLO_CONFIDENCE
) -> Optional[Tuple[List[int], Dict[int, List[str]]]]:
    """
    Analyze video to find the top_k best frames containing Vietnamese traffic signs.
    Returns frame indices AND detected sign class names for each frame.

    Process:
    1. (Stage 1) Super-fast Filtering: Scan through video, find clear frames
       that contain traffic signs (using custom trained YOLO).
    2. (Stage 2) Ranking Candidates: Rank filtered frames based on
       total sign area (sum of all signs in frame).

    Args:
        video_path: Path to the video file
        yolo_model_path: Path to trained YOLO model (.pt file)
        top_k: Number of best frames to return
        device: 'cuda' or 'cpu'
        confidence: Minimum confidence threshold

    Returns:
        Tuple of (frame_indices, detections_dict) where:
        - frame_indices: List of top_k frame indices
        - detections_dict: Dict mapping frame_index -> List of detected sign names
        Returns (None, None) if no suitable frames are found.
    """

    print(f"[Frame Selector] Starting video processing: {video_path}")

    # Check if GPU is available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print(f"[Frame Selector] CUDA not available, using CPU")
    else:
        print(f"[Frame Selector] Using device: {device}")

    # Load custom YOLO model
    try:
        print(f"[Frame Selector] Loading custom YOLO model: {yolo_model_path}")
        model = YOLO(yolo_model_path)
        model.to(device)
    except Exception as e:
        print(f"[Frame Selector] Error loading YOLO model: {e}")
        return (None, None)

    # Open video
    if not os.path.exists(video_path):
        print(f"[Frame Selector] Error: Video file not found: {video_path}")
        return (None, None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Frame Selector] Error: Cannot open video file: {video_path}")
        return (None, None)

    frame_index = 0
    candidates = []  # List to store candidates

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Frame Selector] Total frames: {total_frames}. Starting Stage 1: Super-fast Filtering...")
    start_time = time.time()

    # --- STAGE 1: SUPER-FAST FILTERING ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # 1. Blur Check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < BLUR_THRESHOLD:
            frame_index += 1
            continue  # Skip blurry frames

        # 2. Signal Check
        # Run custom YOLO on clear frames
        results = model(frame, device=device, verbose=False, conf=confidence)

        # Store candidates (ALL classes are relevant)
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            candidates.append({
                'frame_index': frame_index,
                'box': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                'class_id': class_id,
                'confidence': float(box.conf[0])
            })

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"  [Frame Selector] Scanned {frame_index}/{total_frames} frames...")

    cap.release()
    end_time_step1 = time.time()
    print(f"[Frame Selector] --- Stage 1 Complete (in {end_time_step1 - start_time:.2f}s) ---")
    print(f"[Frame Selector] Found {len(candidates)} potential candidates (bounding boxes).")

    if not candidates:
        print("[Frame Selector] No traffic signs detected.")
        return (None, None)

    # --- STAGE 2: RANKING CANDIDATES ---
    print("[Frame Selector] Starting Stage 2: Ranking Candidates...")

    # Calculate TOTAL sign area per frame (sum all signs)
    frame_total_area = defaultdict(float)
    frame_detections = defaultdict(set)  # Use set to avoid duplicates

    for cand in candidates:
        box = cand['box']
        # Calculate area: (x2 - x1) * (y2 - y1)
        area = (box[2] - box[0]) * (box[3] - box[1])

        idx = cand['frame_index']
        class_id = cand['class_id']

        # Sum total area for this frame
        frame_total_area[idx] += area

        # Store detected sign class name
        if class_id in VIETNAMESE_SIGN_CLASSES:
            frame_detections[idx].add(VIETNAMESE_SIGN_CLASSES[class_id])

    # Sort frames by total area (largest first)
    sorted_frames = sorted(frame_total_area.items(), key=lambda item: item[1], reverse=True)

    # Get Top-K frame indices
    best_frame_indices = [idx for idx, area in sorted_frames[:top_k]]

    # Convert detections to dict with lists
    detections_dict = {
        idx: sorted(list(frame_detections[idx]))
        for idx in best_frame_indices
    }

    end_time_step2 = time.time()
    print(f"[Frame Selector] --- Stage 2 Complete (in {end_time_step2 - end_time_step1:.2f}s) ---")
    print(f"[Frame Selector] Total processing time: {end_time_step2 - start_time:.2f} seconds")
    print(f"[Frame Selector] Top {top_k} frames: {best_frame_indices}")

    # Print detected signs for each frame
    for idx in best_frame_indices:
        signs = detections_dict[idx]
        print(f"  Frame {idx}: {len(signs)} sign(s) - {', '.join(signs[:3])}{'...' if len(signs) > 3 else ''}")

    return (best_frame_indices, detections_dict)


# --- Testing ---
if __name__ == "__main__":
    import sys

    # Test the frame selector
    video_file = "../../RoadBuddy/traffic_buddy_train+public_test/public_test/videos/efc9909e_908_clip_001_0000_0009_Y.mp4"
    yolo_model = "path/to/your/trained/model.pt"  # UPDATE THIS PATH

    if len(sys.argv) > 1:
        yolo_model = sys.argv[1]

    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' does not exist. Please update the path.")
    elif not os.path.exists(yolo_model):
        print(f"YOLO model '{yolo_model}' does not exist. Please provide model path.")
        print(f"Usage: python frame_selector.py <yolo_model_path>")
    else:
        # Run pipeline
        top_frames, detections = find_best_frames_with_context(
            video_file,
            yolo_model,
            top_k=3
        )

        if top_frames and detections:
            print(f"\n[RESULT] Top {TOP_K_FRAMES} 'golden' frames are (indices): {top_frames}")
            print("\nDetected signs per frame:")
            for idx in top_frames:
                print(f"  Frame {idx}: {', '.join(detections[idx])}")

            # (Optional) Save these frames as images for verification
            print("\nExtracting 'golden' frames to image files for verification...")
            cap = cv2.VideoCapture(video_file)
            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count in top_frames:
                    save_name = f"best_frame_{frame_count}.jpg"
                    cv2.imwrite(save_name, frame)
                    print(f"  Saved: {save_name}")
                    saved_count += 1
                frame_count += 1
            cap.release()
            print(f"Saved {saved_count} images.")
        else:
            print("\n[RESULT] No suitable frames found.")
