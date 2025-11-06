"""
Model utilities for InternVL3-8B video inference.
Includes image preprocessing, video loading, and model setup functions.
"""

import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig
from typing import List, Dict, Optional, Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """
    Build image transformation pipeline with IMAGENET normalization.

    Args:
        input_size: Target size for resizing (square)

    Returns:
        torchvision Transform composition
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    DEPRECATED: Use load_video_force_2x1_grid() for 16:9 videos instead.

    Find the closest aspect ratio from target ratios for dynamic preprocessing.

    Args:
        aspect_ratio: Original image aspect ratio (width/height)
        target_ratios: List of (width_ratio, height_ratio) tuples
        width: Original image width
        height: Original image height
        image_size: Base image size

    Returns:
        Best matching (width_ratio, height_ratio) tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    DEPRECATED: Use load_video_force_2x1_grid() for 16:9 videos instead.

    Dynamically preprocess image into multiple patches based on aspect ratio.

    Args:
        image: PIL Image
        min_num: Minimum number of patches
        max_num: Maximum number of patches
        image_size: Base size for each patch
        use_thumbnail: Whether to include a thumbnail of the whole image

    Returns:
        List of PIL Images (patches)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate possible aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split into patches
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """
    Load and preprocess a single image file.

    Args:
        image_file: Path to image file
        input_size: Size for preprocessing
        max_num: Maximum number of patches

    Returns:
        Tensor of stacked image patches
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    Calculate frame indices for uniform video sampling.

    Args:
        bound: Optional (start_time, end_time) tuple in seconds
        fps: Video frames per second
        max_frame: Total number of frames in video
        first_idx: Starting frame index
        num_segments: Number of segments to sample

    Returns:
        numpy array of frame indices
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000

    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments

    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])

    return frame_indices


def load_video_force_2x1_grid(
    video_path: str,
    num_segments: int = 10,
    input_size: int = 448
) -> Tuple[torch.Tensor, List[int], List[Image.Image]]:
    """
    Load video frames with fixed 2x1 grid preprocessing for 16:9 videos.

    This function replaces dynamic preprocessing with a fixed approach:
    - Uniformly samples num_segments frames from the video
    - Resizes each frame to 896x448 (2:1 ratio, perfect for 16:9 videos)
    - Splits each frame into 2 patches: [left 448x448, right 448x448]
    - Preserves high resolution text for better OCR performance

    Args:
        video_path: Path to video file
        num_segments: Number of frames to uniformly sample (default: 10)
        input_size: Base size for each patch (default: 448, creates 896x448 frames)

    Returns:
        tuple: (pixel_values, num_patches_list, pil_images)
            - pixel_values: Tensor [num_segments*2, 3, 448, 448] of all patches
            - num_patches_list: List of 2's, one per frame [2, 2, 2, ...]
            - pil_images: List of PIL Images (896x448) for OCR processing

    Example:
        For 10 frames:
        - pixel_values shape: [20, 3, 448, 448] (10 frames * 2 patches each)
        - num_patches_list: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        - pil_images: 10 PIL Images at 896x448 resolution
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list = []
    num_patches_list = []
    pil_images_for_ocr = []

    # Transform for normalization only (no resize, we handle that manually)
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Get uniformly sampled frame indices
    frame_indices = get_index(
        bound=None,
        fps=fps,
        max_frame=max_frame,
        num_segments=num_segments
    )

    # Grid dimensions: 2x1 (width x height in patches)
    target_height = input_size       # 448
    target_width = input_size * 2    # 896

    for frame_index in frame_indices:
        # Load frame as PIL Image
        img_pil = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')

        # Store original resized frame for OCR
        resized_img = img_pil.resize((target_width, target_height), Image.BICUBIC)
        pil_images_for_ocr.append(resized_img)

        # Split into 2 patches: left half and right half
        patch_left = resized_img.crop((0, 0, input_size, input_size))
        patch_right = resized_img.crop((input_size, 0, target_width, input_size))

        # Apply normalization to each patch
        pixel_values_left = transform_normalize(patch_left)
        pixel_values_right = transform_normalize(patch_right)

        # Stack the 2 patches for this frame
        frame_patches = torch.stack([pixel_values_left, pixel_values_right])  # [2, 3, 448, 448]

        pixel_values_list.append(frame_patches)
        num_patches_list.append(2)  # Always 2 patches per frame

    # Concatenate all patches from all frames
    pixel_values = torch.cat(pixel_values_list)  # [num_segments*2, 3, 448, 448]

    print(f"[2x1 Grid Loader] Loaded {len(pil_images_for_ocr)} frames from video")
    print(f"[2x1 Grid Loader] Created {pixel_values.shape[0]} patches ({len(pil_images_for_ocr)} frames × 2 patches)")
    print(f"[2x1 Grid Loader] Pixel values shape: {pixel_values.shape}")

    return pixel_values, num_patches_list, pil_images_for_ocr


def load_video_from_indices_2x1_grid(
    video_path: str,
    frame_indices: List[int],
    input_size: int = 448
) -> Tuple[torch.Tensor, List[int], List[Image.Image]]:
    """
    Load specific frames from video with fixed 2x1 grid preprocessing.
    Used for YOLO-selected frames with high-resolution text preservation.

    This combines YOLO's intelligent frame selection with the 2x1 grid approach:
    - Loads frames at specific indices (from YOLO detection)
    - Resizes each frame to 896x448 (2:1 ratio, perfect for 16:9 videos)
    - Splits each frame into 2 patches: [left 448x448, right 448x448]
    - Preserves high resolution text for better OCR performance

    Args:
        video_path: Path to video file
        frame_indices: List of specific frame indices to load (from YOLO)
        input_size: Base size for each patch (default: 448, creates 896x448 frames)

    Returns:
        tuple: (pixel_values, num_patches_list, pil_images)
            - pixel_values: Tensor [len(frame_indices)*2, 3, 448, 448] of all patches
            - num_patches_list: List of 2's, one per frame [2, 2, 2, ...]
            - pil_images: List of PIL Images (896x448) for OCR processing

    Example:
        For 8 YOLO-selected frames:
        - pixel_values shape: [16, 3, 448, 448] (8 frames * 2 patches each)
        - num_patches_list: [2, 2, 2, 2, 2, 2, 2, 2]
        - pil_images: 8 PIL Images at 896x448 resolution
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    pixel_values_list = []
    num_patches_list = []
    pil_images_for_ocr = []

    # Transform for normalization only (no resize, we handle that manually)
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Grid dimensions: 2x1 (width x height in patches)
    target_height = input_size       # 448
    target_width = input_size * 2    # 896

    for frame_index in frame_indices:
        # Ensure frame_index is within bounds
        frame_index = min(frame_index, len(vr) - 1)

        # Load frame as PIL Image
        img_pil = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')

        # Store original resized frame for OCR
        resized_img = img_pil.resize((target_width, target_height), Image.BICUBIC)
        pil_images_for_ocr.append(resized_img)

        # Split into 2 patches: left half and right half
        patch_left = resized_img.crop((0, 0, input_size, input_size))
        patch_right = resized_img.crop((input_size, 0, target_width, input_size))

        # Apply normalization to each patch
        pixel_values_left = transform_normalize(patch_left)
        pixel_values_right = transform_normalize(patch_right)

        # Stack the 2 patches for this frame
        frame_patches = torch.stack([pixel_values_left, pixel_values_right])  # [2, 3, 448, 448]

        pixel_values_list.append(frame_patches)
        num_patches_list.append(2)  # Always 2 patches per frame

    # Concatenate all patches from all frames
    pixel_values = torch.cat(pixel_values_list)  # [len(frame_indices)*2, 3, 448, 448]

    print(f"[2x1 Grid YOLO Loader] Loaded {len(pil_images_for_ocr)} YOLO-selected frames")
    print(f"[2x1 Grid YOLO Loader] Created {pixel_values.shape[0]} patches ({len(pil_images_for_ocr)} frames × 2 patches)")

    return pixel_values, num_patches_list, pil_images_for_ocr


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    DEPRECATED: Use load_video_force_2x1_grid() for 16:9 videos instead.

    Load and preprocess video frames with uniform sampling.

    Args:
        video_path: Path to video file
        bound: Optional (start_time, end_time) tuple
        input_size: Size for preprocessing
        max_num: Maximum patches per frame
        num_segments: Number of frames to sample

    Returns:
        tuple: (pixel_values tensor, list of num_patches per frame)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_video_from_indices(video_path, frame_indices, input_size=448, max_num=1):
    """
    DEPRECATED: Use load_video_force_2x1_grid() for 16:9 videos instead.

    Load specific frames from video by their indices.
    Used for YOLO-based frame selection.

    Args:
        video_path: Path to video file
        frame_indices: List of specific frame indices to load
        input_size: Size for preprocessing
        max_num: Maximum patches per frame

    Returns:
        tuple: (pixel_values tensor, list of num_patches per frame)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    for frame_index in frame_indices:
        # Ensure frame_index is within bounds
        frame_index = min(frame_index, len(vr) - 1)

        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_video_from_indices_with_context(
    video_path: str,
    frame_indices: List[int],
    detections_dict: Dict[int, List[str]],
    input_size: int = 448,
    max_num: int = 1
) -> Tuple[torch.Tensor, List[int], Dict[int, List[str]]]:
    """
    Load specific frames from video with detection context.
    Enhanced version that also returns detection metadata for prompt generation.

    Args:
        video_path: Path to video file
        frame_indices: List of specific frame indices to load
        detections_dict: Dict mapping frame_index -> List of detected sign names
        input_size: Size for preprocessing
        max_num: Maximum patches per frame

    Returns:
        tuple: (pixel_values tensor, list of num_patches per frame, detections_dict)
    """
    pixel_values, num_patches_list = load_video_from_indices(
        video_path,
        frame_indices,
        input_size,
        max_num
    )

    return pixel_values, num_patches_list, detections_dict


def split_model(model_name):
    """
    Create device map for multi-GPU model distribution.
    Distributes model layers across available GPUs.

    Args:
        model_name: Name of the model (for configuration)

    Returns:
        Dictionary mapping layer names to GPU indices
    """
    device_map = {}
    world_size = torch.cuda.device_count()

    if world_size <= 1:
        return "auto"

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
    except:
        # Default fallback
        return "auto"

    # First GPU handles ViT (treat as half GPU)
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    # Place core components
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
