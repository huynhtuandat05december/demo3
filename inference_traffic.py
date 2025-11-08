"""
Traffic Video Question Answering using InternVL3-8B with YOLO-based frame selection.

This script loads traffic videos, uses intelligent frame selection via YOLO,
and performs inference using the InternVL3-8B vision-language model.
"""

import os
import json
import argparse
import time
from datetime import datetime
import csv
import re
from typing import Dict, List, Optional, Tuple
import warnings

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from model_utils import load_video, load_video_from_indices, load_video_from_indices_with_context, split_model, load_video_force_2x1_grid, load_video_from_indices_2x1_grid
from prompt_template import create_traffic_prompt, create_traffic_prompt_with_context, format_video_prefix_with_detections
from frame_selector import find_best_frames_with_context
from ocr_enhancement import SignTextExtractor
from enhanced_prompts import create_enhanced_prompt_with_few_shot, get_optimal_frame_counts

warnings.filterwarnings('ignore')


def extract_answer(response: str, num_choices: int = 4) -> str:
    """
    Extract the answer letter (A, B, C, D) from model response.

    Args:
        response: Raw model response text
        num_choices: Number of answer choices (default 4)

    Returns:
        Single letter answer (A, B, C, or D)
    """
    # Define valid answers based on num_choices
    valid_answers = ['A', 'B', 'C', 'D'][:num_choices]

    # Pattern 1: Look for explicit answer patterns
    patterns = [
        r'\b([ABCD])\b[.)]?\s*(?:là|is|:|đúng|correct)',
        r'(?:đáp án|answer|chọn|choice)[\s:]*([ABCD])\b',
        r'\b([ABCD])[.)]',
        r'\b([ABCD])\s*[-:]',
        r'^([ABCD])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).upper()
            if answer in valid_answers:
                return answer

    # Pattern 2: Find any occurrence of A, B, C, D
    for answer in valid_answers:
        if re.search(rf'\b{answer}\b', response):
            return answer

    # Default fallback
    print(f"[WARNING] Could not extract answer from response: {response[:100]}...")
    return 'A'


def load_model(model_name: str, load_in_8bit: bool = False, device: str = 'cuda'):
    """
    Load InternVL3 model and tokenizer.

    Args:
        model_name: HuggingFace model name or local path
        load_in_8bit: Whether to use 8-bit quantization
        device: Device to load model on

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"[Model] Loading {model_name}...")
    print(f"[Model] 8-bit quantization: {load_in_8bit}")

    # Determine device map
    if torch.cuda.device_count() > 1:
        device_map = split_model(model_name)
        print(f"[Model] Using multi-GPU with {torch.cuda.device_count()} GPUs")
    else:
        device_map = "auto"
        print(f"[Model] Using single device: {device}")

    # Load model
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    print("[Model] Model loaded successfully!")
    return model, tokenizer


def create_video_prefix(num_frames: int) -> str:
    """
    Create video frame prefix for prompt with 2x1 grid explanation.

    Args:
        num_frames: Number of frames in the video

    Returns:
        Formatted prefix string with grid context
    """
    header = "Mỗi khung hình video được hiển thị dưới dạng hình ảnh toàn cảnh (chiều rộng đầy đủ).\n\n"
    frames = ''.join([f'Khung hình {i+1}: <image>\n' for i in range(num_frames)])
    return header + frames


def process_single_question(
    question_data: Dict,
    model,
    tokenizer,
    video_cache: Dict,
    base_path: str,
    num_frames_yolo: int,
    num_frames_normal: int,
    max_num: int,
    use_yolo: bool = True,
    yolo_model_path: Optional[str] = None,
    device: str = 'cuda',
    ocr_extractor: Optional[SignTextExtractor] = None,
    ocr_confidence: float = 0.6
) -> Dict:
    """
    Process a single question with detection-aware processing.

    Args:
        question_data: Dictionary containing question info
        model: Loaded InternVL3 model
        tokenizer: Loaded tokenizer
        video_cache: Cache for video frames
        base_path: Base path to data directory
        num_frames_yolo: Number of frames to extract when using YOLO
        num_frames_normal: Number of frames to extract when using uniform sampling
        max_num: Maximum number of patches per frame
        use_yolo: Whether to use YOLO for frame selection
        yolo_model_path: Path to trained YOLO model
        device: Device for inference

    Returns:
        Dictionary with results
    """
    question_id = question_data['id']
    question_text = question_data['question']
    choices = question_data['choices']
    video_path = question_data['video_path']
    num_choices = len(choices)

    # Adaptive frame selection based on question type
    adaptive_yolo, adaptive_uniform = get_optimal_frame_counts(
        question_text,
        default_yolo=num_frames_yolo,
        default_uniform=num_frames_normal
    )
    # Use adaptive counts
    num_frames_yolo_adaptive = adaptive_yolo
    num_frames_normal_adaptive = adaptive_uniform

    # Get full video path
    full_video_path = os.path.join(base_path, video_path)

    if not os.path.exists(full_video_path):
        print(f"[ERROR] Video not found: {full_video_path}")
        return {
            'id': question_id,
            'answer': 'A',
            'raw_response': 'ERROR: Video file not found',
            'prompt': '',
            'frame_strategy': 'error',
            'num_detections': 0
        }

    # Load video frames (with caching)
    if video_path not in video_cache:
        try:
            # Use 2x1 grid preprocessing with YOLO + uniform sampling combination
            yolo_pixel_values = None
            yolo_num_patches_list = None
            yolo_pil_images = None
            frame_indices = None
            detections_dict = None

            # Method 1: YOLO-based frame selection with 2x1 grid (using adaptive count)
            yolo_succeeded = False
            if use_yolo and yolo_model_path and num_frames_yolo_adaptive > 0:
                frame_indices, detections_dict = find_best_frames_with_context(
                    full_video_path,
                    yolo_model_path,
                    top_k=num_frames_yolo_adaptive,
                    device=device
                )

                if frame_indices is not None and len(frame_indices) > 0:
                    print(f"  [Video] YOLO-selected frames: {frame_indices}")
                    yolo_pixel_values, yolo_num_patches_list, yolo_pil_images = load_video_from_indices_2x1_grid(
                        video_path=full_video_path,
                        frame_indices=frame_indices,
                        input_size=448
                    )
                    yolo_succeeded = True
                else:
                    print(f"  [WARNING] YOLO detection found no frames, compensating with uniform sampling")

            # Method 2: Uniform sampling with 2x1 grid (with compensation if YOLO failed, using adaptive count)
            normal_pixel_values = None
            normal_num_patches_list = None
            normal_pil_images = None

            # Calculate uniform frame count: add YOLO frames if YOLO failed
            uniform_frame_count = num_frames_normal_adaptive
            if use_yolo and yolo_model_path and num_frames_yolo_adaptive > 0 and not yolo_succeeded:
                uniform_frame_count += num_frames_yolo_adaptive
                print(f"  [Video] Compensating for YOLO failure: {num_frames_normal_adaptive} + {num_frames_yolo_adaptive} = {uniform_frame_count} uniform frames")

            uniform_frame_indices = None
            if uniform_frame_count > 0:
                if yolo_succeeded:
                    print(f"  [Video] Uniform sampling: {uniform_frame_count} frames with 2x1 grid")
                normal_pixel_values, normal_num_patches_list, normal_pil_images, uniform_frame_indices = load_video_force_2x1_grid(
                    video_path=full_video_path,
                    num_segments=uniform_frame_count,
                    input_size=448
                )

            # Concatenate frames from both methods
            all_frame_indices = []
            if yolo_pixel_values is not None and normal_pixel_values is not None:
                # Both methods succeeded - concatenate
                pixel_values = torch.cat([yolo_pixel_values, normal_pixel_values], dim=0)
                num_patches_list = yolo_num_patches_list + normal_num_patches_list
                pil_images = yolo_pil_images + normal_pil_images
                # Combine frame indices: YOLO indices + uniform indices
                all_frame_indices = frame_indices + (uniform_frame_indices if uniform_frame_indices else [])
                strategy = '2x1_grid_yolo_and_uniform'
                print(f"  [Video] Combined: {len(yolo_num_patches_list)} YOLO + {len(normal_num_patches_list)} uniform = {len(num_patches_list)} total frames")
            elif yolo_pixel_values is not None:
                # Only YOLO succeeded
                pixel_values = yolo_pixel_values
                num_patches_list = yolo_num_patches_list
                pil_images = yolo_pil_images
                all_frame_indices = frame_indices if frame_indices else []
                strategy = '2x1_grid_yolo_only'
                print(f"  [Video] Using only YOLO frames: {len(num_patches_list)} frames")
            elif normal_pixel_values is not None:
                # Only normal succeeded
                pixel_values = normal_pixel_values
                num_patches_list = normal_num_patches_list
                pil_images = normal_pil_images
                all_frame_indices = uniform_frame_indices if uniform_frame_indices else []
                strategy = '2x1_grid_uniform_only'
                # Keep detections_dict as None for uniform-only case
                print(f"  [Video] Using only uniform frames: {len(num_patches_list)} frames")
            else:
                # Neither method succeeded
                raise Exception(f"Both YOLO and uniform sampling failed for {video_path}")

            # Cache on CPU
            video_cache[video_path] = {
                'pixel_values': pixel_values.cpu(),
                'num_patches_list': num_patches_list,
                'num_frames': len(num_patches_list),
                'pil_images': pil_images,  # Store for OCR integration
                'all_frame_indices': all_frame_indices,  # All frames (YOLO + uniform)
                'yolo_frame_indices': frame_indices,  # YOLO frames only (for detections)
                'detections_dict': detections_dict,  # Only has YOLO detections
                'strategy': strategy
            }

        except Exception as e:
            print(f"[ERROR] Failed to load video {video_path}: {e}")
            return {
                'id': question_id,
                'answer': 'A',
                'raw_response': f'ERROR: {str(e)}',
                'prompt': '',
                'frame_strategy': 'error',
                'num_detections': 0
            }

    # Get cached video data
    cached_data = video_cache[video_path]
    pixel_values = cached_data['pixel_values'].to(torch.bfloat16).to(device)
    num_patches_list = cached_data['num_patches_list']
    actual_num_frames = cached_data['num_frames']
    all_frame_indices = cached_data.get('all_frame_indices', [])
    yolo_frame_indices = cached_data.get('yolo_frame_indices')
    detections_dict = cached_data.get('detections_dict')
    strategy = cached_data.get('strategy', 'uniform_sampling')
    pil_images = cached_data.get('pil_images', [])

    # Add OCR text extraction to ALL frames if OCR is available
    if ocr_extractor and pil_images and len(all_frame_indices) > 0:
        print(f"  [OCR] Extracting text from {len(all_frame_indices)} frames (YOLO + uniform)...")
        # Convert PIL images to numpy arrays for OCR
        import numpy as np
        from PIL import Image

        video_frames = []
        for pil_img in pil_images:
            if isinstance(pil_img, Image.Image):
                # Convert PIL to numpy (RGB -> BGR for cv2 compatibility)
                img_array = np.array(pil_img)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = img_array[:, :, ::-1]  # RGB -> BGR
                video_frames.append(img_array)

        if len(video_frames) > 0:
            # Enhance detections with OCR (processes ALL frames: YOLO + uniform)
            detections_dict = ocr_extractor.batch_enhance_detections(
                video_frames,
                all_frame_indices,
                detections_dict,
                confidence_threshold=ocr_confidence
            )

    # Create prompt with enhanced few-shot examples and detection context
    # Use all_frame_indices for detections (includes both YOLO and uniform frames after OCR)
    video_prefix = format_video_prefix_with_detections(
        actual_num_frames,
        detections_dict,
        all_frame_indices
    ) if (detections_dict and all_frame_indices) else create_video_prefix(actual_num_frames)

    # Use enhanced prompt with few-shot examples
    prompt_text = create_enhanced_prompt_with_few_shot(
        question_text,
        choices,
        detections_dict=detections_dict,
        frame_indices=all_frame_indices,
        num_choices=num_choices
    )

    full_question = video_prefix + prompt_text

    # Generate response
    try:
        generation_config = {
            'max_new_tokens': 256,  # Increased from 10 to allow chain-of-thought reasoning
            'do_sample': False,
        }

        response = model.chat(
            tokenizer,
            pixel_values,
            full_question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False
        )

        # Handle if response is a tuple
        if isinstance(response, tuple):
            response = response[0]

        # Extract answer
        answer = extract_answer(response, num_choices=len(choices))

        # Count detected signs if available
        num_detections = 0
        if detections_dict and frame_indices:
            for idx in frame_indices:
                if idx in detections_dict:
                    num_detections += len(detections_dict[idx])

        result = {
            'id': question_id,
            'answer': answer,
            'raw_response': response,
            'prompt': full_question,
            'frame_strategy': strategy,
            'num_detections': num_detections
        }

        # Clear GPU cache
        if device == 'cuda':
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[ERROR] Inference failed for {question_id}: {e}")
        return {
            'id': question_id,
            'answer': 'A',
            'raw_response': f'ERROR: {str(e)}',
            'prompt': full_question,
            'frame_strategy': strategy,
            'num_detections': 0
        }


def save_results(results: List[Dict], output_dir: str, model_name: str):
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        model_name: Model name for filename
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split('/')[-1]
    filename = f"submission_{model_short}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Write CSV with all fields including frame strategy
    fieldnames = ['id', 'answer', 'raw_response', 'prompt', 'frame_strategy', 'num_detections']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Results] Saved to: {filepath}")

    # Print strategy statistics
    strategy_counts = {}
    total_detections = 0
    for result in results:
        strategy = result.get('frame_strategy', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        total_detections += result.get('num_detections', 0)

    print("\n[Statistics] Frame Selection Strategies:")
    for strategy, count in strategy_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {strategy}: {count} ({percentage:.1f}%)")
    if 'yolo_detection' in strategy_counts:
        avg_detections = total_detections / strategy_counts['yolo_detection']
        print(f"  Average detections per YOLO frame: {avg_detections:.1f}")

    # Also create a minimal submission file (only id, answer)
    minimal_filename = f"submission_minimal_{model_short}_{timestamp}.csv"
    minimal_filepath = os.path.join(output_dir, minimal_filename)

    with open(minimal_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        for result in results:
            writer.writerow({'id': result['id'], 'answer': result['answer']})

    print(f"[Results] Minimal submission saved to: {minimal_filepath}")


def main():
    parser = argparse.ArgumentParser(description='InternVL3 Traffic Video QA with YOLO Frame Selection')

    # Model arguments
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3-8B',
                        help='Model name or path')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load model in 8-bit mode')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='../RoadBuddy/traffic_buddy_train+public_test',
                        help='Path to test data directory')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to process (default: all)')

    # Video processing arguments
    parser.add_argument('--num_frames_yolo', type=int, default=8,
                        help='Number of frames to extract when using YOLO selection')
    parser.add_argument('--num_frames_normal', type=int, default=8,
                        help='Number of frames to extract when using uniform sampling')
    parser.add_argument('--max_num', type=int, default=3,
                        help='Maximum number of patches per frame')
    parser.add_argument('--yolo_model', type=str, default=None,
                        help='Path to trained YOLO model (.pt file) for Vietnamese traffic signs')
    parser.add_argument('--no_yolo', action='store_true',
                        help='Disable YOLO frame selection, use uniform sampling only')

    # OCR arguments
    parser.add_argument('--use_ocr', action='store_true',
                        help='Enable OCR text extraction from traffic signs (requires paddleocr)')
    parser.add_argument('--ocr_confidence', type=float, default=0.6,
                        help='Minimum OCR confidence threshold (default: 0.6)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save results')

    args = parser.parse_args()

    print("="*80)
    print("InternVL3 Traffic Video Question Answering")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"8-bit quantization: {args.load_in_8bit}")
    print(f"\nFrame Extraction Strategy:")
    if not args.no_yolo and args.yolo_model:
        print(f"  - YOLO frames: {args.num_frames_yolo}")
    else:
        print(f"  - YOLO frames: 0 (disabled)")
    print(f"  - Uniform frames: {args.num_frames_normal}")
    total_frames = (args.num_frames_yolo if (not args.no_yolo and args.yolo_model) else 0) + args.num_frames_normal
    print(f"  - Total frames per video: {total_frames}")
    print(f"\nMax patches per frame: {args.max_num}")
    if not args.no_yolo:
        if args.yolo_model:
            print(f"YOLO model: {args.yolo_model}")
        else:
            print("[WARNING] YOLO enabled but no model path provided. Will only use uniform sampling.")
    print("="*80)

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("[WARNING] CUDA not available, using CPU. This will be very slow!")

    # Load model
    model, tokenizer = load_model(args.model, args.load_in_8bit, device)

    # Initialize OCR extractor for Vietnamese traffic signs (if enabled)
    ocr_extractor = None
    if args.use_ocr:
        print("\n[OCR] Initializing OCR engine for Vietnamese text extraction...")
        print(f"[OCR] Confidence threshold: {args.ocr_confidence}")
        try:
            ocr_extractor = SignTextExtractor(use_paddleocr=True)
            if ocr_extractor.ocr is not None:
                print("[OCR] OCR engine initialized successfully")
            else:
                print("[OCR] OCR engine not available - will skip text extraction")
                print("[OCR] Install with: pip install paddleocr")
                ocr_extractor = None
        except Exception as e:
            print(f"[OCR] Failed to initialize OCR: {e}")
            print("[OCR] Continuing without OCR...")
            ocr_extractor = None
    else:
        print("\n[OCR] OCR disabled (use --use_ocr to enable)")

    # Load test data
    json_path = os.path.join(args.data_path, 'public_test/public_test.json')
    print(f"\n[Data] Loading test data from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data['data']
    total_questions = len(questions)
    print(f"[Data] Total questions: {total_questions}")

    # Limit samples if specified
    if args.samples is not None:
        questions = questions[:args.samples]
        print(f"[Data] Processing first {args.samples} samples")

    # Process questions
    results = []
    video_cache = {}  # Cache for video frames

    print("\n[Inference] Starting inference...")
    start_time = time.time()

    for idx, question_data in enumerate(tqdm(questions, desc="Processing")):
        result = process_single_question(
            question_data=question_data,
            model=model,
            tokenizer=tokenizer,
            video_cache=video_cache,
            base_path=args.data_path,
            num_frames_yolo=args.num_frames_yolo,
            num_frames_normal=args.num_frames_normal,
            max_num=args.max_num,
            use_yolo=not args.no_yolo,
            yolo_model_path=args.yolo_model,
            device=device,
            ocr_extractor=ocr_extractor,
            ocr_confidence=args.ocr_confidence
        )
        results.append(result)

        # Print progress every 10 questions
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(questions) - idx - 1)
            print(f"\n  Progress: {idx+1}/{len(questions)} | "
                  f"Avg: {avg_time:.2f}s/q | "
                  f"ETA: {remaining/60:.1f}min")

    # Save results
    save_results(results, args.output_dir, args.model)

    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total questions processed: {len(results)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per question: {total_time/len(results):.2f} seconds")
    print(f"Unique videos processed: {len(video_cache)}")
    print("="*80)


if __name__ == '__main__':
    main()
