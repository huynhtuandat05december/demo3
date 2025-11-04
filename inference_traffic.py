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

from model_utils import load_video, load_video_from_indices, load_video_from_indices_with_context, split_model
from prompt_template import create_traffic_prompt, create_traffic_prompt_with_context, format_video_prefix_with_detections
from frame_selector import find_best_frames_with_context

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
    Create video frame prefix for prompt.

    Args:
        num_frames: Number of frames in the video

    Returns:
        Formatted prefix string
    """
    return ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])


def process_single_question(
    question_data: Dict,
    model,
    tokenizer,
    video_cache: Dict,
    base_path: str,
    num_frames: int,
    max_num: int,
    use_yolo: bool = True,
    yolo_model_path: Optional[str] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Process a single question with detection-aware processing.

    Args:
        question_data: Dictionary containing question info
        model: Loaded InternVL3 model
        tokenizer: Loaded tokenizer
        video_cache: Cache for video frames
        base_path: Base path to data directory
        num_frames: Number of frames to extract
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
            frame_indices = None
            detections_dict = None

            # Try YOLO-based frame selection with detection context
            if use_yolo and yolo_model_path:
                frame_indices, detections_dict = find_best_frames_with_context(
                    full_video_path,
                    yolo_model_path,
                    top_k=num_frames,
                    device=device
                )

            # Load frames
            if frame_indices is not None and len(frame_indices) > 0:
                print(f"  [Video] Using YOLO-selected frames: {frame_indices}")
                pixel_values, num_patches_list = load_video_from_indices(
                    full_video_path,
                    frame_indices,
                    input_size=448,
                    max_num=max_num
                )
            else:
                # Fallback to uniform sampling
                print(f"  [Video] Using uniform sampling ({num_frames} frames)")
                pixel_values, num_patches_list = load_video(
                    full_video_path,
                    num_segments=num_frames,
                    input_size=448,
                    max_num=max_num
                )
                frame_indices = None
                detections_dict = None

            # Determine strategy used
            if frame_indices is not None and detections_dict is not None:
                strategy = 'yolo_detection'
            else:
                strategy = 'uniform_sampling'

            # Cache on CPU
            video_cache[video_path] = {
                'pixel_values': pixel_values.cpu(),
                'num_patches_list': num_patches_list,
                'num_frames': len(num_patches_list),
                'frame_indices': frame_indices,
                'detections_dict': detections_dict,
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
    frame_indices = cached_data.get('frame_indices')
    detections_dict = cached_data.get('detections_dict')
    strategy = cached_data.get('strategy', 'uniform_sampling')

    # Create prompt with detection context if available
    if detections_dict and frame_indices:
        video_prefix = format_video_prefix_with_detections(
            actual_num_frames,
            detections_dict,
            frame_indices
        )
        prompt_text = create_traffic_prompt_with_context(
            question_text,
            choices,
            detections_dict,
            frame_indices
        )
    else:
        video_prefix = create_video_prefix(actual_num_frames)
        prompt_text = create_traffic_prompt(question_text, choices)

    full_question = video_prefix + prompt_text

    # Generate response
    try:
        generation_config = {
            'max_new_tokens': 10,
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
    parser.add_argument('--num_frames', type=int, default=8,
                        help='Number of frames to extract per video')
    parser.add_argument('--max_num', type=int, default=3,
                        help='Maximum number of patches per frame')
    parser.add_argument('--yolo_model', type=str, default=None,
                        help='Path to trained YOLO model (.pt file) for Vietnamese traffic signs')
    parser.add_argument('--no_yolo', action='store_true',
                        help='Disable YOLO frame selection, use uniform sampling only')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save results')

    args = parser.parse_args()

    print("="*80)
    print("InternVL3 Traffic Video Question Answering")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"8-bit quantization: {args.load_in_8bit}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Max patches per frame: {args.max_num}")
    print(f"YOLO frame selection: {not args.no_yolo}")
    if not args.no_yolo:
        if args.yolo_model:
            print(f"YOLO model: {args.yolo_model}")
        else:
            print("[WARNING] YOLO enabled but no model path provided. Will use uniform sampling.")
    print("="*80)

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("[WARNING] CUDA not available, using CPU. This will be very slow!")

    # Load model
    model, tokenizer = load_model(args.model, args.load_in_8bit, device)

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
            num_frames=args.num_frames,
            max_num=args.max_num,
            use_yolo=not args.no_yolo,
            yolo_model_path=args.yolo_model,
            device=device
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
