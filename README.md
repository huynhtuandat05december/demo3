# Traffic Video Question Answering - Demo3 (Quick Win Package)

InternVL3-8B based video question answering system with intelligent YOLO-based frame selection, **OCR text extraction**, **few-shot prompting**, and optimized inference for Vietnamese traffic safety questions.

**Expected Accuracy: 70-80%** (up from 48.89% baseline)

## ✨ New Features (Quick Win Package)

- **🔤 OCR Text Extraction** (NEW! - Highest Impact)
  - Extracts Vietnamese text from traffic signs using PaddleOCR
  - Adds extracted text to model context
  - Enable with `--use_ocr` flag
  - Expected improvement: +15-20% accuracy

- **📚 Few-Shot Prompting** (NEW!)
  - Automatically selects relevant examples based on question type
  - Guides model with 1-2 solved examples
  - Structured reasoning framework
  - Expected improvement: +5-10% accuracy

- **🎯 Adaptive Frame Selection** (NEW!)
  - Adjusts YOLO/uniform frame ratio based on question keywords
  - Sign questions → more YOLO frames
  - Temporal questions → more uniform frames
  - Expected improvement: +3-6% accuracy

- **💬 Increased max_new_tokens** (CRITICAL FIX!)
  - Changed from 10 → 256 tokens
  - Allows full chain-of-thought reasoning
  - Expected improvement: +5-10% accuracy

### Core Features (From Demo2)

- **Fixed 2x1 Grid Preprocessing**: Optimized for 16:9 videos (98% of dataset)
  - Resizes frames to 896x448 (preserves aspect ratio)
  - Splits into 2×448x448 patches for high-resolution text
  - No more dynamic preprocessing complexity
- **Custom Vietnamese Traffic Sign Detection**: Uses trained YOLO model with 52 Vietnamese traffic sign classes
- **Hybrid YOLO + Uniform Sampling**: Combines intelligent frame selection with comprehensive coverage
  - YOLO frames: Frames with detected traffic signs
  - Uniform frames: Evenly sampled across video for context
  - Both use 2x1 grid preprocessing
- **Detection-Aware Prompting**: Passes detected sign names to InternVL3 as additional context
- **Efficient Caching**: Caches video frames to avoid redundant processing (multiple questions per video)
- **Multi-GPU Support**: Automatic distribution across multiple GPUs if available
- **8-bit Quantization**: Optional 8-bit mode for reduced memory usage (~10-12GB vs 16-20GB)

## Installation

```bash
cd demo3

# Install core dependencies (if not already installed)
pip install torch transformers torchvision pillow opencv-python tqdm ultralytics

# Install OCR for text extraction (REQUIRED for best accuracy)
pip install paddleocr

# Alternative OCR (if PaddleOCR fails)
pip install easyocr

# Optional but recommended for faster inference:
pip install flash-attn --no-build-isolation
```

**Note:** OCR is optional but strongly recommended. Without OCR (`--use_ocr`), expected accuracy is ~60-65%. With OCR, expected accuracy is **70-80%**.

## Usage

### Quick Start (Recommended - With OCR)

**For best accuracy (70-80%), use OCR:**

```bash
python inference_traffic.py \
    --yolo_model /path/to/your/trained_model.pt \
    --use_ocr \
    --ocr_confidence 0.6
```

### Basic Usage (Without OCR)

**Expected accuracy: ~60-65%**

```bash
python inference_traffic.py --yolo_model /path/to/your/trained_model.pt
```

### Common Options

**Test on small sample first (RECOMMENDED):**
```bash
python inference_traffic.py \
    --yolo_model /path/to/model.pt \
    --use_ocr \
    --samples 10
```

**With 8-bit quantization** (saves GPU memory):
```bash
python inference_traffic.py \
    --yolo_model /path/to/model.pt \
    --use_ocr \
    --load_in_8bit
```

**Adjust OCR confidence threshold:**
```bash
python inference_traffic.py \
    --yolo_model /path/to/model.pt \
    --use_ocr \
    --ocr_confidence 0.7  # Higher = more strict (default: 0.6)
```

**Adjust frame settings** (separate control for YOLO and uniform frames):
```bash
# More YOLO frames, fewer uniform frames (focus on detected signs)
python inference_traffic.py --yolo_model /path/to/model.pt --num_frames_yolo 12 --num_frames_normal 4

# Balanced approach (default)
python inference_traffic.py --yolo_model /path/to/model.pt --num_frames_yolo 8 --num_frames_normal 8

# More uniform frames, fewer YOLO frames (broader context)
python inference_traffic.py --yolo_model /path/to/model.pt --num_frames_yolo 4 --num_frames_normal 12
```

**Disable YOLO** (use uniform sampling only, no detection context):
```bash
python inference_traffic.py --no_yolo
```

**Custom data path**:
```bash
python inference_traffic.py --yolo_model /path/to/model.pt --data_path /path/to/public_test
```

### Full Command Example

```bash
python inference_traffic.py \
    --model OpenGVLab/InternVL3-8B \
    --yolo_model /path/to/your/trained_model.pt \
    --load_in_8bit \
    --num_frames_yolo 8 \
    --num_frames_normal 8 \
    --samples 50 \
    --output_dir ./results
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| **Model** | | |
| `--model` | `OpenGVLab/InternVL3-8B` | Model name or local path |
| `--load_in_8bit` | False | Enable 8-bit quantization |
| **Data** | | |
| `--data_path` | `../RoadBuddy/traffic_buddy_train+public_test` | Path to test data |
| `--samples` | None (all) | Number of samples to process |
| **Frame Selection** | | |
| `--yolo_model` | None | **Path to trained YOLO model (.pt file)** |
| `--num_frames_yolo` | 8 | Frames to extract using YOLO selection (adaptive) |
| `--num_frames_normal` | 8 | Frames to extract using uniform sampling (adaptive) |
| `--no_yolo` | False | Disable YOLO frame selection |
| **OCR (NEW!)** | | |
| `--use_ocr` | False | **Enable OCR text extraction (recommended!)** |
| `--ocr_confidence` | 0.6 | Minimum OCR confidence threshold (0.0-1.0) |
| **Output** | | |
| `--output_dir` | `./output` | Output directory for results |

**Note:** `--num_frames_yolo` and `--num_frames_normal` are automatically adjusted by the adaptive frame selection based on question type. The values you provide are used as defaults.

## Output Format

The script generates two CSV files:

### 1. Full Results (`submission_<model>_<timestamp>.csv`)
Contains all fields for debugging:
```csv
id,answer,raw_response,prompt
testa_0001,A,"",Frame1: <image>...
testa_0002,B," ",Frame1: <image>...
```

### 2. Minimal Submission (`submission_minimal_<model>_<timestamp>.csv`)
Only required fields for submission:
```csv
id,answer
testa_0001,A
testa_0002,B
testa_0003,D
```

## Architecture

```
demo3/
├── inference_traffic.py      # Main inference script
├── frame_selector.py          # YOLO-based intelligent frame selection
├── model_utils.py             # InternVL3 utilities (video loading, preprocessing)
├── prompt_template.py         # Vietnamese prompt engineering
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Frame Selection & Preprocessing Strategy

### New 2x1 Grid Preprocessing (Optimized for 16:9 Videos)

All frames (both YOLO and uniform) now use fixed 2x1 grid preprocessing:

1. **Resize to 896x448**:
   - Preserves 16:9 aspect ratio (98% of dataset)
   - 2:1 ratio perfect for splitting into 2 patches
   - Maintains high resolution for text (vs squashing to 448x448)

2. **Split into 2 patches**:
   - Left patch: 0-448px → 448x448
   - Right patch: 448-896px → 448x448
   - Always 2 patches per frame (predictable, no dynamic complexity)

3. **Benefits**:
   - **High-res text preservation**: Road signs, street names stay sharp
   - **OCR-ready**: PIL images at 896x448 available for future OCR
   - **Consistent**: No variable patch counts, always 2 per frame
   - **Fast**: Simpler than dynamic preprocessing

### Hybrid YOLO + Uniform Strategy (Default)

**Method 1: YOLO Frame Selection (with 2x1 grid)**

1. **Intelligent Detection**:
   - Scan all frames with trained YOLO model
   - Filter blurry frames (Laplacian variance < 100)
   - Detect Vietnamese traffic signs (52 classes)
   - Rank by total sign area (sum of all detected signs)

2. **Select Top Frames**:
   - Choose `num_frames_yolo` frames with most/largest signs
   - Apply 2x1 grid preprocessing (896x448 → 2×448x448)
   - Extract detection metadata for context

3. **Detection Context**:
   - Pass detected sign names to InternVL3 in prompt
   - Helps model focus on relevant traffic information

**Method 2: Uniform Sampling (with 2x1 grid)**

1. **Even Distribution**:
   - Sample `num_frames_normal` frames evenly across video
   - Provides general context beyond detected signs
   - Apply 2x1 grid preprocessing (896x448 → 2×448x448)

2. **Compensation**:
   - If YOLO fails: `num_frames_normal` += `num_frames_yolo`
   - Ensures consistent total frame count

**Combined Result**:
- Total frames: `num_frames_yolo` + `num_frames_normal`
- Total patches: (Total frames) × 2
- Example: 8 YOLO + 8 uniform = 16 frames = 32 patches

### Vietnamese Traffic Sign Classes (52 total)

The YOLO model detects all Vietnamese traffic signs including:
- Prohibitory signs (Cấm rẽ trái, Cấm quay đầu, Cấm dừng đỗ, etc.)
- Warning signs (Nguy hiểm khác, Chỗ ngoặt nguy hiểm, Gồ giảm tốc, etc.)
- Regulatory signs (Giới hạn tốc độ 40/50/60/80km/h, etc.)
- Directional signs (Đường một chiều, Chỉ được rẽ trái, etc.)
- Informational signs (Bến xe buýt, Camera giám sát, etc.)

### Pure Uniform Sampling (No YOLO)

Use `--no_yolo` flag to disable YOLO and use only uniform sampling:
- Evenly distributed frames across video duration
- All frames use 2x1 grid preprocessing
- Simple and reliable, no detection context
- Useful when YOLO model not available or for baseline comparison

## Performance

### Speed (with 2x1 Grid Preprocessing)
- **16 total frames** (8 YOLO + 8 uniform = 32 patches): ~2-3 minutes per video (balanced)
- **24 total frames** (12 YOLO + 12 uniform = 48 patches): ~4-6 minutes per video (high quality)
- **10 total frames** (5 YOLO + 5 uniform = 20 patches): ~1-2 minutes per video (fast)

*Note: 2x1 grid always creates 2 patches per frame, regardless of settings*

### Memory Requirements
- **Full precision**: 16-20GB VRAM
- **8-bit quantization**: 10-12GB VRAM
- **With YOLO**: +2-3GB VRAM
- **2x1 Grid overhead**: Minimal (same as before)

### Expected Runtime (405 questions, 182 videos)
- **Fast mode** (10 total frames): ~2-3 hours
- **Balanced mode** (16 total frames): ~3-5 hours
- **High quality** (24 total frames): ~5-8 hours

*Note: Video caching significantly reduces time since multiple questions share videos*

## Troubleshooting

### CUDA Out of Memory
```bash
# Use 8-bit quantization
python inference_traffic.py --yolo_model /path/to/model.pt --load_in_8bit

# Or reduce total frames (YOLO + uniform)
python inference_traffic.py --yolo_model /path/to/model.pt --num_frames_yolo 4 --num_frames_normal 4
```

### YOLO Model Not Provided
If you run without `--yolo_model`, the system will warn and fall back to uniform sampling:
```
[WARNING] YOLO enabled but no model path provided. Will use uniform sampling.
```

To use YOLO features, you must provide your trained model:
```bash
python inference_traffic.py --yolo_model /path/to/your_model.pt
```

### Video Not Found Errors
Check that your data path is correct:
```bash
ls ../RoadBuddy/traffic_buddy_train+public_test/public_test/videos/
```

## Testing Individual Components

### Test Frame Selector with Your YOLO Model
```bash
cd demo3
python frame_selector.py /path/to/your_model.pt
```

This will:
- Process a sample video
- Show detected signs per frame
- Save best frames as images for verification
- Display processing statistics

### Test Model Loading
```python
from model_utils import load_video
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True)
```

### Verify Detection Context
Run a small test to see detection context in action:
```bash
python inference_traffic.py \
    --yolo_model /path/to/model.pt \
    --samples 1 \
    --output_dir ./test_output
```

Check the output CSV to see the `prompt` field - it should include detected sign names.

## Comparison with Demo2

| Feature | Demo2 | Demo3 |
|---------|-------|-------|
| Frame Selection | Uniform / CLIP | Custom YOLO (52 Vietnamese signs) |
| Detection Context | Grounding DINO (optional) | YOLO with sign names in prompt |
| Sign Classes | Generic (COCO) | 52 Vietnamese traffic signs |
| Ranking Strategy | Largest single object | Total sign area (sum of all) |
| Context Integration | Bounding boxes | Sign names + visual context |
| Memory Usage | Higher (with GDINO) | Lower (YOLO only) |
| Speed | Slower (GDINO overhead) | Faster |
| Accuracy | Generic object detection | Domain-specific (Vietnamese traffic) |

Demo3 is specifically optimized for Vietnamese traffic signs with:
- Custom-trained YOLO for 52 sign classes
- Detection-aware prompting with sign names
- Total area ranking (prioritizes frames with multiple signs)
- More relevant context for traffic safety questions

## License

This project uses:
- InternVL3 (MIT License)
- YOLO11 (AGPL-3.0 License)
- See respective repositories for details

## Acknowledgments

- InternVL3: https://github.com/OpenGVLab/InternVL
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics

## Frame Strategy Tracking

The output CSV now includes two additional columns to track frame selection performance:

### Additional Columns

1. **`frame_strategy`** - Which preprocessing strategy was used:
   - `2x1_grid_yolo_and_uniform`: Both YOLO and uniform frames combined (most common)
   - `2x1_grid_yolo_only`: Only YOLO frames (when num_frames_normal=0)
   - `2x1_grid_uniform_only`: Only uniform frames (YOLO failed or --no_yolo flag)
   - `error`: Video loading/processing error

2. **`num_detections`** - Total number of traffic signs detected across all YOLO frames
   - For strategies with YOLO: Sum of all signs detected in YOLO-selected frames
   - For `2x1_grid_uniform_only` or `error`: Always 0

### Strategy Statistics

After processing, the script automatically prints statistics:

```
[Statistics] Frame Selection Strategies:
  2x1_grid_yolo_and_uniform: 350 (86.4%)
  2x1_grid_uniform_only: 55 (13.6%)
  Average detections per YOLO frame: 4.2
```

### New 2x1 Grid Benefits Tracking

All strategies now use 2x1 grid preprocessing:
- **Consistent patches**: Always 2 patches per frame
- **Total patches**: (num_frames_yolo + num_frames_normal) × 2
- **High-res text**: All frames at 896x448 before splitting
- **OCR-ready**: PIL images available in cache for future enhancement
