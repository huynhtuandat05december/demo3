"""
Enhanced prompt templates with few-shot examples for Vietnamese traffic safety QA.
"""

from typing import List, Dict, Optional

# Few-shot examples for different question types
FEW_SHOT_EXAMPLES = {
    'traffic_sign': """
Ví dụ về Biển Báo:
Câu hỏi: Biển báo tròn màu đỏ có hình xe máy bị gạch chéo có ý nghĩa gì?
A. Cấm xe máy
B. Cấm ô tô
C. Hết cấm xe máy
D. Đường dành cho xe máy
Phân tích: Biển tròn màu đỏ là biển cấm. Hình xe máy bị gạch chéo nghĩa là cấm xe máy đi qua.
Đáp án: A
""",

    'road_name': """
Ví dụ về Tên Đường:
Câu hỏi: Biển chỉ dẫn màu xanh có mũi tên trái ghi "Đường Lê Lợi". Muốn đi Đường Lê Lợi thì phải rẽ hướng nào?
A. Đi thẳng
B. Rẽ trái
C. Rẽ phải
D. Quay đầu
Phân tích: Biển xanh chỉ dẫn với mũi tên trái chỉ rằng phải rẽ trái để vào Đường Lê Lợi.
Đáp án: B
""",

    'direction': """
Ví dụ về Hướng Đi:
Câu hỏi: Tại ngã tư có biển cấm rẽ trái, xe có được rẽ trái không?
A. Có
B. Không
Phân tích: Biển cấm rẽ trái có nghĩa là phương tiện không được phép rẽ trái tại vị trí này.
Đáp án: B
""",

    'yes_no': """
Ví dụ về Câu Hỏi Đúng/Sai:
Câu hỏi: Xe đang di chuyển trên làn đường có biển "Chỉ dành cho xe ô tô". Xe máy có được đi trên làn này không?
A. Có
B. Không
Phân tích: Biển "Chỉ dành cho xe ô tô" nghĩa là chỉ xe ô tô mới được phép, xe máy không được đi.
Đáp án: B
"""
}


def select_relevant_examples(question: str, max_examples: int = 2) -> str:
    """
    Select most relevant few-shot examples based on question keywords.

    Args:
        question: The question text
        max_examples: Maximum number of examples to include

    Returns:
        String containing selected examples
    """
    question_lower = question.lower()
    examples = []

    # Check question type and add relevant examples
    if any(kw in question_lower for kw in ['biển báo', 'biển hiệu', 'ý nghĩa', 'loại biển']):
        examples.append(FEW_SHOT_EXAMPLES['traffic_sign'])

    if any(kw in question_lower for kw in ['tên đường', 'đến đường', 'đi đường', 'vào đường']):
        examples.append(FEW_SHOT_EXAMPLES['road_name'])

    if any(kw in question_lower for kw in ['rẽ trái', 'rẽ phải', 'quay đầu', 'đi thẳng', 'hướng']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['direction'])

    # For yes/no questions (typically 2 choices)
    if any(kw in question_lower for kw in ['có được', 'có thể', 'được phép', 'được không']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['yes_no'])

    # Limit to max_examples
    examples = examples[:max_examples]

    if examples:
        return "\n---\n".join(examples) + "\n---\n"
    return ""


def create_enhanced_prompt_with_few_shot(
    question: str,
    choices: List[str],
    detections_dict: Optional[Dict] = None,
    frame_indices: Optional[List[int]] = None,
    num_choices: int = 4
) -> str:
    """
    Create an enhanced prompt with few-shot examples and better structure.

    Args:
        question: The question text
        choices: List of answer choices
        detections_dict: Optional detection information
        frame_indices: Optional frame indices
        num_choices: Number of choices (2 or 4)

    Returns:
        Enhanced prompt string
    """
    choices_text = "\n".join(choices)

    # Section 1: Few-shot examples (max 2 to not exceed token limit)
    few_shot_section = ""
    examples = select_relevant_examples(question, max_examples=1)  # Reduced to 1 to save tokens
    if examples:
        few_shot_section = f"\nCÁC VÍ DỤ THAM KHẢO:\n{examples}\n"

    # Section 2: Detection context (if available)
    detection_section = ""
    if detections_dict and frame_indices:
        detection_lines = []
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in detections_dict and detections_dict[frame_idx]:
                for det in detections_dict[frame_idx]:
                    sign = det['sign_name']
                    loc = det['location']
                    ocr = det.get('ocr_text', '')
                    conf = det.get('ocr_confidence', 0.0)

                    line = f"  • Khung {i+1}: {sign} ({loc})"
                    if ocr and conf >= 0.6:
                        line += f' - Nội dung: "{ocr}"'
                    elif ocr:
                        # OCR text exists but confidence is too low
                        print(f"  [Prompt] ⚠ Skipping low-confidence OCR for '{sign}': '{ocr}' (conf: {conf:.2f} < 0.6)")
                    detection_lines.append(line)

        if detection_lines:
            detection_section = f"\nTHÔNG TIN BIỂN BÁO ĐÃ PHÁT HIỆN:\n" + "\n".join(detection_lines) + "\n"

    # Main prompt with structured reasoning
    prompt = f"""BẠN LÀ CHUYÊN GIA AN TOÀN GIAO THÔNG VIỆT NAM.
Nhiệm vụ: Phân tích video camera hành trình và trả lời câu hỏi dựa trên:
- Biển báo giao thông (hình dạng, màu sắc, ký hiệu, nội dung chữ)
- Đèn tín hiệu (màu sắc, trạng thái)
- Vạch kẻ đường và mũi tên
- Phương tiện giao thông
- Môi trường (thời tiết, ánh sáng, vị trí)
{few_shot_section}{detection_section}
BÀI TOÁN CẦN GIẢI:

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Hãy phân tích theo các bước:
1. Quan sát các yếu tố trong video
2. Xác định loại biển báo hoặc tình huống
3. Áp dụng luật giao thông Việt Nam
4. Chọn đáp án chính xác

Trả lời:
Phân tích: [Giải thích ngắn gọn]
Đáp án: [Chữ cái A/B/C/D]"""

    return prompt


# Adaptive frame selection helper
def get_optimal_frame_counts(question: str, default_yolo: int = 8, default_uniform: int = 8) -> tuple:
    """
    Determine optimal frame counts based on question type.

    Args:
        question: The question text
        default_yolo: Default YOLO frame count
        default_uniform: Default uniform frame count

    Returns:
        (num_yolo_frames, num_uniform_frames)
    """
    question_lower = question.lower()

    # Sign-heavy questions: more YOLO frames
    if any(kw in question_lower for kw in ['biển báo', 'biển hiệu', 'loại biển', 'ý nghĩa']):
        return (10, 6)  # More YOLO, less uniform

    # Direction/flow questions: more temporal coverage
    elif any(kw in question_lower for kw in ['trước khi', 'sau khi', 'khi nào', 'lúc nào']):
        return (6, 10)  # Less YOLO, more uniform for temporal coverage

    # Yes/no or simple questions: fewer frames needed
    elif any(kw in question_lower for kw in ['có được', 'có thể', 'được phép']):
        return (6, 6)  # Fewer frames total

    # Default: balanced
    else:
        return (default_yolo, default_uniform)


if __name__ == "__main__":
    # Test the enhanced prompt generation
    test_question = "Biển báo cấm rẽ trái có ý nghĩa gì?"
    test_choices = ["A. Không được rẽ trái", "B. Không được rẽ phải", "C. Không được quay đầu", "D. Hết cấm rẽ trái"]

    prompt = create_enhanced_prompt_with_few_shot(test_question, test_choices)
    print(prompt)
    print("\n" + "="*80 + "\n")

    # Test adaptive frame selection
    frame_counts = get_optimal_frame_counts(test_question)
    print(f"Optimal frame counts for question: YOLO={frame_counts[0]}, Uniform={frame_counts[1]}")
