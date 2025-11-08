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
Phân tích: Đọc biển - Biển tròn màu đỏ là biển cấm. Hình xe máy bị gạch chéo nghĩa là cấm xe máy đi qua.
Đáp án: A
""",

    'road_name': """
Ví dụ về Tên Đường (Biển Chỉ Đường):
Câu hỏi: Biển chỉ dẫn màu xanh có mũi tên trái ghi "Đường Lê Lợi". Muốn đi Đường Lê Lợi thì phải rẽ hướng nào?
A. Đi thẳng
B. Rẽ trái
C. Rẽ phải
D. Quay đầu
Phân tích: ĐỌC NỘI DUNG BIỂN - Biển xanh ghi "Đường Lê Lợi" với mũi tên trái → phải rẽ trái để vào Đường Lê Lợi.
Đáp án: B
""",

    'street_name_direction': """
Ví dụ về Đọc Tên Đường Từ Biển Chỉ Đường:
Câu hỏi: Theo biển báo trong video, muốn đi đến đường Lương Định Của thì phải đi hướng nào?
A. Đi thẳng
B. Rẽ trái
C. Rẽ phải
D. Quay đầu
Phân tích: Bước 1: ĐỌC biển chỉ đường màu xanh/trắng để tìm "Lương Định Của". Bước 2: Xem mũi tên trên biển chỉ hướng nào (trái/phải/thẳng). Bước 3: Theo hướng mũi tên đó.
Đáp án: [Tùy theo mũi tên trên biển]
""",

    'prohibition_text': """
Ví dụ về Đọc Nội Dung Biển Cấm:
Câu hỏi: Biển báo màu đỏ ghi "Cấm dừng và đỗ xe" có nghĩa là gì?
A. Được dừng nhưng không được đỗ
B. Được đỗ nhưng không được dừng
C. Không được cả dừng và đỗ
D. Chỉ được dừng tạm thời
Phân tích: ĐỌC KỸ NỘI DUNG - Biển ghi RÕ "Cấm dừng VÀ đỗ xe" nghĩa là không được phép cả hai hành động dừng và đỗ.
Đáp án: C
""",

    'warning_text': """
Ví dụ về Đọc Biển Báo Hiệu:
Câu hỏi: Biển tam giác màu vàng có ghi chữ "Đường người đi bộ cắt ngang" cảnh báo điều gì?
A. Có đường dành cho người đi bộ
B. Cấm người đi bộ
C. Sắp có vạch sang đường
D. Ưu tiên người đi bộ
Phân tích: ĐỌC BIỂN CẢNH BÁO - Biển tam giác vàng ghi "Đường người đi bộ cắt ngang" → cảnh báo phía trước có vạch sang đường cho người đi bộ, cần giảm tốc độ.
Đáp án: C
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
Phân tích: ĐỌC BIỂN - "Chỉ dành cho xe ô tô" nghĩa là chỉ xe ô tô mới được phép, xe máy không được đi.
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

    # Priority 1: Street name and direction questions (most important for text reading)
    if any(kw in question_lower for kw in ['tên đường', 'đến đường', 'đi đường', 'vào đường', 'muốn đi']):
        examples.append(FEW_SHOT_EXAMPLES['street_name_direction'])
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['road_name'])

    # Priority 2: Prohibition signs with text
    elif any(kw in question_lower for kw in ['cấm dừng', 'cấm đỗ', 'cấm', 'biển đỏ']):
        examples.append(FEW_SHOT_EXAMPLES['prohibition_text'])
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['traffic_sign'])

    # Priority 3: Warning signs with text
    elif any(kw in question_lower for kw in ['cảnh báo', 'tam giác', 'biển vàng', 'người đi bộ']):
        examples.append(FEW_SHOT_EXAMPLES['warning_text'])

    # Priority 4: General traffic signs
    elif any(kw in question_lower for kw in ['biển báo', 'biển hiệu', 'ý nghĩa', 'loại biển']):
        examples.append(FEW_SHOT_EXAMPLES['traffic_sign'])

    # Priority 5: Direction questions
    if any(kw in question_lower for kw in ['rẽ trái', 'rẽ phải', 'quay đầu', 'đi thẳng', 'hướng']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['direction'])

    # Priority 6: Yes/no questions (typically 2 choices)
    if any(kw in question_lower for kw in ['có được', 'có thể', 'được phép', 'được không', 'đúng hay sai']):
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

    # Section 2: Detection context (if available) - emphasize OCR text
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

                    # Make OCR text more prominent by putting it first if available
                    if ocr and conf >= 0.6:
                        line = f"  • Khung {i+1}: {sign} ({loc}) - NỘI DUNG: \"{ocr}\""
                    else:
                        line = f"  • Khung {i+1}: {sign} ({loc})"
                        if ocr:
                            # OCR text exists but confidence is too low
                            print(f"  [Prompt] ⚠ Skipping low-confidence OCR for '{sign}': '{ocr}' (conf: {conf:.2f} < 0.6)")
                    detection_lines.append(line)

        if detection_lines:
            detection_section = f"\nTHÔNG TIN BIỂN BÁO ĐÃ PHÁT HIỆN (ưu tiên đọc nội dung chữ):\n" + "\n".join(detection_lines) + "\n"

    # Main prompt with structured reasoning - TEXT FIRST
    prompt = f"""BẠN LÀ CHUYÊN GIA AN TOÀN GIAO THÔNG VIỆT NAM.

QUAN TRỌNG - ƯU TIÊN ĐỌC NỘI DUNG CHỮ:
• Biển chỉ đường (xanh/trắng): ĐỌC KỸ tên đường, hướng đi, khoảng cách
• Biển cấm (đỏ): ĐỌC nội dung cấm chỉ cụ thể (cấm rẽ, cấm dừng, cấm xe...)
• Biển báo hiệu (tam giác): ĐỌC nội dung cảnh báo
• Văn bản trên biển là NGUỒN THÔNG TIN CHÍNH XÁC NHẤT

Nhiệm vụ: Phân tích video camera hành trình và trả lời câu hỏi dựa trên:
- Nội dung CHỮ trên các biển báo (quan trọng nhất!)
- Hình dạng, màu sắc, ký hiệu của biển
- Đèn tín hiệu (màu sắc, trạng thái)
- Vạch kẻ đường và mũi tên
- Phương tiện giao thông
- Môi trường (thời tiết, ánh sáng, vị trí)
{few_shot_section}{detection_section}
BÀI TOÁN CẦN GIẢI:

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Hãy phân tích theo các bước (ƯU TIÊN ĐỌC CHỮ TRƯỚC):
1. ĐỌC KỸ nội dung văn bản trên các biển báo (tên đường, nội dung cấm, cảnh báo...)
2. Quan sát hình dạng, màu sắc, mũi tên chỉ hướng trên biển
3. Kết hợp với các yếu tố khác trong video (đèn, vạch, xe...)
4. Áp dụng luật giao thông Việt Nam và chọn đáp án chính xác

Trả lời:
Phân tích: [Nêu rõ nội dung đã đọc được từ biển, sau đó giải thích]
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
