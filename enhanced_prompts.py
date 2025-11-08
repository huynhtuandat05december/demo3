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
""",

    'lane_arrow': """
Ví dụ về Đọc Mũi Tên Trên Mặt Đường:
Câu hỏi: Xe đang ở làn giữa có mũi tên ↑←. Xe có được rẽ trái không?
A. Có, được rẽ trái
B. Không, chỉ được đi thẳng
C. Không, chỉ được rẽ phải
D. Phải đi thẳng hoặc rẽ phải
Phân tích: Bước 1: ĐỌC mũi tên trên làn giữa - có 2 mũi tên ↑← (thẳng và trái). Bước 2: Mũi tên kết hợp nghĩa là xe được chọn ĐI THẲNG HOẶC RẼ TRÁI. Bước 3: Xe có quyền rẽ trái.
Đáp án: A
""",

    'spatial_positioning': """
Ví dụ về Xác Định Vị Trí Làn Đường:
Câu hỏi: Đường có 3 làn cùng chiều. Xe đang ở làn bên phải cùng. Muốn rẽ trái thì phải làm gì?
A. Rẽ trái ngay
B. Chuyển sang làn giữa rồi rẽ trái
C. Chuyển sang làn trái rồi rẽ trái
D. Không được rẽ trái
Phân tích: Bước 1: Đếm làn - có 3 làn (trái, giữa, phải). Bước 2: Xe ở làn phải. Bước 3: Muốn rẽ trái phải ở làn trái nhất. Bước 4: Phải chuyển 2 làn sang trái trước khi rẽ.
Đáp án: C
""",

    'vehicle_interaction': """
Ví dụ về Phân Tích Phương Tiện:
Câu hỏi: Tại ngã tư, xe ô tô từ bên phải đang tiến vào và xe máy từ phía trước. Ai có quyền ưu tiên?
A. Xe ô tô (từ bên phải)
B. Xe máy (từ phía trước)
C. Ai đến trước đi trước
D. Xe lớn ưu tiên
Phân tích: Bước 1: Quan sát vị trí - ô tô bên phải, xe máy phía trước. Bước 2: Áp dụng luật giao thông VN - nhường đường cho xe bên PHẢI. Bước 3: Xe ô tô có quyền ưu tiên.
Đáp án: A
""",

    'temporal_sequence': """
Ví dụ về Phân Tích Thời Gian:
Câu hỏi: Trong video, đèn tín hiệu chuyển từ xanh sang vàng. Xe nên làm gì?
A. Tăng tốc để qua giao lộ
B. Dừng lại trước vạch dừng
C. Tiếp tục đi với tốc độ bình thường
D. Rẽ phải để tránh đèn đỏ
Phân tích: Bước 1: Quan sát thay đổi - đèn XAnh → VÀNG (sắp đổi đỏ). Bước 2: Luật giao thông - đèn vàng là tín hiệu CẨN TRỌNG, chuẩn bị DỪNG. Bước 3: Hành động đúng là dừng trước vạch.
Đáp án: B
""",

    'lane_sign_mapping': """
Ví dụ về Quan Hệ Làn Đường - Biển Báo:
Câu hỏi: Có 3 làn đường. Biển "Đường Lê Lợi" ở phía trên làn trái, biển "Cầu Rạch Chiếc" ở phía trên làn phải. Xe đang ở làn trái. Muốn đi Cầu Rạch Chiếc thì phải làm gì?
A. Đi thẳng (đang ở làn đúng)
B. Chuyển sang làn phải
C. Chuyển sang làn giữa
D. Không thể đi Cầu Rạch Chiếc
Phân tích: Bước 1: XÁC ĐỊNH vị trí xe - đang ở LÀN TRÁI. Bước 2: ĐỌC biển ở trên làn trái - "Đường Lê Lợi" → làn trái đi vào Đường Lê Lợi. Bước 3: ĐỌC biển ở trên làn phải - "Cầu Rạch Chiếc" → làn phải đi vào Cầu Rạch Chiếc. Bước 4: Muốn đi Cầu Rạch Chiếc → PHẢI CHUYỂN sang làn phải.
Đáp án: B
""",

    'multi_lane_signs': """
Ví dụ về Đọc Biển Theo Từng Làn:
Câu hỏi: Theo biển báo, xe đang ở làn giữa sẽ đi đến đường nào?
(Giả sử: Biển trên làn trái ghi "Trần Hưng Đạo", biển trên làn giữa ghi "Nguyễn Huệ", biển trên làn phải ghi "Lê Lợi")
A. Trần Hưng Đạo
B. Nguyễn Huệ
C. Lê Lợi
D. Cả ba đường
Phân tích: Bước 1: XÁC ĐỊNH - xe ở LÀN GIỮA. Bước 2: ĐỌC biển ở PHÍA TRÊN làn giữa - ghi "Nguyễn Huệ". Bước 3: NGUYÊN TẮC - biển ở trên làn nào → làn đó đi đến đường trên biển. Bước 4: Xe ở làn giữa → đi vào đường "Nguyễn Huệ".
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

    # Priority 0: Lane-sign mapping (HIGHEST - critical for understanding which lane goes to which street)
    if any(kw in question_lower for kw in ['làn trái', 'làn phải', 'làn giữa', 'đang ở làn', 'chuyển làn']):
        if any(kw in question_lower for kw in ['đường', 'cầu', 'tên đường', 'đến', 'đi vào', 'muốn đi']):
            examples.append(FEW_SHOT_EXAMPLES['lane_sign_mapping'])

    # If question asks about which street based on current lane
    if any(kw in question_lower for kw in ['xe đang ở làn', 'theo biển', 'làn nào']) and any(kw in question_lower for kw in ['đi đến', 'sẽ đi', 'đường nào']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['multi_lane_signs'])

    # Priority 1: Lane arrow questions (NEW - critical for road markings)
    if any(kw in question_lower for kw in ['mũi tên', 'làn', 'lane', 'arrow', 'vạch kẻ']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['lane_arrow'])

    # Priority 2: Spatial positioning questions (NEW - which lane, position)
    if any(kw in question_lower for kw in ['làn nào', 'làn đường', 'vị trí', 'chuyển làn', 'làn trái', 'làn phải', 'làn giữa']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['spatial_positioning'])

    # Priority 3: Vehicle interaction questions (NEW - traffic flow, right-of-way)
    if any(kw in question_lower for kw in ['xe nào', 'phương tiện', 'ưu tiên', 'nhường đường', 'xe ô tô', 'xe máy', 'xe tải']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['vehicle_interaction'])

    # Priority 4: Temporal/signal change questions (NEW - before/after, signal changes)
    if any(kw in question_lower for kw in ['đèn', 'tín hiệu', 'chuyển', 'thay đổi', 'trước khi', 'sau khi', 'xanh', 'đỏ', 'vàng']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['temporal_sequence'])

    # Priority 5: Street name and direction questions (text reading from signs)
    if any(kw in question_lower for kw in ['tên đường', 'đến đường', 'đi đường', 'vào đường', 'muốn đi']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['street_name_direction'])
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['road_name'])

    # Priority 6: Prohibition signs with text
    if any(kw in question_lower for kw in ['cấm dừng', 'cấm đỗ', 'cấm', 'biển đỏ']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['prohibition_text'])
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['traffic_sign'])

    # Priority 7: Warning signs with text
    if any(kw in question_lower for kw in ['cảnh báo', 'tam giác', 'biển vàng', 'người đi bộ']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['warning_text'])

    # Priority 8: General traffic signs
    if any(kw in question_lower for kw in ['biển báo', 'biển hiệu', 'ý nghĩa', 'loại biển']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['traffic_sign'])

    # Priority 9: Direction questions
    if any(kw in question_lower for kw in ['rẽ trái', 'rẽ phải', 'quay đầu', 'đi thẳng', 'hướng']):
        if len(examples) < max_examples:
            examples.append(FEW_SHOT_EXAMPLES['direction'])

    # Priority 10: Yes/no questions (typically 2 choices)
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

                    # Make OCR text more prominent and emphasize lane-sign mapping
                    # Convert location to lane indication
                    lane_info = ""
                    if "trái" in loc or "left" in loc.lower():
                        lane_info = " → LÀN TRÁI"
                    elif "phải" in loc or "right" in loc.lower():
                        lane_info = " → LÀN PHẢI"
                    elif "giữa" in loc or "center" in loc.lower() or "middle" in loc.lower():
                        lane_info = " → LÀN GIỮA"

                    if ocr and conf >= 0.6:
                        line = f"  • Khung {i+1}: {sign} ({loc}{lane_info}) - NỘI DUNG: \"{ocr}\""
                    else:
                        line = f"  • Khung {i+1}: {sign} ({loc}{lane_info})"
                        if ocr:
                            # OCR text exists but confidence is too low
                            print(f"  [Prompt] ⚠ Skipping low-confidence OCR for '{sign}': '{ocr}' (conf: {conf:.2f} < 0.6)")
                    detection_lines.append(line)

        if detection_lines:
            detection_section = f"\nTHÔNG TIN BIỂN BÁO ĐÃ PHÁT HIỆN (ưu tiên đọc nội dung chữ):\n" + "\n".join(detection_lines) + "\n"

    # Main prompt with balanced multi-element emphasis
    prompt = f"""BẠN LÀ CHUYÊN GIA AN TOÀN GIAO THÔNG VIỆT NAM.

Nhiệm vụ: Phân tích TOÀN BỘ video camera hành trình và trả lời câu hỏi dựa trên TẤT CẢ các yếu tố:

1. ĐỌC NỘI DUNG CHỮ trên biển báo:
   • Biển chỉ đường: tên đường, hướng, khoảng cách
   • Biển cấm: nội dung cấm chỉ cụ thể
   • Biển cảnh báo: nội dung cảnh báo

2. ĐỌC MŨI TÊN trên mặt đường:
   • Hướng: thẳng (↑), trái (←), phải (→), quay đầu (↶)
   • Mũi tên kết hợp: (←↑) = trái HOẶC thẳng
   • Xác định mũi tên thuộc làn nào

3. XÁC ĐỊNH VỊ TRÍ LÀN ĐƯỜNG:
   • Xe đang ở làn nào (trái/giữa/phải)?
   • Tổng số làn (đếm từ trái sang)
   • Loại đường: 1 chiều/2 chiều

4. PHÂN TÍCH VẠCH KẺ và PHƯƠNG TIỆN:
   • Vạch liền (không vượt), vạch đứt (được vượt)
   • Vạch sang đường, vạch dừng, vạch vàng
   • Loại xe: ô tô, xe máy, xe tải, xe buýt
   • Vị trí và hành động của xe

5. ĐÈN TÍN HIỆU và THỜI GIAN:
   • Màu đèn: đỏ/vàng/xanh, đèn tròn/đèn mũi tên
   • Thứ tự sự kiện, thay đổi trong video
   • Môi trường: thời tiết, ánh sáng
{few_shot_section}{detection_section}
BÀI TOÁN CẦN GIẢI:

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Hãy phân tích theo 6 BƯỚC (CHÚ Ý QUAN HỆ LÀN-BIỂN):
1. XÁC ĐỊNH xe đang ở làn nào (trái/giữa/phải) - QUAN TRỌNG!
2. XÁC ĐỊNH biển chỉ đường ở phía trên làn nào (trái/giữa/phải)
3. ĐỌC nội dung biển Ở ĐÚNG LÀN (biển trên làn X → làn X đi đường đó)
4. ĐỌC mũi tên trên mặt đường của từng làn
5. QUAN SÁT phương tiện, vạch kẻ, đèn tín hiệu, thay đổi trong video
6. ÁP DỤNG luật giao thông Việt Nam và chọn đáp án chính xác

Trả lời:
Phân tích: [Nêu vị trí làn, biển áp dụng cho làn nào, rồi giải thích]
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
