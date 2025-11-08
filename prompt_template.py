"""
Prompt templates for Vietnamese traffic safety question answering.
"""

from typing import List, Dict, Optional, Any


def create_traffic_prompt(question: str, choices: list[str]) -> str:
    """
    Create a structured prompt for traffic safety questions.

    Args:
        question: The traffic safety question in Vietnamese
        choices: List of answer choices (e.g., ["A. ...", "B. ...", "C. ...", "D. ..."])

    Returns:
        Formatted prompt string
    """
    choices_text = "\n".join(choices)

    prompt = f"""Bạn là một chuyên gia về an toàn giao thông tại Việt Nam. Hãy phân tích TOÀN BỘ video từ camera hành trình và trả lời câu hỏi dựa trên:

1. ĐỌC KỸ NỘI DUNG CHỮ trên biển báo:
   - Biển chỉ đường (xanh/trắng): tên đường, hướng đi, khoảng cách
   - Biển cấm (đỏ): nội dung cấm chỉ (cấm rẽ, cấm dừng, cấm xe...)
   - Biển báo hiệu (tam giác): nội dung cảnh báo
   - Hình dạng, màu sắc, ký hiệu của biển

2. ĐỌC MŨI TÊN trên mặt đường:
   - Hướng mũi tên: thẳng (↑), trái (←), phải (→), quay đầu (↶)
   - Mũi tên kết hợp: ví dụ (←↑) = được rẽ trái HOẶC đi thẳng
   - Xác định mũi tên thuộc làn đường nào

3. XÁC ĐỊNH VỊ TRÍ và LÀN ĐƯỜNG:
   - Xe đang ở làn nào (làn trái/giữa/phải)?
   - Tổng số làn đường (đếm từ trái sang phải)
   - Loại đường: 1 chiều hay 2 chiều, đường phố hay cao tốc
   - Khoảng cách đến giao lộ/biển báo

4. PHÂN TÍCH VẠCH KẺ ĐƯỜNG:
   - Vạch liền trắng: KHÔNG được vượt/chuyển làn
   - Vạch đứt trắng: ĐƯỢC PHÉP vượt/chuyển làn
   - Vạch sang đường (zebra): ưu tiên người đi bộ
   - Vạch dừng (stop line): phải dừng trước vạch
   - Vạch vàng: cấm dừng/đỗ xe

5. QUAN SÁT PHƯƠNG TIỆN:
   - Loại xe: xe máy, ô tô, xe tải, xe buýt, xe đạp, người đi bộ
   - Vị trí: ở làn nào, phía trước/sau/bên cạnh
   - Hành động: đang dừng/di chuyển/rẽ/vượt
   - Mật độ: đông/vắng, tắc nghẽn hay thông thoáng

6. ĐÈN TÍN HIỆU GIAO THÔNG:
   - Màu đèn: đỏ (dừng), vàng (chờ), xanh (đi)
   - Loại: đèn tròn (tất cả hướng) hay đèn mũi tên (hướng cụ thể)
   - Trạng thái và thay đổi của đèn trong video

7. PHÂN TÍCH THỜI GIAN:
   - Thứ tự các sự kiện (trước/sau/trong lúc)
   - Thay đổi: đèn tín hiệu, vị trí xe, giao thông
   - Môi trường: thời tiết, ánh sáng, điều kiện đường

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Hãy trả lời bằng cách chỉ ra chữ cái của đáp án đúng (A, B, C, hoặc D) và giải thích ngắn gọn lý do."""

    return prompt


def create_simple_prompt(question: str, choices: list[str]) -> str:
    """
    Create a simple prompt without detailed instructions.

    Args:
        question: The question
        choices: List of answer choices

    Returns:
        Simple formatted prompt
    """
    choices_text = "\n".join(choices)

    prompt = f"""Câu hỏi: {question}

{choices_text}

Trả lời (chỉ ghi chữ cái A, B, C, hoặc D):"""

    return prompt


def create_traffic_prompt_with_context(
    question: str,
    choices: List[str],
    detections_dict: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    frame_indices: Optional[List[int]] = None
) -> str:
    """
    Create a prompt with detected traffic sign context including location information.

    Args:
        question: The traffic safety question in Vietnamese
        choices: List of answer choices
        detections_dict: Dict mapping frame_index -> List of detection objects
                         Each detection object: {'sign_name': str, 'bbox': list,
                                                 'confidence': float, 'location': str}
        frame_indices: List of frame indices (in order)

    Returns:
        Formatted prompt string with detection context and locations
    """
    choices_text = "\n".join(choices)

    # Build detection context text with locations
    detection_context = ""
    if detections_dict and frame_indices:
        detection_lines = []
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in detections_dict and detections_dict[frame_idx]:
                detections = detections_dict[frame_idx]
                # Format each detection with location and OCR text if available
                for detection in detections:
                    sign_name = detection['sign_name']
                    location = detection['location']
                    ocr_text = detection.get('ocr_text')
                    ocr_conf = detection.get('ocr_confidence', 0.0)

                    # Make OCR text more prominent by putting it first if available
                    if ocr_text and ocr_conf >= 0.6:
                        line = f"Khung hình {i+1}: Biển '{sign_name}' ở {location} - NỘI DUNG: \"{ocr_text}\""
                    else:
                        line = f"Khung hình {i+1}: Phát hiện biển báo '{sign_name}' ở {location}"
                    detection_lines.append(line)
            else:
                detection_lines.append(f"Khung hình {i+1}: Không phát hiện biển báo")

        if detection_lines:
            detection_context = "\n\nTHÔNG TIN BIỂN BÁO ĐÃ PHÁT HIỆN (chú ý nội dung chữ):\n" + "\n".join(detection_lines) + "\n"

    prompt = f"""Bạn là một chuyên gia về an toàn giao thông tại Việt Nam. Hãy phân tích TOÀN BỘ video từ camera hành trình và trả lời câu hỏi dựa trên:

1. ĐỌC KỸ NỘI DUNG CHỮ trên biển báo:
   - Biển chỉ đường (xanh/trắng): tên đường, hướng đi, khoảng cách
   - Biển cấm (đỏ): nội dung cấm chỉ (cấm rẽ, cấm dừng, cấm xe...)
   - Biển báo hiệu (tam giác): nội dung cảnh báo
   - Hình dạng, màu sắc, ký hiệu của biển

2. ĐỌC MŨI TÊN trên mặt đường:
   - Hướng mũi tên: thẳng (↑), trái (←), phải (→), quay đầu (↶)
   - Mũi tên kết hợp: ví dụ (←↑) = được rẽ trái HOẶC đi thẳng
   - Xác định mũi tên thuộc làn đường nào

3. XÁC ĐỊNH VỊ TRÍ và LÀN ĐƯỜNG:
   - Xe đang ở làn nào (làn trái/giữa/phải)?
   - Tổng số làn đường (đếm từ trái sang phải)
   - Loại đường: 1 chiều hay 2 chiều, đường phố hay cao tốc
   - Khoảng cách đến giao lộ/biển báo

4. PHÂN TÍCH VẠCH KẺ ĐƯỜNG:
   - Vạch liền trắng: KHÔNG được vượt/chuyển làn
   - Vạch đứt trắng: ĐƯỢC PHÉP vượt/chuyển làn
   - Vạch sang đường (zebra): ưu tiên người đi bộ
   - Vạch dừng (stop line): phải dừng trước vạch
   - Vạch vàng: cấm dừng/đỗ xe

5. QUAN SÁT PHƯƠNG TIỆN:
   - Loại xe: xe máy, ô tô, xe tải, xe buýt, xe đạp, người đi bộ
   - Vị trí: ở làn nào, phía trước/sau/bên cạnh
   - Hành động: đang dừng/di chuyển/rẽ/vượt
   - Mật độ: đông/vắng, tắc nghẽn hay thông thoáng

6. ĐÈN TÍN HIỆU GIAO THÔNG:
   - Màu đèn: đỏ (dừng), vàng (chờ), xanh (đi)
   - Loại: đèn tròn (tất cả hướng) hay đèn mũi tên (hướng cụ thể)
   - Trạng thái và thay đổi của đèn trong video

7. PHÂN TÍCH THỜI GIAN:
   - Thứ tự các sự kiện (trước/sau/trong lúc)
   - Thay đổi: đèn tín hiệu, vị trí xe, giao thông
   - Môi trường: thời tiết, ánh sáng, điều kiện đường{detection_context}
Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Hãy trả lời bằng cách chỉ ra chữ cái của đáp án đúng (A, B, C, hoặc D) và giải thích ngắn gọn lý do."""

    return prompt


def format_video_prefix_with_detections(
    num_frames: int,
    detections_dict: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    frame_indices: Optional[List[int]] = None
) -> str:
    """
    Create video frame prefix for prompt with optional detection annotations and 2x1 grid explanation.

    Args:
        num_frames: Number of frames
        detections_dict: Dict mapping frame_index -> List of detection objects
                         Each detection object: {'sign_name': str, 'bbox': list,
                                                 'confidence': float, 'location': str}
        frame_indices: List of frame indices (in order)

    Returns:
        Formatted prefix string with grid context and detection info
    """
    # Add header explaining 2x1 grid structure
    header = "Mỗi khung hình video được hiển thị dưới dạng hình ảnh toàn cảnh (chiều rộng đầy đủ).\n\n"

    lines = []

    for i in range(num_frames):
        frame_line = f'Khung hình {i+1}: <image>'

        # # Add detection annotation if available
        # if detections_dict and frame_indices and i < len(frame_indices):
        #     frame_idx = frame_indices[i]
        #     if frame_idx in detections_dict and detections_dict[frame_idx]:
        #         signs = detections_dict[frame_idx]
        #         # Limit to first 3 signs to avoid too long prefixes
        #         signs_display = signs[:3]
        #         signs_text = ", ".join(signs_display)
        #         if len(signs) > 3:
        #             signs_text += f" (+{len(signs)-3} khác)"
        #         frame_line += f' [Phát hiện: {signs_text}]'

        lines.append(frame_line)

    return header + '\n'.join(lines) + '\n'
