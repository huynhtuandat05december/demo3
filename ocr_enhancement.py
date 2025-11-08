"""
OCR enhancement for Vietnamese traffic sign text extraction.
Runs at inference time on detected sign regions.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignTextExtractor:
    """Extract Vietnamese text from traffic sign regions using OCR"""

    def __init__(self, use_paddleocr: bool = True):
        """
        Initialize OCR engine.

        Args:
            use_paddleocr: If True, use PaddleOCR (recommended for Vietnamese).
                          If False, use EasyOCR as fallback.
        """
        self.use_paddleocr = use_paddleocr
        self.ocr = None

        try:
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        except ImportError as e:
            logger.warning(f"[OCR] Failed to initialize: {e}")
            logger.warning("[OCR] OCR will be disabled. Install with: pip install paddleocr")
            self.ocr = None

    def preprocess_sign_region(self, sign_patch: np.ndarray) -> np.ndarray:
        """
        Preprocess traffic sign image for better OCR accuracy.

        Args:
            sign_patch: Sign region as numpy array (H, W, 3) in BGR

        Returns:
            Enhanced image for OCR
        """
        if sign_patch is None or sign_patch.size == 0:
            return sign_patch

        # Convert to RGB if BGR
        if len(sign_patch.shape) == 3 and sign_patch.shape[2] == 3:
            rgb = cv2.cvtColor(sign_patch, cv2.COLOR_BGR2RGB)
        else:
            rgb = sign_patch

        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Increase sharpness
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def extract_text_from_region(
        self,
        frame: np.ndarray,
        bbox: List[float],
        confidence_threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        """
        Extract text from a bounding box region in the frame.

        Args:
            frame: Full video frame (H, W, 3) in BGR
            bbox: Bounding box as [x1, y1, x2, y2]
            confidence_threshold: Minimum OCR confidence to accept result

        Returns:
            (extracted_text, confidence) or (None, 0.0) if OCR disabled/failed
        """
        if self.ocr is None:
            return None, 0.0

        try:
            # Extract sign region with padding
            x1, y1, x2, y2 = map(int, bbox)

            # Add padding (10% of bbox size)
            padding_x = max(10, int((x2 - x1) * 0.1))
            padding_y = max(10, int((y2 - y1) * 0.1))

            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(frame.shape[1], x2 + padding_x)
            y2 = min(frame.shape[0], y2 + padding_y)

            # Extract and validate
            sign_patch = frame[y1:y2, x1:x2]

            if sign_patch.size == 0 or sign_patch.shape[0] < 10 or sign_patch.shape[1] < 10:
                return None, 0.0

            # Preprocess
            enhanced = self.preprocess_sign_region(sign_patch)

            # Run OCR
            if self.use_paddleocr:
                result = self.ocr.ocr(enhanced, cls=True)

                if result and result[0]:
                    texts = []
                    confidences = []

                    for line in result[0]:
                        text = line[1][0]
                        conf = line[1][1]

                        if conf >= confidence_threshold:
                            texts.append(text)
                            confidences.append(conf)

                    if texts:
                        combined_text = ' '.join(texts)
                        avg_confidence = sum(confidences) / len(confidences)
                        return combined_text, avg_confidence
            else:
                # EasyOCR
                result = self.ocr.readtext(enhanced)

                if result:
                    texts = []
                    confidences = []

                    for detection in result:
                        text = detection[1]
                        conf = detection[2]

                        if conf >= confidence_threshold:
                            texts.append(text)
                            confidences.append(conf)

                    if texts:
                        combined_text = ' '.join(texts)
                        avg_confidence = sum(confidences) / len(confidences)
                        return combined_text, avg_confidence

            return None, 0.0

        except Exception as e:
            logger.warning(f"[OCR] Error during text extraction: {e}")
            return None, 0.0

    def enhance_detections_with_ocr(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        confidence_threshold: float = 0.6
    ) -> List[Dict]:
        """
        Add OCR text to detection dictionaries.

        Args:
            frame: Full video frame (H, W, 3) in BGR
            detections: List of detection dicts with 'bbox', 'sign_name', etc.
            confidence_threshold: Minimum OCR confidence

        Returns:
            Updated detections list with 'ocr_text' and 'ocr_confidence' fields
        """
        if self.ocr is None:
            # OCR disabled, return unchanged
            for det in detections:
                det['ocr_text'] = None
                det['ocr_confidence'] = 0.0
            return detections

        for detection in detections:
            bbox = detection['bbox']
            ocr_text, ocr_conf = self.extract_text_from_region(
                frame, bbox, confidence_threshold
            )

            detection['ocr_text'] = ocr_text
            detection['ocr_confidence'] = ocr_conf

            if ocr_text:
                logger.debug(f"[OCR] Detected '{detection['sign_name']}' → Text: '{ocr_text}' (conf: {ocr_conf:.2f})")

        return detections

    def batch_enhance_detections(
        self,
        video_frames: List[np.ndarray],
        frame_indices: List[int],
        detections_dict: Dict[int, List[Dict]],
        confidence_threshold: float = 0.6
    ) -> Dict[int, List[Dict]]:
        """
        Batch process multiple frames with detections.

        Args:
            video_frames: List of video frames
            frame_indices: List of frame indices
            detections_dict: Dict mapping frame_idx -> list of detections
            confidence_threshold: Minimum OCR confidence

        Returns:
            Updated detections_dict with OCR information
        """
        if self.ocr is None:
            logger.info("[OCR] OCR engine not available, skipping text extraction")
            return detections_dict

        logger.info(f"[OCR] Processing {len(frame_indices)} frames for text extraction...")

        for i, frame_idx in enumerate(frame_indices):
            if frame_idx in detections_dict and detections_dict[frame_idx]:
                frame = video_frames[i]
                detections_dict[frame_idx] = self.enhance_detections_with_ocr(
                    frame,
                    detections_dict[frame_idx],
                    confidence_threshold
                )

        return detections_dict


# Convenience function for easy integration
def add_ocr_to_detections(
    video_frames: List[np.ndarray],
    frame_indices: List[int],
    detections_dict: Dict[int, List[Dict]],
    use_paddleocr: bool = True,
    confidence_threshold: float = 0.6
) -> Dict[int, List[Dict]]:
    """
    Convenience function to add OCR text to detections.

    Args:
        video_frames: List of video frames
        frame_indices: List of frame indices
        detections_dict: Dict mapping frame_idx -> list of detections
        use_paddleocr: Use PaddleOCR (True) or EasyOCR (False)
        confidence_threshold: Minimum OCR confidence

    Returns:
        Updated detections_dict with OCR information
    """
    extractor = SignTextExtractor(use_paddleocr=use_paddleocr)
    return extractor.batch_enhance_detections(
        video_frames,
        frame_indices,
        detections_dict,
        confidence_threshold
    )


if __name__ == "__main__":
    # Test the OCR extractor
    print("Testing OCR Enhancement Module...")

    # Create dummy test data
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 200, 200]

    extractor = SignTextExtractor(use_paddleocr=True)

    if extractor.ocr:
        text, conf = extractor.extract_text_from_region(dummy_frame, dummy_bbox)
        print(f"OCR Result: text='{text}', confidence={conf:.2f}")
    else:
        print("OCR not available. Install with: pip install paddleocr")
