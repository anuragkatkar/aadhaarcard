import os
import re
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

try:
	from paddleocr import TextRecognition
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"paddleocr is required. Install dependencies from requirements.txt"
	) from exc


app = FastAPI(title="Aadhaar OCR Service", version="0.1.0")


DETAIL_MODEL_PATH = os.getenv("DETAIL_MODEL_PATH", "yolo-aadhaar-details-v01.pt")
REC_MODEL_NAME = os.getenv("REC_MODEL_NAME", "PP-OCRv5_mobile_rec")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_DIR = os.getenv("DEBUG_DIR", "debug")


# Map model labels to the output fields we want.
LABEL_ALIASES = {
	"name": "name",
	"number": "aadhaar_number",
	"dob": "date_of_birth_or_year_of_birth",
	"gender": "gender",
	"address": "address",
}

OUTPUT_KEYS = [
	"name",
	"aadhaar_number",
	"date_of_birth_or_year_of_birth",
	"gender",
	"address",
]

API_RESPONSE_KEYS = {
	"name": "name",
	"aadhaar_number": "aadhaar number",
	"date_of_birth_or_year_of_birth": "date of birth/year of birth",
	"gender": "gender",
	"address": "address",
}


detail_model = YOLO(DETAIL_MODEL_PATH)
rec_model = TextRecognition(model_name=REC_MODEL_NAME)


def normalize_label(label: str) -> str | None:
	return LABEL_ALIASES.get(label.strip().lower())


def best_unique_detections(result: Any) -> dict[str, dict[str, Any]]:
	selected: dict[str, dict[str, Any]] = {}
	names = result.names

	for box in result.boxes:
		class_id = int(box.cls[0])
		conf = float(box.conf[0])
		raw_label = str(names[class_id])
		label = normalize_label(raw_label)
		if not label:
			continue

		xyxy = box.xyxy[0].tolist()
		current = selected.get(label)
		if current is None or conf > current["conf"]:
			selected[label] = {"conf": conf, "xyxy": xyxy}

	return selected


def crop_boxes(image: np.ndarray, detections: dict[str, dict[str, Any]]) -> tuple[list[str], list[np.ndarray]]:
	h, w = image.shape[:2]
	ordered_labels: list[str] = []
	crops: list[np.ndarray] = []

	for label in OUTPUT_KEYS:
		det = detections.get(label)
		if not det:
			continue

		x1, y1, x2, y2 = det["xyxy"]
		x1 = max(0, min(w, int(x1)))
		y1 = max(0, min(h, int(y1)))
		x2 = max(0, min(w, int(x2)))
		y2 = max(0, min(h, int(y2)))
		if x2 <= x1 or y2 <= y1:
			continue

		crop = image[y1:y2, x1:x2]
		if crop.size == 0:
			continue

		ordered_labels.append(label)
		crops.append(crop)

	return ordered_labels, crops


def extract_text_value(rec_output: Any) -> str | None:
	if isinstance(rec_output, dict):
		text = rec_output.get("rec_text")
		if text is None:
			return None
		return str(text).strip() or None

	# Fallback for future PaddleOCR output shapes.
	if hasattr(rec_output, "rec_text"):
		text = getattr(rec_output, "rec_text")
		return str(text).strip() or None

	return None


def safe_filename(text: str | None) -> str:
	value = (text or "null").strip()
	value = re.sub(r"[\\/:*?\"<>|]", "_", value)
	value = re.sub(r"\s+", " ", value).strip()
	if not value:
		return "null"
	return value[:120]


def save_debug_crops(crops: list[np.ndarray], texts: list[str | None], upload_name: str | None) -> None:
	folder_name = safe_filename(os.path.splitext(upload_name or "upload")[0])
	debug_path = os.path.join(DEBUG_DIR, folder_name)
	os.makedirs(debug_path, exist_ok=True)
	name_counts: dict[str, int] = {}

	for crop, text in zip(crops, texts):
		base_name = safe_filename(text)
		count = name_counts.get(base_name, 0) + 1
		name_counts[base_name] = count
		file_name = f"{base_name}.jpg" if count == 1 else f"{base_name}_{count}.jpg"
		cv2.imwrite(os.path.join(debug_path, file_name), crop)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/extract")
async def extract_aadhaar_details(file: UploadFile = File(...)) -> dict[str, str | None]:
	if not file.content_type or not file.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="Please upload an image file")

	content = await file.read()
	image_array = np.frombuffer(content, dtype=np.uint8)
	image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
	if image is None:
		raise HTTPException(status_code=400, detail="Invalid image")

	result = detail_model.predict(source=image, verbose=False)[0]
	detections = best_unique_detections(result)
	labels, crops = crop_boxes(image, detections)

	extracted: dict[str, str | None] = {key: None for key in OUTPUT_KEYS}
	if not crops:
		return {API_RESPONSE_KEYS[key]: extracted[key] for key in OUTPUT_KEYS}

	rec_outputs = rec_model.predict(input=crops, batch_size=len(crops))
	debug_texts: list[str | None] = []
	for label, rec_output in zip(labels, rec_outputs):
		text = extract_text_value(rec_output)
		extracted[label] = text
		debug_texts.append(text)

	if DEBUG_MODE:
		save_debug_crops(crops, debug_texts, file.filename)

	return {API_RESPONSE_KEYS[key]: extracted[key] for key in OUTPUT_KEYS}
