import base64
import json
import os
import secrets
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse


APP_TITLE = "Aadhaar Extraction API"
API_PREFIX = "/api/v1"

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))
OCR_URL = os.getenv("OCR_URL", "http://localhost:8100")
MOONDREAM_URL = os.getenv("MOONDREAM_URL", "https://api.moondream.ai/v1/query")
MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY", "")
OCR_TEXT_SCORE_THRESHOLD = float(os.getenv("OCR_TEXT_SCORE_THRESHOLD", "0.0"))

API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
API_TOKENS = tuple(token.strip() for token in os.getenv("API_TOKENS", "").split(",") if token.strip())


app = FastAPI(title=APP_TITLE)


def _extract_bearer_token(authorization_header: str | None) -> str | None:
	if not authorization_header:
		return None
	scheme, _, token = authorization_header.partition(" ")
	if scheme.lower() != "bearer" or not token.strip():
		return None
	return token.strip()


def _request_token(request: Request) -> str | None:
	bearer = _extract_bearer_token(request.headers.get("authorization"))
	if bearer:
		return bearer

	for header_name in ("x-api-key", "x-api-token", "authorization-token"):
		value = request.headers.get(header_name)
		if value and value.strip():
			return value.strip()

	return None


def require_api_token(request: Request) -> None:
	if not API_AUTH_ENABLED:
		return

	if not API_TOKENS:
		raise HTTPException(status_code=500, detail="API auth is enabled but API_TOKENS is empty.")

	token = _request_token(request)
	if token is None:
		raise HTTPException(status_code=401, detail="Missing API token.")

	if any(secrets.compare_digest(token, allowed) for allowed in API_TOKENS):
		return

	raise HTTPException(status_code=401, detail="Invalid API token.")


def detect_image_mime(file_bytes: bytes, content_type: str, filename: str) -> str:
	normalized_content_type = (content_type or "").lower()
	normalized_filename = (filename or "").lower()

	if file_bytes.startswith(b"\xff\xd8\xff"):
		return "image/jpeg"
	if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
		return "image/png"

	if normalized_content_type in {"image/jpeg", "image/jpg", "image/pjpeg"}:
		return "image/jpeg"
	if normalized_content_type == "image/png":
		return "image/png"

	if normalized_filename.endswith((".jpg", ".jpeg")):
		return "image/jpeg"
	if normalized_filename.endswith(".png"):
		return "image/png"

	raise HTTPException(status_code=415, detail="Unsupported file type. Only JPEG and PNG are supported.")


def extract_paddlex_ocr_text(payload: Any) -> str:
	fragments: list[str] = []

	if isinstance(payload, dict):
		result = payload.get("result")
		if isinstance(result, dict):
			ocr_results = result.get("ocrResults")
			if isinstance(ocr_results, list):
				for item in ocr_results:
					if not isinstance(item, dict):
						continue
					pruned = item.get("prunedResult")
					if not isinstance(pruned, dict):
						continue

					rec_texts = pruned.get("rec_texts")
					rec_scores = pruned.get("rec_scores")
					if isinstance(rec_texts, list):
						for idx, text in enumerate(rec_texts):
							if isinstance(text, str):
								score_ok = True
								if isinstance(rec_scores, list) and idx < len(rec_scores):
									raw_score = rec_scores[idx]
									if isinstance(raw_score, (int, float)):
										score_ok = float(raw_score) >= OCR_TEXT_SCORE_THRESHOLD
								if not score_ok:
									continue
								stripped = text.strip()
								if stripped:
									fragments.append(stripped)

	if not fragments:
		raise HTTPException(status_code=502, detail="OCR completed but no text was extracted.")

	# Remove duplicates while preserving first-seen order.
	ordered_unique: list[str] = []
	seen: set[str] = set()
	for fragment in fragments:
		if fragment in seen:
			continue
		seen.add(fragment)
		ordered_unique.append(fragment)

	return "\n".join(ordered_unique)


def _json_from_text(text: str) -> dict[str, Any]:
	raw = text.strip()
	if raw.startswith("```"):
		chunks = raw.split("```")
		if len(chunks) >= 2:
			raw = chunks[1].strip()
			if raw.lower().startswith("json"):
				raw = raw[4:].strip()

	parsed = json.loads(raw)
	if not isinstance(parsed, dict):
		raise ValueError("Response is not a JSON object.")
	return parsed


def build_moondream_prompt(ocr_text: str) -> str:
	return (
		"You are an Aadhaar card data extraction engine. "
		"Use the image as the primary source and use OCR text only as supporting evidence for exact words and numbers. "
		"Extract details carefully. Return ONLY a valid JSON object with exactly these keys: "
		'"name", "aadhar number", "date of birth", "gender", "address". '
		"Do not add markdown, explanations, or extra keys. "
		"If a value is missing or unreadable, set it to null. "
		"For \"aadhar number\", return only the 12-digit number without spaces if present. "
		"For \"date of birth\", keep the value exactly as visible on the card. "
		"OCR support text follows:\n\n"
		f"{ocr_text}"
	)


@app.get(f"{API_PREFIX}/health")
async def health() -> dict[str, str]:
	return {"status": "ok"}


@app.get("/health")
async def health_root() -> dict[str, str]:
	return {"status": "ok"}


@app.post(f"{API_PREFIX}/aadhaar/extract")
async def extract_aadhaar(
	request: Request,
	file: UploadFile = File(..., description="Aadhaar card image (jpg/jpeg/png)"),
) -> JSONResponse:
	require_api_token(request)

	if not MOONDREAM_API_KEY.strip():
		raise HTTPException(status_code=500, detail="MOONDREAM_API_KEY is not configured.")

	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	mime = detect_image_mime(file_bytes, file.content_type or "", file.filename or "")
	encoded_image = base64.b64encode(file_bytes).decode("ascii")
	image_url = f"data:{mime};base64,{encoded_image}"

	ocr_payload = {
		"file": encoded_image,
		"fileType": 1,
	}

	async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
		try:
			ocr_resp = await client.post(f"{OCR_URL}/ocr", json=ocr_payload)
			ocr_resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise HTTPException(status_code=502, detail=f"Paddle OCR failed: {exc}") from exc
		except httpx.RequestError as exc:
			raise HTTPException(status_code=503, detail=f"Paddle OCR unavailable: {exc}") from exc

		ocr_json = ocr_resp.json()
		ocr_text = extract_paddlex_ocr_text(ocr_json)
		prompt = build_moondream_prompt(ocr_text)

		try:
			md_resp = await client.post(
				MOONDREAM_URL,
				headers={
					"Content-Type": "application/json",
					"X-Moondream-Auth": MOONDREAM_API_KEY,
				},
				json={
					"image_url": image_url,
					"question": prompt,
				},
			)
			md_resp.raise_for_status()
		except httpx.HTTPStatusError as exc:
			raise HTTPException(status_code=502, detail=f"Moondream API failed: {exc}") from exc
		except httpx.RequestError as exc:
			raise HTTPException(status_code=503, detail=f"Moondream API unavailable: {exc}") from exc

	md_json = md_resp.json()
	answer_text = str(md_json.get("answer", "")).strip()
	if not answer_text:
		raise HTTPException(status_code=502, detail="Moondream response did not include an answer.")

	try:
		extracted = _json_from_text(answer_text)
	except (ValueError, json.JSONDecodeError):
		# Return raw answer if the model ignored strict JSON instructions.
		extracted = {
			"name": None,
			"aadhar number": None,
			"date of birth": None,
			"gender": None,
			"address": None,
		}

	for key in ("name", "aadhar number", "date of birth", "gender", "address"):
		extracted.setdefault(key, None)

	return JSONResponse(
		{
			"extracted": extracted,
			"ocr_text": ocr_text,
			"moondream_answer": answer_text,
		}
	)

