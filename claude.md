# CLAUDE.md

## Goal
Build a simple and clear Aadhaar extraction API.

## Current pipeline (implemented)
1. User uploads an Aadhaar image to FastAPI `/extract`.
2. YOLO model (`yolo-aadhaar-details-v01.pt`) detects fields.
3. Keep only the highest-confidence detection per unique field.
4. Crop all selected boxes.
5. Send all crops as one batch to `PP-OCRv5_mobile_rec`.
6. Return JSON with:
   - `name`
   - `aadhaar number`
   - `date of birth/year of birth`
   - `gender`
   - `address`
7. Any missing field is `null`.

## Planned next stages
- Stage 0 card detector YOLO: detect Aadhaar card and crop card first.
- Stage 2 address-line YOLO: split address into multiple lines.
- OCR each address line and merge into a final address string.

## Coding style for this project
- Keep code short and readable.
- Prefer small helper functions.
- Avoid over-engineering.
- Keep API output stable even when detections are missing.
