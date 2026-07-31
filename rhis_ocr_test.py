#!/usr/bin/env python3
"""
Standalone RHIS OCR test pipeline.

Flow:
    PDF/image -> Tesseract OCR -> raw OCR text
    -> Claude receives OCR text only -> structured JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract
from anthropic import Anthropic
from PIL import Image

SUPPORTED_IMAGES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

DEFAULT_PROMPT = """You are extracting structured information from OCR text produced from one automotive Repair Order.

IMPORTANT:
- You are NOT performing OCR.
- Use only the OCR text provided below.
- Do not invent or infer values that are not present in the OCR text.
- Treat this document as independent from every other document.
- When a value is missing or uncertain, return an empty string, 0, or [] as appropriate.
- For every important scalar field, include the exact OCR evidence and page number.
- Preserve customer complaint wording.
- Do not copy customer complaint wording into repairs_performed.

Return only valid JSON using this schema:
{
  \"ro_number\": {\"value\": \"\", \"evidence\": \"\", \"page\": null},
  \"drop_date\": {\"value\": \"\", \"evidence\": \"\", \"page\": null},
  \"pickup_date\": {\"value\": \"\", \"evidence\": \"\", \"page\": null},
  \"mileage_in\": {\"value\": 0, \"evidence\": \"\", \"page\": null},
  \"mileage_out\": {\"value\": 0, \"evidence\": \"\", \"page\": null},
  \"dealer_name\": {\"value\": \"\", \"evidence\": \"\", \"page\": null},
  \"client_name\": {\"value\": \"\", \"evidence\": \"\", \"page\": null},
  \"vin\": {\"value\": \"\", \"evidence\": \"\", \"page\": null},
  \"customer_complaints\": [],
  \"repairs_performed\": [],
  \"notes\": []
}

OCR TEXT:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tesseract OCR first, Claude formatting second')
    parser.add_argument('input_file', help='Path to an RO PDF or image')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--psm', type=int, default=6)
    parser.add_argument('--prompt', help='Optional text file containing your prompt')
    parser.add_argument('--no-claude', action='store_true')
    return parser.parse_args()


def render_document(input_path: Path, dpi: int) -> list[Image.Image]:
    ext = input_path.suffix.lower()
    if ext == '.pdf':
        doc = fitz.open(input_path)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pages: list[Image.Image] = []
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pages.append(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))
        finally:
            doc.close()
        return pages
    if ext in SUPPORTED_IMAGES:
        return [Image.open(input_path).convert('RGB')]
    raise ValueError(f'Unsupported input type: {ext}')


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 15
    )
    return Image.fromarray(thresholded)


def clean_ocr_text(text: str) -> str:
    text = text.replace('\x0c', '')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def ocr_page(image: Image.Image, page_number: int, psm: int) -> tuple[str, list[dict[str, Any]]]:
    config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'
    text = clean_ocr_text(pytesseract.image_to_string(image, config=config))
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    words: list[dict[str, Any]] = []
    for i, raw_word in enumerate(data.get('text', [])):
        word = str(raw_word or '').strip()
        if not word:
            continue
        try:
            confidence = float(data['conf'][i])
        except (TypeError, ValueError):
            confidence = -1.0
        words.append({
            'page': page_number,
            'text': word,
            'confidence': confidence,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'block': int(data['block_num'][i]),
            'paragraph': int(data['par_num'][i]),
            'line': int(data['line_num'][i]),
        })
    return text, words


def extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```json\s*|^```\s*|```$', '', cleaned, flags=re.MULTILINE).strip()
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not match:
        raise ValueError('Claude response did not contain a JSON object')
    return json.loads(match.group(0))


def call_claude(ocr_text: str, prompt_text: str) -> tuple[dict[str, Any], str, dict[str, int]]:
    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY is not set')
    model = os.environ.get('CLAUDE_MODEL', 'claude-sonnet-4-5').strip()
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=5000,
        messages=[{'role': 'user', 'content': prompt_text + '\n\n' + ocr_text}],
    )
    raw_response = ''.join(
        block.text for block in response.content if getattr(block, 'type', '') == 'text'
    ).strip()
    parsed = extract_json_object(raw_response)
    usage = {
        'input_tokens': int(response.usage.input_tokens),
        'output_tokens': int(response.usage.output_tokens),
    }
    return parsed, raw_response, usage


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        print(f'Input file not found: {input_path}', file=sys.stderr)
        return 1

    job_dir = Path(args.output_dir).expanduser().resolve() / input_path.stem
    pages_dir = job_dir / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages = render_document(input_path, args.dpi)
    if not pages:
        raise RuntimeError('No pages generated')

    all_text: list[str] = []
    all_words: list[dict[str, Any]] = []

    for page_number, original in enumerate(pages, start=1):
        original.save(pages_dir / f'page_{page_number:03d}_original.png')
        processed = preprocess_for_ocr(original)
        processed.save(pages_dir / f'page_{page_number:03d}_ocr.png')
        print(f'OCR page {page_number}/{len(pages)}...')
        page_text, page_words = ocr_page(processed, page_number, args.psm)
        all_text.append(f'================ PAGE {page_number} ================\n{page_text}')
        all_words.extend(page_words)

    combined_ocr = '\n\n'.join(all_text).strip()
    (job_dir / 'raw_ocr.txt').write_text(combined_ocr, encoding='utf-8')
    (job_dir / 'ocr_words.json').write_text(
        json.dumps(all_words, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    print(f'Saved raw OCR: {job_dir / "raw_ocr.txt"}')
    print(f'Saved OCR coordinates: {job_dir / "ocr_words.json"}')

    if args.no_claude:
        print('OCR-only mode complete.')
        return 0

    prompt_text = DEFAULT_PROMPT
    if args.prompt:
        prompt_path = Path(args.prompt).expanduser().resolve()
        prompt_text = prompt_path.read_text(encoding='utf-8').strip()

    extraction, raw_response, usage = call_claude(combined_ocr, prompt_text)
    (job_dir / 'claude_raw_response.txt').write_text(raw_response, encoding='utf-8')
    final_output = {
        'source_file': input_path.name,
        'ocr_engine': 'Tesseract',
        'claude_input': 'OCR text only',
        'tokens_in': usage['input_tokens'],
        'tokens_out': usage['output_tokens'],
        'extraction': extraction,
    }
    (job_dir / 'claude_extraction.json').write_text(
        json.dumps(final_output, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'Saved extraction: {job_dir / "claude_extraction.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
