# RHIS Tesseract OCR Test

This is a separate personal test pipeline. It does not modify the production RHIS code.

## Flow

```text
RO PDF/image
  -> render at 300 DPI
  -> preprocess
  -> Tesseract OCR
  -> save raw OCR text and word coordinates
  -> send OCR text only to Claude
  -> save JSON
```

## Install on macOS

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Claude credentials:

```bash
export ANTHROPIC_API_KEY="your-key"
export CLAUDE_MODEL="your-current-model-name"
```

## First test OCR only

```bash
python rhis_ocr_test.py "/path/to/repair-order.pdf" --no-claude
```

Review:

```text
output/<document-name>/raw_ocr.txt
output/<document-name>/ocr_words.json
output/<document-name>/pages/
```

## Then run OCR plus Claude

```bash
python rhis_ocr_test.py "/path/to/repair-order.pdf"
```

## Use your own prompt

```bash
python rhis_ocr_test.py "/path/to/repair-order.pdf" --prompt prompt_template.txt
```

## Compare Tesseract layouts

```bash
python rhis_ocr_test.py file.pdf --no-claude --psm 6
python rhis_ocr_test.py file.pdf --no-claude --psm 4
python rhis_ocr_test.py file.pdf --no-claude --psm 11
```

Always inspect `raw_ocr.txt` before changing the prompt. If the RO number or date is already wrong in that file, the problem is OCR. If it is correct there but wrong in `claude_extraction.json`, the problem is the prompt/selection logic.
