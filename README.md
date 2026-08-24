# Social Media Content Analyzer 

A Streamlit-based technical assessment project that extracts text from uploaded PDFs and images and provides deterministic, explainable heuristics to evaluate and improve social media content engagement.

Repository metadata cleaned up to remove the Copilot co-author trailer from the latest GitHub commit message.

## Project overview
This application allows a user to upload a PDF or image, extract visible text, and assess the content for social media performance using a transparent heuristic-based scoring model. It is designed for a rapid technical assessment, not a production-grade AI content platform.

## Features
- Upload PDFs and images (PNG, JPG, JPEG, WEBP) via drag-and-drop or file picker
- File validation for extension and size
- PDF text extraction using PyMuPDF, preserving page markers and page-level structure where possible
- Image OCR using Pillow + pytesseract + Tesseract
- Deterministic analysis: engagement score (0-100), hook quality, CTA detection, hashtag/mention counts, readability, and suggestions
- In-memory processing only; no upload persistence, database, or authentication

## Architecture / approach
The app uses a lightweight Streamlit interface with modular functions for file validation, extraction, analysis, and UI rendering. PDF extraction is handled page-by-page with PyMuPDF; OCR handles scanned or image-based documents. Content scoring relies on deterministic heuristics rather than external LLMs or paid APIs, making the assessment transparent and explainable.

## Tech stack
- Python 3.12
- Streamlit
- PyMuPDF
- Pillow
- pytesseract
- Tesseract OCR (system dependency)

## Local installation
1. Clone or download the repository.
2. Open PowerShell in the project root.
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Install Tesseract OCR (see next section).

## Windows setup
- Install Tesseract OCR from the official installer: https://github.com/UB-Mannheim/tesseract/wiki
- Ensure `tesseract.exe` is available on PATH.
- You can verify with:
  ```powershell
  tesseract --version
  ```

## Tesseract installation
If Tesseract is not installed, install it before running OCR-based analysis. On Windows, the official Tesseract package is the easiest route. For Linux, install via package manager as needed.

## Running the application
From the project root, with the virtual environment activated:
```powershell
streamlit run app.py
```

## Testing
- A sample social post is provided in `sample_test.txt`.
- Use the app's built-in "Load sample test text into analyzer" button or upload your own PDF/image.

## GitHub setup
1. Initialize git in the project root if needed:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   ```
2. Create a repository on GitHub and push:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

## Streamlit Community Cloud deployment
- Push the project to GitHub.
- In Streamlit Community Cloud, choose "New app" and connect the repository.
- Set the main file to `app.py`.
- The `packages.txt` file is included for Linux package installation in the deployment environment.

## Project structure
```
social-media-content-analyzer/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── APPROACH.md
├── .gitignore
├── sample_test.txt
│
└── .streamlit/
    └── config.toml
```

## Limitations
- The engagement score is a heuristic and not scientifically validated.
- OCR results vary based on image sharpness, layout, and the Tesseract language data available.
- PDFs with complex layouts may not preserve formatting perfectly.

## Future improvements
- Add unit tests for extraction and analysis functions
- Improve readability scoring with syllable-aware metrics
- Add support for more document formats and multi-language OCR
- Add export or PDF report generation
