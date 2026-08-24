"""
Social Media Content Analyzer - Streamlit app

Features:
- Upload PDF and image files
- Extract text from PDFs (PyMuPDF) and images (pytesseract)
- Deterministic content analysis with heuristics
- Responsive, professional UI with loading states and error handling

This file is intentionally organized with functions and type hints.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import pytesseract
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Ensure Tesseract is discoverable on Windows/local dev while staying compatible with Linux deployment.
TESSERACT_CANDIDATE = shutil.which("tesseract")
if not TESSERACT_CANDIDATE and os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    TESSERACT_CANDIDATE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if TESSERACT_CANDIDATE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CANDIDATE

# Configuration
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = 15  # maximum upload size in megabytes

# Heuristic configuration
OPTIMAL_HASHTAGS_MIN = 1
OPTIMAL_HASHTAGS_MAX = 8
IDEAL_CONTENT_MIN_WORDS = 20
IDEAL_CONTENT_MAX_WORDS = 280

CTA_KEYWORDS = [
    "buy", "shop", "order", "subscribe", "sign up", "signup", "register", "download",
    "join", "learn more", "click", "dm", "message", "apply now", "contact us", "get started"
]
HOOK_POWER_WORDS = [
    "surprising", "shocking", "new", "now", "today", "you", "why", "how", "don\'t miss", "urgent",
    "breaking", "exclusive"
]


def format_bytes_to_mb(size_bytes: int) -> float:
    """Return file size in megabytes."""
    return size_bytes / (1024 * 1024)


def allowed_file(filename: str) -> bool:
    """Check the file extension is allowed."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_upload(uploaded_file: UploadedFile) -> Tuple[bool, str]:
    """Validate uploaded file for extension and size.

    Returns (is_valid, message).
    """
    if not uploaded_file:
        return False, "No file provided."

    if not allowed_file(uploaded_file.name):
        return False, f"Unsupported file type: {uploaded_file.name}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    size_mb = format_bytes_to_mb(uploaded_file.size)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB."

    return True, "OK"


def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Extract text from a PDF stream using PyMuPDF (fitz).

    Returns concatenated text with page markers like [Page 1]. Handles PDFs with no extractable
    text gracefully by returning an empty string.
    """
    text_pages: List[str] = []
    try:
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                try:
                    page_text = page.get_text("text") or ""
                except Exception:
                    page_text = ""
                if page_text.strip():
                    text_pages.append(f"[Page {i}]\n" + page_text.strip())
                else:
                    # try extracting blocks or fallback to empty
                    text_pages.append(f"[Page {i}]\n")
    except Exception as exc:
        raise RuntimeError("Failed to parse PDF: " + str(exc))

    return "\n\n".join(text_pages)


def extract_text_from_image(file_stream: io.BytesIO) -> str:
    """Extract text from image using Pillow + pytesseract.

    Returns the OCR text (may be empty string on failure).
    """
    try:
        image = Image.open(file_stream)
    except UnidentifiedImageError:
        raise RuntimeError("Uploaded file is not a valid image.")

    # Convert to RGB to normalize formats
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        raise RuntimeError("OCR processing failed: " + str(exc))

    return text or ""


# Analysis functions

def simple_tokenize(text: str) -> List[str]:
    """Split text into words (simple heuristic)."""
    return re.findall(r"\b\w+'?\w*\b", text)


def count_hashtags(text: str) -> List[str]:
    return re.findall(r"#\w+", text)


def count_mentions(text: str) -> List[str]:
    return re.findall(r"@\w+", text)


def detect_cta(text: str) -> Tuple[bool, List[str]]:
    found = []
    ltext = text.lower()
    for kw in CTA_KEYWORDS:
        if kw in ltext:
            found.append(kw)
    return (len(found) > 0, found)


def score_hook(text: str) -> Tuple[float, str]:
    """Heuristic scoring for the opening hook.

    Returns (score 0-1, explanation).
    """
    if not text.strip():
        return 0.0, "No content."

    # Use first 1-2 sentences as opening
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    first = sentences[0] if sentences else text.strip()
    words = simple_tokenize(first)
    word_count = len(words)
    score = 0.0
    reasons = []

    # short and punchy openings get higher score
    if 3 <= word_count <= 25:
        score += 0.5
        reasons.append("Concise opening")
    elif word_count < 3:
        score += 0.2
        reasons.append("Too short to form a hook")
    else:
        # long opening sentence
        score += 0.2
        reasons.append("Long opening sentence")

    # presence of hook words or punctuation
    if any(w in first.lower() for w in ["?", "!"]):
        score += 0.25
        reasons.append("Engaging punctuation detected")

    if any(pw in first.lower() for pw in HOOK_POWER_WORDS):
        score += 0.25
        reasons.append("Power words detected in opening")

    score = min(score, 1.0)
    explanation = ", ".join(reasons) if reasons else "No clear hook signals detected"
    return score, explanation


def readability_score(text: str) -> float:
    """A simple readability proxy: average sentence length and word length.

    Lower average sentence length increases readability. Returns 0-1 where 1 is more readable.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    words = simple_tokenize(text)
    if not sentences or not words:
        return 0.0

    avg_sentence_len = len(words) / len(sentences)
    avg_word_len = sum(len(w) for w in words) / len(words)

    # heuristics: prefer avg sentence len between 8-18, avg word len < 6
    score = 0.0
    if 8 <= avg_sentence_len <= 18:
        score += 0.6
    elif avg_sentence_len < 8:
        score += 0.4
    else:
        score += 0.2

    if avg_word_len <= 6:
        score += 0.4
    else:
        score += 0.1

    return min(score, 1.0)


def engagement_score(analysis: Dict[str, object]) -> Tuple[int, Dict[str, float]]:
    """Compute overall engagement score (0-100) from weighted heuristics.

    Returns (score, breakdown)
    """
    # Weights
    weights = {
        "hook": 0.30,
        "readability": 0.25,
        "cta": 0.20,
        "hashtags": 0.10,
        "mentions": 0.05,
        "length": 0.10,
    }

    hook = analysis.get("hook_score", 0.0)
    readability = analysis.get("readability_score", 0.0)
    cta = 1.0 if analysis.get("has_cta") else 0.0

    hashtags = analysis.get("hashtag_count", 0)
    if OPTIMAL_HASHTAGS_MIN <= hashtags <= OPTIMAL_HASHTAGS_MAX:
        hashtags_score = 1.0
    elif hashtags == 0:
        hashtags_score = 0.3
    else:
        hashtags_score = 0.6

    mentions = analysis.get("mention_count", 0)
    mentions_score = min(mentions / 3.0, 1.0)  # more mentions up to 3 improves score

    # length score: best if within ideal range
    word_count = analysis.get("word_count", 0)
    if IDEAL_CONTENT_MIN_WORDS <= word_count <= IDEAL_CONTENT_MAX_WORDS:
        length_score = 1.0
    elif word_count < IDEAL_CONTENT_MIN_WORDS:
        length_score = 0.4
    else:
        length_score = 0.6

    breakdown = {
        "hook": hook,
        "readability": readability,
        "cta": cta,
        "hashtags": hashtags_score,
        "mentions": mentions_score,
        "length": length_score,
    }

    total = sum(breakdown[k] * weights[k] for k in weights)
    score_0_100 = int(round(total * 100))
    return score_0_100, breakdown


def analyze_text(text: str) -> Dict[str, object]:
    """Run deterministic analysis on text and return structured results."""
    words = simple_tokenize(text)
    word_count = len(words)
    char_count = len(text)
    hashtags = count_hashtags(text)
    mentions = count_mentions(text)
    has_cta, cta_list = detect_cta(text)
    hook_score, hook_expl = score_hook(text)
    read_score = readability_score(text)

    analysis = {
        "word_count": word_count,
        "char_count": char_count,
        "hashtag_count": len(hashtags),
        "hashtags": hashtags,
        "mention_count": len(mentions),
        "mentions": mentions,
        "has_cta": has_cta,
        "cta_list": cta_list,
        "hook_score": hook_score,
        "hook_explanation": hook_expl,
        "readability_score": read_score,
    }

    engagement, breakdown = engagement_score(analysis)
    analysis["engagement_score"] = engagement
    analysis["score_breakdown"] = breakdown

    # Generate suggestions
    suggestions = []
    if hook_score < 0.45:
        suggestions.append("Consider a stronger opening: use a concise, curiosity-driving hook (question, bold statement, or power word).")

    if not has_cta:
        suggestions.append("Add a clear call-to-action (CTA) such as 'Sign up', 'Learn more', 'Download', or 'DM to learn more'.")

    if analysis["hashtag_count"] < OPTIMAL_HASHTAGS_MIN:
        suggestions.append(f"Consider adding relevant hashtags (up to {OPTIMAL_HASHTAGS_MAX}) to improve discoverability.")
    elif analysis["hashtag_count"] > OPTIMAL_HASHTAGS_MAX:
        suggestions.append("Consider reducing hashtags to focus on the most relevant ones — too many can look spammy.")

    if word_count < IDEAL_CONTENT_MIN_WORDS:
        suggestions.append("Content is very short — add context or value to help readers engage.")
    elif word_count > IDEAL_CONTENT_MAX_WORDS:
        suggestions.append("Content may be long for social posts — consider trimming to a concise core message or breaking into a thread.")

    if read_score < 0.45:
        suggestions.append("Shorten long sentences and prefer simpler words to improve clarity and readability.")

    analysis["suggestions"] = suggestions
    return analysis


# UI helpers

<<<<<<< HEAD
def show_analysis_ui(original_text: str, analysis: Dict[str, object]) -> None:
    """Render analysis results in Streamlit UI."""
    st.subheader("Analysis Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Engagement Score", f"{analysis['engagement_score']}/100")
    col2.metric("Word Count", analysis["word_count"])
    col3.metric("Hashtags", analysis["hashtag_count"])

    st.markdown("**Hook**")
    st.write(f"Score: {int(round(analysis['hook_score']*100))}/100 — {analysis['hook_explanation']}")

    st.markdown("**Readability**")
    st.write(f"Score: {int(round(analysis['readability_score']*100))}/100 — shorter sentences and simple words improve this.")

    st.markdown("**CTA Detection**")
    if analysis["has_cta"]:
        st.success(f"CTA detected: {', '.join(analysis['cta_list'])}")
    else:
        st.info("No CTA detected.")

    with st.expander("Detailed counts and breakdown"):
=======

def inject_custom_css() -> None:
    """Inject a refined stylesheet and typography for a premium, human-designed look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        .main > div { padding-top: 1.0rem; }
        .app-shell { background: linear-gradient(180deg,#ffffff 0%, #f8fafc 100%); border-radius: 14px; padding: 1.6rem; margin-bottom: 1rem; border: 1px solid rgba(15,23,42,0.04); }
        .hero-badge { display:inline-block; background:#0f172a; color:#fff; padding:0.28rem 0.6rem; border-radius:999px; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase; }
        .upload-box { border: 1px dashed rgba(99,102,241,0.28); border-radius:12px; padding:1rem; background: linear-gradient(180deg, rgba(99,102,241,0.03), rgba(148,163,184,0.02)); }
        .metric-card { background: linear-gradient(180deg,#ffffff,#fbfdff); border-radius:12px; padding:0.8rem; border:1px solid rgba(14,165,233,0.06); }
        .metric-title { color:#475569; font-size:0.95rem; margin-bottom:0.2rem; }
        .metric-value { font-weight:700; font-size:1.5rem; color:#0f172a; }
        .score-badge { display:inline-block; padding:0.35rem 0.7rem; border-radius:8px; color:#fff; font-weight:600; }
        .suggestion-list { background: linear-gradient(90deg, rgba(236,253,245,0.8), rgba(255,255,255,0.6)); border-left:4px solid rgba(16,185,129,0.9); padding:0.9rem; border-radius:8px; }
        .footer-note { color: #64748b; font-size:0.9rem; }
        .progress-bar .stProgress > div { background: linear-gradient(90deg,#06b6d4,#3b82f6); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _score_color(score: int) -> str:
    """Return a hex color based on score for the score badge."""
    if score >= 80:
        return "#10b981"  # green
    if score >= 60:
        return "#f59e0b"  # amber
    return "#ef4444"      # red


def show_analysis_ui(original_text: str, analysis: Dict[str, object]) -> None:
    """Render analysis results in a more polished layout with progress and actionable items."""
    st.subheader("Analysis Results")

    # Top metrics row
    m1, m2, m3 = st.columns([1.6, 1, 1])
    with m1:
        st.markdown("<div class='metric-card'><div class='metric-title'>Engagement Score</div><div class='metric-value'>{}</div></div>".format(f"{analysis['engagement_score']}/100"), unsafe_allow_html=True)
        st.progress(min(max(int(analysis['engagement_score']), 0), 100))
        st.markdown(f"<div style='margin-top:8px'><span class='score-badge' style='background:{_score_color(analysis['engagement_score'])}'>Heuristic</span> <span style='margin-left:8px' class='footer-note'>Explainable engagement score (heuristic)</span></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'><div class='metric-title'>Word Count</div><div class='metric-value'>{}</div></div>".format(analysis["word_count"]), unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'><div class='metric-title'>Hashtags</div><div class='metric-value'>{}</div></div>".format(analysis["hashtag_count"]), unsafe_allow_html=True)

    # Second row: hook, readability, CTA
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Hook quality")
        st.write(f"Score: {int(round(analysis['hook_score']*100))}/100 — {analysis['hook_explanation']}")
    with c2:
        st.markdown("### Readability")
        st.write(f"Score: {int(round(analysis['readability_score']*100))}/100 — shorter sentences and simple words improve this.")
    with c3:
        st.markdown("### CTA Detection")
        if analysis["has_cta"]:
            st.success(f"CTA detected: {', '.join(analysis['cta_list'])}")
        else:
            st.info("No CTA detected.")

    # Expandable detailed breakdown
    with st.expander("Detailed counts and breakdown", expanded=False):
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
        st.write({
            "characters": analysis["char_count"],
            "word_count": analysis["word_count"],
            "hashtags": analysis["hashtags"],
            "mentions": analysis["mentions"],
            "score_breakdown": analysis["score_breakdown"],
        })

<<<<<<< HEAD
    st.markdown("**Suggestions**")
    if analysis["suggestions"]:
        for s in analysis["suggestions"]:
            st.write(f"- {s}")
=======
    # Suggestions
    st.markdown("### Improvement Suggestions")
    if analysis["suggestions"]:
        st.markdown("<div class='suggestion-list'>", unsafe_allow_html=True)
        for s in analysis["suggestions"]:
            st.markdown(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
    else:
        st.write("No specific suggestions — this looks good for a social post!")

    st.markdown("---")
<<<<<<< HEAD
    st.markdown("**Extracted content preview**")
    st.text_area("Extracted text", value=original_text, height=300)

=======
    # Show extracted content and quick actions in two columns
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Extracted content preview")
        st.text_area("Extracted text", value=original_text, height=320)
    with right:
        st.markdown("### Quick actions")
        if st.button("Copy text"):
            # Streamlit cannot access clipboard from server; provide a copy helper
            st.write("Select and copy the text from the preview box on the left.")
        if st.button("Download as .txt"):
            st.download_button("Download extracted text", data=original_text, file_name="extracted_text.txt", mime="text/plain")
        st.markdown("---")
        st.markdown("#### Notes")
        st.markdown("- Heuristic score only — not a replacement for A/B testing.\n- Improve hook and add a concise CTA to increase engagement.")
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)

def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="Social Media Content Analyzer", layout="wide")
<<<<<<< HEAD

    # Sidebar
=======
    inject_custom_css()

>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
    with st.sidebar:
        st.title("About")
        st.write("Deterministic social media content analyzer. Upload a PDF or image to extract text and receive heuristics-based suggestions to improve engagement. No files are stored.")
        st.markdown("---")
        st.write("Built with: Streamlit, PyMuPDF, Pillow, pytesseract")
<<<<<<< HEAD

    # Header / Hero
    st.title("Social Media Content Analyzer")
    st.write("Upload a PDF or image of a social post; the app extracts text and provides actionable suggestions to improve engagement.")

    # Upload area
    st.header("Upload document")
    uploaded = st.file_uploader("Drag & drop or click to select a file (PDF, PNG, JPG, JPEG, WEBP)", type=list(ALLOWED_EXTENSIONS))

    if not uploaded:
        st.info("No file uploaded yet. Try the included sample test or upload your own document.")
        if st.button("Load sample test text into analyzer"):
            with st.spinner("Loading sample..."):
                try:
                    sample_text = open("sample_test.txt", "r", encoding="utf-8").read()
=======
        st.caption("Heuristic engagement score only — useful for rapid assessment, not scientific measurement.")

    st.markdown(
        """
        <div class="app-shell">
            <div class="hero-badge">Content Intelligence</div>
            <h1 style="margin:0;">Social Media Content Analyzer</h1>
            <p style="margin-top:0.6rem; color:#475569; font-size:1.03rem;">Upload a PDF or image, extract the social post text, and get actionable engagement feedback with clear, explainable heuristics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.header("Upload document")
    uploaded = st.file_uploader(
        "Drag & drop or click to select a file (PDF, PNG, JPG, JPEG, WEBP)",
        type=list(ALLOWED_EXTENSIONS),
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded:
        st.info("No file uploaded yet. Try the included sample test or upload your own document.")
        if st.button("Load sample content", type="primary"):
            with st.spinner("Loading sample..."):
                try:
                    with open("sample_test.txt", "r", encoding="utf-8") as sample_file:
                        sample_text = sample_file.read()
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
                    analysis = analyze_text(sample_text)
                    show_analysis_ui(sample_text, analysis)
                except Exception as exc:
                    st.error(f"Failed to load sample: {exc}")
        return

<<<<<<< HEAD
    # Validate
=======
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
    is_valid, message = validate_upload(uploaded)
    if not is_valid:
        st.error(message)
        return

<<<<<<< HEAD
    # Process
    file_bytes = uploaded.read()
    file_stream = io.BytesIO(file_bytes)
=======
    file_bytes = uploaded.read()
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
    ext = uploaded.name.rsplit('.', 1)[1].lower()

    extracted_text = ""
    processing_error = None
    with st.spinner("Extracting text — this may take a few seconds..."):
        try:
            if ext == "pdf":
                extracted_text = extract_text_from_pdf(io.BytesIO(file_bytes))
<<<<<<< HEAD
                # If PDF extraction yields only page markers or empty, note it
=======
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
                if not extracted_text.strip() or re.fullmatch(r"(\[Page \d+\]\s*)+", extracted_text.strip()):
                    st.warning("PDF contained little to no extractable text. If the PDF is scanned or image-based, try uploading it as an image for OCR.")
            else:
                extracted_text = extract_text_from_image(io.BytesIO(file_bytes))
                if not extracted_text.strip():
                    st.warning("OCR returned no text. Try increasing image quality or checking that the image contains clear, horizontal text.")
        except RuntimeError as exc:
            processing_error = str(exc)
        except Exception:
            processing_error = "An unexpected error occurred while processing the file."

    if processing_error:
        st.error(processing_error)
        return

    if not extracted_text.strip():
        st.info("No text was extracted from the uploaded file.")
        return

<<<<<<< HEAD
    # Analyze
=======
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
    with st.spinner("Analyzing content..."):
        analysis = analyze_text(extracted_text)

    show_analysis_ui(extracted_text, analysis)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
<<<<<<< HEAD
        # Friendly error message; avoid exposing internals to end users
        st.error("An unexpected error occurred. Please try again or contact the developer.")
        # For local debugging (developer): print to stderr
=======
        st.error("An unexpected error occurred. Please try again or contact the developer.")
>>>>>>> 161c0d7 (UI: premium polish — improved layout and UX)
        print("ERROR:", exc, file=sys.stderr)
