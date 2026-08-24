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
MAX_FILE_SIZE_MB = 15

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
    "surprising", "shocking", "new", "now", "today", "you", "why", "how", "don't miss",
    "urgent", "breaking", "exclusive"
]


def format_bytes_to_mb(size_bytes: int) -> float:
    """Return file size in megabytes."""
    return size_bytes / (1024 * 1024)


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded filename has an allowed extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_upload(uploaded_file: UploadedFile) -> Tuple[bool, str]:
    """Validate extension and size of uploaded file."""
    if not uploaded_file:
        return False, "No file provided."
    if not allowed_file(uploaded_file.name):
        return False, f"Unsupported file type: {uploaded_file.name}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    size_mb = format_bytes_to_mb(uploaded_file.size)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB."

    return True, "OK"


def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Extract text from a PDF page-by-page using PyMuPDF."""
    text_pages: List[str] = []
    try:
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    text_pages.append(f"[Page {page_number}]\n{page_text.strip()}")
                else:
                    text_pages.append(f"[Page {page_number}]\n")
    except Exception as exc:
        raise RuntimeError("Failed to parse PDF: " + str(exc)) from exc

    return "\n\n".join(text_pages)


def extract_text_from_image(file_stream: io.BytesIO) -> str:
    """Extract text from an image using Pillow and pytesseract."""
    try:
        image = Image.open(file_stream)
    except UnidentifiedImageError as exc:
        raise RuntimeError("Uploaded file is not a valid image.") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        raise RuntimeError("OCR processing failed: " + str(exc)) from exc

    return text or ""


def simple_tokenize(text: str) -> List[str]:
    """Split text into simple word tokens."""
    return re.findall(r"\b\w+'?\w*\b", text)


def count_hashtags(text: str) -> List[str]:
    """Return hashtags found in the text."""
    return re.findall(r"#\w+", text)


def count_mentions(text: str) -> List[str]:
    """Return mentions found in the text."""
    return re.findall(r"@\w+", text)


def detect_cta(text: str) -> Tuple[bool, List[str]]:
    """Check for common call-to-action phrases."""
    lowered = text.lower()
    found: List[str] = []
    for keyword in CTA_KEYWORDS:
        if keyword in lowered:
            found.append(keyword)
    return bool(found), found


def score_hook(text: str) -> Tuple[float, str]:
    """Score the opening hook heuristically."""
    if not text.strip():
        return 0.0, "No content."

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    first_sentence = sentences[0] if sentences else text.strip()
    word_count = len(simple_tokenize(first_sentence))

    score = 0.0
    reasons: List[str] = []

    if 3 <= word_count <= 25:
        score += 0.5
        reasons.append("Concise opening")
    elif word_count < 3:
        score += 0.2
        reasons.append("Opening is very brief")
    else:
        score += 0.2
        reasons.append("Opening is lengthy")

    if any(ch in first_sentence for ch in "?!"):
        score += 0.25
        reasons.append("Punctuation creates urgency")

    if any(phrase in first_sentence.lower() for phrase in HOOK_POWER_WORDS):
        score += 0.25
        reasons.append("Power words detected")

    score = min(score, 1.0)
    explanation = ", ".join(reasons) if reasons else "No clear hook signals detected"
    return score, explanation


def readability_score(text: str) -> float:
    """Measure readability heuristically based on sentence and word length."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    words = simple_tokenize(text)
    if not sentences or not words:
        return 0.0

    avg_sentence_len = len(words) / len(sentences)
    avg_word_len = sum(len(word) for word in words) / len(words)

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
    """Compute a heuristic overall engagement score from 0 to 100."""
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

    hashtag_count = analysis.get("hashtag_count", 0)
    if OPTIMAL_HASHTAGS_MIN <= hashtag_count <= OPTIMAL_HASHTAGS_MAX:
        hashtag_score = 1.0
    elif hashtag_count == 0:
        hashtag_score = 0.3
    else:
        hashtag_score = 0.6

    mention_count = analysis.get("mention_count", 0)
    mention_score = min(mention_count / 3.0, 1.0)

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
        "hashtags": hashtag_score,
        "mentions": mention_score,
        "length": length_score,
    }

    total = sum(breakdown[key] * weights[key] for key in weights)
    score_0_100 = int(round(total * 100))
    return score_0_100, breakdown


def analyze_text(text: str) -> Dict[str, object]:
    """Run deterministic content analysis and return structured insights."""
    words = simple_tokenize(text)
    word_count = len(words)
    hashtags = count_hashtags(text)
    mentions = count_mentions(text)
    has_cta, cta_list = detect_cta(text)
    hook_score, hook_explanation = score_hook(text)
    readability = readability_score(text)

    analysis = {
        "word_count": word_count,
        "char_count": len(text),
        "hashtag_count": len(hashtags),
        "hashtags": hashtags,
        "mention_count": len(mentions),
        "mentions": mentions,
        "has_cta": has_cta,
        "cta_list": cta_list,
        "hook_score": hook_score,
        "hook_explanation": hook_explanation,
        "readability_score": readability,
    }

    total_score, breakdown = engagement_score(analysis)
    analysis["engagement_score"] = total_score
    analysis["score_breakdown"] = breakdown

    suggestions: List[str] = []
    if hook_score < 0.45:
        suggestions.append("Consider a stronger opening: use a concise, curiosity-driven hook, question, or bold statement.")
    if not has_cta:
        suggestions.append("Add a clear call-to-action such as 'Learn more', 'Sign up', or 'DM to get started'.")
    if analysis["hashtag_count"] < OPTIMAL_HASHTAGS_MIN:
        suggestions.append(f"Add a few relevant hashtags (up to {OPTIMAL_HASHTAGS_MAX}) to improve discoverability.")
    elif analysis["hashtag_count"] > OPTIMAL_HASHTAGS_MAX:
        suggestions.append("Reduce the number of hashtags so the post stays focused and less spammy.")
    if word_count < IDEAL_CONTENT_MIN_WORDS:
        suggestions.append("The content is short; add a bit more value, examples, or context.")
    elif word_count > IDEAL_CONTENT_MAX_WORDS:
        suggestions.append("The content may be too long for a typical social post; trim to the key message or split it into a thread.")
    if readability < 0.45:
        suggestions.append("Shorten long sentences and use simpler wording to improve clarity.")

    analysis["suggestions"] = suggestions
    return analysis


def inject_custom_css() -> None:
    """Inject a premium, minimal visual theme for the app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
        .main > div { padding-top: 1.0rem; }
        .app-shell {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15, 23, 42, 0.04);
            border-radius: 14px;
            padding: 1.6rem;
            margin-bottom: 1rem;
        }
        .hero-badge {
            display: inline-block;
            background: #0f172a;
            color: #fff;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .upload-box {
            border: 1px dashed rgba(99, 102, 241, 0.28);
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(180deg, rgba(99, 102, 241, 0.03), rgba(148, 163, 184, 0.02));
        }
        .metric-card {
            background: linear-gradient(180deg, #ffffff, #fbfdff);
            border-radius: 12px;
            padding: 0.8rem;
            border: 1px solid rgba(14, 165, 233, 0.08);
        }
        .metric-title { color: #475569; font-size: 0.95rem; margin-bottom: 0.2rem; }
        .metric-value { font-weight: 700; font-size: 1.5rem; color: #0f172a; }
        .score-badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 8px;
            color: #fff;
            font-weight: 600;
        }
        .suggestion-list {
            background: linear-gradient(90deg, rgba(236, 253, 245, 0.8), rgba(255, 255, 255, 0.6));
            border-left: 4px solid rgba(16, 185, 129, 0.9);
            padding: 0.9rem;
            border-radius: 8px;
        }
        .footer-note { color: #64748b; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _score_color(score: int) -> str:
    """Choose a color for the engagement score badge."""
    if score >= 80:
        return "#10b981"
    if score >= 60:
        return "#f59e0b"
    return "#ef4444"


def show_analysis_ui(original_text: str, analysis: Dict[str, object]) -> None:
    """Render analysis results in a polished, readable layout."""
    st.subheader("Analysis Results")

    col1, col2, col3 = st.columns([1.6, 1, 1])
    with col1:
        st.markdown(
            "<div class='metric-card'><div class='metric-title'>Engagement Score</div><div class='metric-value'>{}</div></div>".format(f"{analysis['engagement_score']}/100"),
            unsafe_allow_html=True,
        )
        st.progress(min(max(int(analysis["engagement_score"]), 0), 100))
        st.markdown(
            f"<div style='margin-top:8px'><span class='score-badge' style='background:{_score_color(int(analysis['engagement_score']))}'>Heuristic</span> <span style='margin-left:8px' class='footer-note'>Explainable engagement score (heuristic)</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div class='metric-card'><div class='metric-title'>Word Count</div><div class='metric-value'>{}</div></div>".format(analysis["word_count"]),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div class='metric-card'><div class='metric-title'>Hashtags</div><div class='metric-value'>{}</div></div>".format(analysis["hashtag_count"]),
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Hook quality")
        st.write(f"Score: {int(round(analysis['hook_score'] * 100))}/100 — {analysis['hook_explanation']}")
    with c2:
        st.markdown("### Readability")
        st.write(f"Score: {int(round(analysis['readability_score'] * 100))}/100 — shorter sentences and simple words improve this.")
    with c3:
        st.markdown("### CTA Detection")
        if analysis["has_cta"]:
            st.success(f"CTA detected: {', '.join(analysis['cta_list'])}")
        else:
            st.info("No CTA detected.")

    with st.expander("Detailed counts and breakdown", expanded=False):
        st.write({
            "characters": analysis["char_count"],
            "word_count": analysis["word_count"],
            "hashtags": analysis["hashtags"],
            "mentions": analysis["mentions"],
            "score_breakdown": analysis["score_breakdown"],
        })

    st.markdown("### Improvement Suggestions")
    if analysis["suggestions"]:
        st.markdown("<div class='suggestion-list'>", unsafe_allow_html=True)
        for item in analysis["suggestions"]:
            st.markdown(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("No specific suggestions — this looks good for a social post!")

    st.markdown("---")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Extracted content preview")
        st.text_area("Extracted text", value=original_text, height=320)
    with right:
        st.markdown("### Quick actions")
        st.write("Select and copy the text from the preview on the left.")
        st.download_button("Download as .txt", data=original_text, file_name="extracted_text.txt", mime="text/plain")


def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="Social Media Content Analyzer", layout="wide")
    inject_custom_css()

    with st.sidebar:
        st.title("About")
        st.write(
            "Deterministic social media content analyzer. Upload a PDF or image to extract text and receive heuristics-based suggestions to improve engagement. No files are stored."
        )
        st.markdown("---")
        st.write("Built with: Streamlit, PyMuPDF, Pillow, pytesseract")
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
                    analysis = analyze_text(sample_text)
                    show_analysis_ui(sample_text, analysis)
                except Exception as exc:
                    st.error(f"Failed to load sample: {exc}")
        return

    is_valid, message = validate_upload(uploaded)
    if not is_valid:
        st.error(message)
        return

    file_bytes = uploaded.read()
    file_extension = uploaded.name.rsplit(".", 1)[1].lower()

    extracted_text = ""
    processing_error = None
    with st.spinner("Extracting text — this may take a few seconds..."):
        try:
            if file_extension == "pdf":
                extracted_text = extract_text_from_pdf(io.BytesIO(file_bytes))
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

    with st.spinner("Analyzing content..."):
        analysis = analyze_text(extracted_text)

    show_analysis_ui(extracted_text, analysis)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error("An unexpected error occurred. Please try again or contact the developer.")
        print("ERROR:", exc, file=sys.stderr)
