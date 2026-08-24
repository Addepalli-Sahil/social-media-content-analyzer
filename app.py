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
    """Score the opening hook with smoother heuristics.

    Returns a score between 0.0 and 1.0 and a short explanation.
    """
    if not text.strip():
        return 0.0, "No content."

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    first = sentences[0] if sentences else text.strip()
    tokens = simple_tokenize(first)
    wc = max(len(tokens), 0)

    # length heuristic: ideal opening length ~8-15 words
    ideal = 12.0
    length_score = max(0.0, 1.0 - (abs(wc - ideal) / (ideal * 1.5)))

    # punctuation boost for ? or ! in opening
    punct_boost = 0.15 if any(ch in first for ch in "?!") else 0.0

    # power words boost
    pw_boost = 0.25 if any(pw in first.lower() for pw in HOOK_POWER_WORDS) else 0.0

    # direct-address boost (starts with 'you', 'what', 'why', 'how')
    direct_start = bool(re.match(r"^(you\b|what\b|why\b|how\b|don\'t\b|here\'s\b)", first.strip().lower()))
    direct_boost = 0.12 if direct_start else 0.0

    raw_score = (0.6 * length_score) + punct_boost + pw_boost + direct_boost
    score = min(raw_score, 1.0)

    reasons = []
    if length_score > 0.7:
        reasons.append("Good opening length")
    elif length_score > 0.4:
        reasons.append("OK opening length")
    else:
        reasons.append("Consider shortening or tightening the opening")

    if punct_boost:
        reasons.append("Engaging punctuation")
    if pw_boost:
        reasons.append("Power words present")
    if direct_boost:
        reasons.append("Direct audience address detected")

    explanation = ", ".join(reasons)
    return score, explanation


def readability_score(text: str) -> float:
    """Return a readability proxy (0.0 to 1.0).

    Uses average sentence length and proportion of long words to compute a smoother score.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    words = simple_tokenize(text)
    if not sentences or not words:
        return 0.0

    avg_sentence_len = len(words) / len(sentences)
    long_words = [w for w in words if len(w) >= 7]
    long_ratio = len(long_words) / len(words)

    # sentence length score (ideal 10-18)
    if avg_sentence_len <= 10:
        sent_score = 1.0
    elif avg_sentence_len <= 18:
        sent_score = 1.0 - ((avg_sentence_len - 10) / 20.0)
    else:
        sent_score = max(0.0, 1.0 - ((avg_sentence_len - 18) / 30.0))

    # long word penalty
    long_penalty = max(0.0, 1.0 - (long_ratio * 1.5))

    score = (0.65 * sent_score) + (0.35 * long_penalty)
    return max(0.0, min(score, 1.0))


def engagement_score(analysis: Dict[str, object]) -> Tuple[int, Dict[str, float]]:
    """Compute a realistic-looking engagement score (0-100) using weighted, smoothed heuristics."""
    weights = {
        "hook": 0.28,
        "readability": 0.22,
        "cta": 0.22,
        "hashtags": 0.12,
        "mentions": 0.04,
        "length": 0.12,
    }

    hook = analysis.get("hook_score", 0.0)
    readability = analysis.get("readability_score", 0.0)
    cta = 1.0 if analysis.get("has_cta") else 0.0

    hashtags = analysis.get("hashtag_count", 0)
    # Smooth hashtag score: best around 2-5 hashtags
    if hashtags == 0:
        hashtag_score = 0.25
    else:
        # gaussian-like preference around mean 3
        diff = (hashtags - 3) / 3.0
        hashtag_score = max(0.0, 1.0 - (diff * diff))

    mentions = analysis.get("mention_count", 0)
    mention_score = min(mentions / 2.0, 1.0)

    word_count = analysis.get("word_count", 0)
    # length score favors mid-length posts; penalize very short/very long smoothly
    if word_count <= IDEAL_CONTENT_MIN_WORDS:
        length_score = 0.3 + (word_count / max(1.0, IDEAL_CONTENT_MIN_WORDS)) * 0.7
    elif word_count <= IDEAL_CONTENT_MAX_WORDS:
        length_score = 1.0
    else:
        # decay after max
        length_score = max(0.2, 1.0 - ((word_count - IDEAL_CONTENT_MAX_WORDS) / IDEAL_CONTENT_MAX_WORDS))

    breakdown = {
        "hook": round(hook, 3),
        "readability": round(readability, 3),
        "cta": round(cta, 3),
        "hashtags": round(hashtag_score, 3),
        "mentions": round(mention_score, 3),
        "length": round(length_score, 3),
    }

    total = sum(breakdown[k] * weights[k] for k in weights)
    # small calibration to avoid extreme 0/100 for most natural content
    calibrated = (total * 0.92) + 0.04
    score_0_100 = int(round(max(0.0, min(calibrated, 1.0)) * 100))
    return score_0_100, breakdown


def detect_document_type(text: str) -> str:
    """Detect whether the text looks like a resume/CV or a social post.

    Returns: 'resume' or 'social'. Uses simple heuristics: section keywords, presence of email/phone, and typical resume headings.
    """
    ltext = text.lower()
    # resume indicators
    resume_markers = ["curriculum vitae", "resume", "objective", "experience", "education", "skills", "professional summary", "work experience"]
    contact_pattern = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone_pattern = re.search(r"\+?\d[\d \-()]{6,}\d", text)

    marker_score = sum(1 for m in resume_markers if m in ltext)
    if marker_score >= 2 or (contact_pattern and phone_pattern) or (marker_score >=1 and (contact_pattern or phone_pattern)):
        return "resume"
    # otherwise assume social short-form content if it's short and contains hashtags or mentions
    if len(text.split()) < 400 and ("#" in text or "@" in text or any(kw in ltext for kw in ["learn more", "sign up", "click", "download"])):
        return "social"
    # default to social for anything shorter; resumes are usually longer and structured
    return "social"


def analyze_resume_text(text: str) -> Dict[str, object]:
    """Analyze resume-style documents and return a resume-focused report.

    Produces fields: resume_score (0-100), checks, suggestions, and component breakdown.
    """
    words = simple_tokenize(text)
    word_count = len(words)
    char_count = len(text)

    email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))
    phone = bool(re.search(r"\+?\d[\d \-()]{6,}\d", text))
    has_contact = email or phone

    # Sections
    sections = {"education": bool(re.search(r"\beducation\b", text, re.I)),
                "experience": bool(re.search(r"\bexperience\b", text, re.I)),
                "skills": bool(re.search(r"\bskills\b", text, re.I)),
                "projects": bool(re.search(r"\bprojects\b", text, re.I)),
                }
    section_count = sum(1 for v in sections.values() if v)

    # Bullet/formatting density (approx)
    bullets = len(re.findall(r"^\s*[-•*]\s+", text, flags=re.M))

    # Action verbs presence
    action_verbs = ["managed", "developed", "led", "designed", "implemented", "improved", "achieved", "built", "created", "delivered"]
    action_count = sum(text.lower().count(v) for v in action_verbs)

    # Readability reuse
    read_score = readability_score(text)

    # Component scores (0-1)
    contact_score = 1.0 if has_contact else 0.0
    section_score = min(1.0, section_count / 3.0)
    format_score = min(1.0, bullets / 5.0) if bullets > 0 else 0.4
    action_score = min(1.0, action_count / 3.0)
    length_score = 1.0 if 300 <= word_count <= 1200 else max(0.2, min(1.0, word_count / 300.0))

    weights = {
        "contact": 0.25,
        "sections": 0.25,
        "format": 0.15,
        "action": 0.15,
        "readability": 0.1,
        "length": 0.1,
    }

    breakdown = {
        "contact": contact_score,
        "sections": section_score,
        "format": format_score,
        "action": action_score,
        "readability": read_score,
        "length": length_score,
    }

    total = sum(breakdown[k] * weights[k] for k in weights)
    resume_score = int(round(max(0.0, min(total, 1.0)) * 100))

    suggestions: List[str] = []
    if not has_contact:
        suggestions.append("Add clear contact information (email and phone) near the top of the resume.")
    if section_count < 2:
        suggestions.append("Add standard sections such as Experience, Education, and Skills to help recruiters scan your profile.")
    if bullets < 3:
        suggestions.append("Use bullet points for role responsibilities and achievements to improve scannability.")
    if action_count < 2:
        suggestions.append("Use stronger action verbs (managed, developed, led, designed) to describe achievements.")
    if read_score < 0.45:
        suggestions.append("Shorten long sentences and simplify wording to improve clarity.")

    return {
        "doc_type": "resume",
        "word_count": word_count,
        "char_count": char_count,
        "resume_score": resume_score,
        "breakdown": breakdown,
        "checks": {"contact": has_contact, "sections": sections, "bullets": bullets, "action_count": action_count},
        "suggestions": suggestions,
    }


def analyze_text(text: str) -> Dict[str, object]:
    """Main analysis router that detects document type and delegates to the appropriate analyzer."""
    doc_type = detect_document_type(text)
    if doc_type == "resume":
        return analyze_resume_text(text)

    # default: social analysis
    words = simple_tokenize(text)
    word_count = len(words)
    hashtags = count_hashtags(text)
    mentions = count_mentions(text)
    has_cta, cta_list = detect_cta(text)
    hook_score, hook_explanation = score_hook(text)
    readability = readability_score(text)

    analysis = {
        "doc_type": "social",
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
    """Render analysis results in a polished, readable layout.

    Automatically adapts UI for social posts or resumes based on analysis['doc_type'].
    """
    doc_type = analysis.get("doc_type", "social")
    st.subheader("Analysis Results")

    if doc_type == "resume":
        # Resume-focused UI
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Resume Score</div><div class='metric-value'>{analysis['resume_score']}/100</div></div>", unsafe_allow_html=True)
            st.progress(min(max(int(analysis['resume_score']), 0), 100))
            st.markdown("### Checks")
            checks = analysis.get("checks", {})
            st.write(f"Contact info: {'Yes' if checks.get('contact') else 'No'}")
            st.write(f"Sections found: {', '.join([k for k,v in analysis.get('breakdown',{}).items() if k in ['sections'] and v>0]) or 'See details below'}")

            with st.expander("Detailed resume breakdown"):
                st.write({
                    "word_count": analysis.get("word_count"),
                    "char_count": analysis.get("char_count"),
                    "action_verbs_used": analysis.get("checks", {}).get("action_count"),
                    "bullet_points": analysis.get("checks", {}).get("bullets"),
                    "component_scores": analysis.get("breakdown"),
                })

            st.markdown("### Suggestions")
            if analysis.get("suggestions"):
                st.markdown("<div class='suggestion-list'>", unsafe_allow_html=True)
                for s in analysis["suggestions"]:
                    st.markdown(f"- {s}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("No suggestions — resume looks good to our heuristic checks.")

        with right:
            st.markdown("### Quick resume tips")
            st.write("- Ensure contact info is prominent")
            st.write("- Use bulleted achievements with action verbs")
            st.write("- Keep length between ~300–1000 words for most roles")
            st.markdown("---")
            st.download_button("Download extracted text", data=original_text, file_name="resume_extracted.txt", mime="text/plain")

        st.markdown("---")
        st.markdown("### Extracted content preview")
        st.text_area("Extracted text", value=original_text, height=300)
        return

    # Default: social post UI
    col1, col2, col3 = st.columns([1.6, 1, 1])
    with col1:
        st.markdown(
            "<div class='metric-card'><div class='metric-title'>Engagement Score</div><div class='metric-value'>{}</div></div>".format(f"{analysis['engagement_score']}/100"),
            unsafe_allow_html=True,
        )
        st.progress(min(max(int(analysis['engagement_score']), 0), 100))
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
