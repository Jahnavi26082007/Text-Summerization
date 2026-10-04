import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


st.set_page_config(
    page_title="Text Summarization",
    page_icon="📑",
    layout="wide"
)


MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}


LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}


CHUNK_WORDS = 600


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


def chunk_text(text, size=CHUNK_WORDS):
    words = text.split()

    chunks = []

    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i + size]))

    return chunks


def summarize_chunk(text, tokenizer, model, min_len, max_len):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    summary = tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )

    return summary


def summarize_text(text, tokenizer, model, min_len, max_len):

    chunks = chunk_text(text)

    progress = st.progress(
        0.0,
        text="Summarizing..."
    )

    parts = []

    for i, chunk in enumerate(chunks, 1):

        words = len(chunk.split())

        chunk_min_len = min(
            min_len,
            max(10, words // 2)
        )

        result = summarize_chunk(
            chunk,
            tokenizer,
            model,
            chunk_min_len,
            max_len
        )

        parts.append(result)

        progress.progress(
            i / len(chunks),
            text=f"Summarizing chunk {i}/{len(chunks)}..."
        )

    progress.empty()

    combined = " ".join(parts)

    return combined


def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# Sidebar
st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys())
)

length = st.sidebar.radio(
    "Summary Length",
    list(LENGTHS.keys()),
    index=1
)


# Main page
st.title("📑 Text Summarization")

st.write(
    "Paste text or upload a file (.txt / PDF) "
    "to get a concise summary."
)


source = st.radio(
    "Input type",
    ["Paste text", "Upload file"],
    horizontal=True
)


text = ""


# Paste text
if source == "Paste text":

    text = st.text_area(
        "Your text",
        height=300,
        placeholder="Paste an article, notes, or report here..."
    )


# Upload file
else:

    file = st.file_uploader(
        "Upload .txt or .pdf",
        type=["txt", "pdf"]
    )

    if file:

        if file.type == "application/pdf":
            text = read_pdf(file)

        else:
            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

        with st.expander("Preview extracted text"):

            st.write(
                text[:3000]
                + ("..." if len(text) > 3000 else "")
            )


# Summarize button
if st.button("✨ Summarize", type="primary"):

    word_count = len(text.split())

    if word_count < 50:

        st.warning(
            "Please provide at least 50 words of text."
        )

    else:

        tokenizer, model = load_model(
            MODELS[model_name]
        )

        min_len, max_len = LENGTHS[length]

        summary = summarize_text(
            text,
            tokenizer,
            model,
            min_len,
            max_len
        )

        col1, col2 = st.columns(2)


        # Original
        with col1:

            st.subheader("Original")

            st.caption(
                f"{word_count} words"
            )

            st.write(
                text[:5000]
                + ("..." if len(text) > 5000 else "")
            )


        # Summary
        with col2:

            st.subheader("Summary")

            summary_count = len(
                summary.split()
            )

            reduction = (
                100
                - (summary_count * 100 / word_count)
            )

            st.caption(
                f"{summary_count} words. "
                f"{reduction:.1f}% shorter"
            )

            st.success(summary)

            st.download_button(
                "⬇️ Download summary",
                summary,
                "summary.txt",
                "text/plain"
            )