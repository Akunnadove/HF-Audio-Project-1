
import streamlit as st
import torch
from transformers import pipeline
import tempfile
import os

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(page_title="Audio Describer", layout="centered")
st.title("🎧 Audio Describer")
st.markdown("Upload an audio file, click the button, and get transcription with summary.")

# Upload audio
audio_file = st.file_uploader("Upload an audio file (.mp3, .ogg, .wav)", type=["mp3", "ogg", "wav"])

# Load Whisper once
@st.cache_resource
def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        device=device,
        generate_kwargs={"language": "English", "task": "translate"}
    )

# Load summarizer once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == 'cuda' else -1)

if audio_file:
    st.audio(audio_file, format="audio/wav")

    if st.button("📝 Describe Audio"):
        with st.spinner("Transcribing... please wait"):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            # Run transcription
            whisper_pipe = load_whisper()
            transcription = whisper_pipe(tmp_path)

            # Build formatted output
            formatted_lyrics = ""
            for line in transcription['chunks']:
                text = line["text"]
                ts = line["timestamp"]
                formatted_lyrics += f"{ts} --> {text}\n"

            full_text = " ".join(chunk["text"] for chunk in transcription["chunks"])
            os.remove(tmp_path)

        st.subheader("📄 Transcription with Timestamps")
        st.text_area("Transcription", formatted_lyrics.strip(), height=300)

        with st.spinner("Summarizing..."):
            summarizer = load_summarizer()
            summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

        st.subheader("🧠 Summary")
        st.success(summary)
