import streamlit as st
import numpy as np
import torch
import librosa
import io
import soundfile as sf
import torchaudio
from faster_whisper import WhisperModel
import noisereduce as nr
import speech_recognition as sr
from nltk.corpus import cmudict
import Levenshtein
import nltk

# Download phoneme dictionary
nltk.download("cmudict")
prondict = cmudict.dict()

# Load ASR model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("base", device=device, compute_type="int8")

# Streamlit UI
st.set_page_config(page_title="Ingeleze Advanced AI Solution for English Mastery", layout="centered")
st.title("🎙 Ingeleze Advanced AI Solution for English Mastery")
st.write("Upload an audio file to analyze speech pronunciation.")

# File uploader
audio_file = st.file_uploader("🔼 Upload an audio file", type=["mp3", "wav", "flac"])

def load_audio(audio_data, file_format):
    """Load audio file using torchaudio (without pydub)."""
    audio_bytes = io.BytesIO(audio_data)
    waveform, sample_rate = torchaudio.load(audio_bytes)
    return waveform.numpy().flatten(), sample_rate

def remove_noise(audio, sample_rate):
    """Reduce noise from the audio."""
    return nr.reduce_noise(y=audio, sr=sample_rate), sample_rate

def transcribe_audio(audio, sample_rate):
    """Transcribe speech using Faster Whisper."""
    try:
        temp_filename = "temp.wav"
        sf.write(temp_filename, audio, sample_rate)
        segments, _ = whisper_model.transcribe(temp_filename)
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        return f"Error in transcription: {str(e)}"

def get_audio_phonemes(audio_data):
    """Extract phonemes using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    
    with io.BytesIO(audio_data) as audio_file:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        words = text.split()
        phonemes = [prondict[word.lower()][0] for word in words if word.lower() in prondict]
        return [" ".join(phoneme_list) for phoneme_list in phonemes]  # Convert list of phonemes to strings
    except sr.UnknownValueError:
        return []

def compare_phonemes(word, user_phonemes):
    """Compare phonemes between expected and user speech."""
    if word.lower() in prondict:
        correct_phonemes = prondict[word.lower()][0]  # Get first phoneme sequence
        correct_phonemes_str = " ".join(correct_phonemes)  # Convert to string
        user_phonemes_str = " ".join(user_phonemes)  # Convert to string
        
        return Levenshtein.ratio(correct_phonemes_str, user_phonemes_str) * 100
    return 0

def calculate_fluency(audio, sample_rate):
    """Estimate fluency using words per minute (WPM)."""
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    avg_syllables_per_word = 1.4
    words_spoken = len(audio) / (sample_rate * avg_syllables_per_word)
    wpm = (words_spoken / duration) * 60
    return min(100, (wpm / 150) * 100)

def detect_pauses(audio, sample_rate):
    """Detect excessive pauses in speech."""
    non_silent_intervals = librosa.effects.split(audio, top_db=20)
    return max(0, 100 - ((len(non_silent_intervals) / 10) * 100))

def analyze_intonation(audio, sample_rate):
    """Analyze pitch variation for intonation scoring."""
    pitches, _ = librosa.piptrack(y=audio, sr=sample_rate)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        return 0
    return min(100, (np.std(pitches) / 50) * 100)

def calculate_final_audio_score(audio, sample_rate, expected_text):
    """Compute the final speech assessment score."""
    # Convert numpy audio array to WAV bytes for phoneme extraction
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio, sample_rate, format="WAV")
        wav_bytes = wav_buffer.getvalue()
    
    user_phonemes = get_audio_phonemes(wav_bytes)  # Fixed phoneme extraction issue
    pronunciation_scores = [compare_phonemes(word, user_phonemes) for word in expected_text.split()]
    pronunciation_score = sum(pronunciation_scores) / len(pronunciation_scores) if pronunciation_scores else 0
    fluency_score = calculate_fluency(audio, sample_rate)
    pause_score = detect_pauses(audio, sample_rate)
    intonation_score = analyze_intonation(audio, sample_rate)
    
    final_score = (0.4 * pronunciation_score + 0.2 * fluency_score +
                   0.15 * pause_score + 0.15 * intonation_score)
    
    return final_score, pronunciation_score, fluency_score, pause_score, intonation_score

def generate_audio_feedback(final_score):
    """Generate feedback based on the final score."""
    if final_score > 85:
        return "✅ Excellent pronunciation and fluency! Keep up the great work."
    elif final_score > 70:
        return "👍 Good job! Try to reduce pauses and improve stress patterns."
    else:
        return "❌ Needs improvement. Focus on pronunciation and intonation."

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    st.success("✅ Audio file uploaded successfully!")
    
    # Get file format
    file_format = audio_file.type.split("/")[-1]
    
    # Load & process audio
    audio_data, sample_rate = load_audio(audio_file.read(), file_format)
    cleaned_audio, sample_rate = remove_noise(audio_data, sample_rate)

    # Transcription
    transcription = transcribe_audio(cleaned_audio, sample_rate)
    st.write("📝 **Transcription:**", transcription)
    
    # Expected sentence for pronunciation assessment
    expected_text = "a long time ago an old man lived in a small village"

    # Compute Scores
    final_score, pronunciation_score, fluency_score, pause_score, intonation_score = calculate_final_audio_score(cleaned_audio, sample_rate, expected_text)

    # Feedback
    feedback = generate_audio_feedback(final_score)

    # Display Scores
    st.subheader("📊 Speech Analysis Scores")
    st.write(f"**Pronunciation Score:** {pronunciation_score:.2f}")
    st.write(f"**Fluency Score:** {fluency_score:.2f}")
    st.write(f"**Pause Score:** {pause_score:.2f}")
    st.write(f"**Intonation Score:** {intonation_score:.2f}")
    st.write(f"🎯 **Final Pronunciation Score:** {final_score:.2f}")

    # Display Feedback
    st.subheader("🗣 Feedback")
    st.write(feedback)
