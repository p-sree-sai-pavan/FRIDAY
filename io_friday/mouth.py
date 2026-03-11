"""
io/mouth.py
===========
FRIDAY Voice Output — Text-to-Speech.

ARCHITECTURE:
  Online  → edge-tts  (Microsoft Neural TTS — free, no API key, high quality,
                        uses the same engine as Windows 11 Narrator)
  Offline → pyttsx3   (Windows SAPI — built into Windows, no internet needed,
                        decent quality, zero latency)

  Playback in both cases: sounddevice (consistent cross-platform audio output)
  edge-tts produces MP3 → decoded with pydub → played via sounddevice
  pyttsx3 writes WAV → played via sounddevice

VOICE SELECTION:
  Online:  ONLINE_VOICE  — Microsoft Jenny (natural, clear, neutral US accent)
  Offline: OFFLINE_VOICE — Windows SAPI en-US voice (whatever is installed)
  Both can be overridden in config.py

USAGE:
  from io.mouth import speak
  await speak("Hello Pavan, how can I help you?")
  await speak("...")       # queues behind current speech
  await stop_speaking()    # interrupt current speech

DEPENDENCIES:
  pip install edge-tts pydub sounddevice
  Offline: pyttsx3 (pip install pyttsx3) — Windows only
  pydub also needs ffmpeg on PATH for MP3 decode:
    winget install ffmpeg   OR   choco install ffmpeg
"""

import asyncio
import io
import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

log = logging.getLogger("mouth")

# ========================
# VOICE CONFIG
# ========================
# edge-tts voices: run `edge-tts --list-voices` to see all options
# Good alternatives: en-US-AriaNeural, en-US-GuyNeural, en-GB-SoniaNeural
ONLINE_VOICE  = "en-US-JennyNeural"

# pyttsx3 voice: None = use system default, or set to a voice name substring
OFFLINE_VOICE = None   # e.g. "Zira" for Microsoft Zira on Windows

# Speech rate adjustments
ONLINE_RATE   = "+0%"    # edge-tts rate: "+10%" faster, "-10%" slower
OFFLINE_RATE  = 175      # pyttsx3 words-per-minute (default ~200)

# ========================
# PLAYBACK STATE
# ========================
_speaking       = False          # True while audio is playing
_stop_requested = False          # set by stop_speaking() to interrupt


# ========================
# ONLINE TTS — edge-tts
# ========================

async def _speak_online(text: str) -> None:
    """
    Generate speech using Microsoft edge-tts and play it.

    edge-tts streams MP3 audio from Microsoft's neural TTS service.
    We collect the full audio, decode to PCM, then play via sounddevice
    so we can honour stop_speaking() without leaving dangling streams.
    """
    global _speaking, _stop_requested

    try:
        import edge_tts
        import sounddevice as sd
    except ImportError as e:
        log.error(f"[Mouth] Missing dependency: {e}. Run: pip install edge-tts sounddevice")
        return

    try:
        communicate = edge_tts.Communicate(text, ONLINE_VOICE, rate=ONLINE_RATE)

        # Collect all audio bytes from the async generator
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if _stop_requested:
                log.debug("[Mouth] Stop requested during generation — aborting")
                return
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        if not audio_chunks or _stop_requested:
            return

        raw_mp3 = b"".join(audio_chunks)

        # Decode MP3 → PCM using pydub
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_mp3(io.BytesIO(raw_mp3))
            # Normalise to 16kHz mono for consistent playback
            segment   = segment.set_frame_rate(config.SAMPLE_RATE).set_channels(1)
            pcm_array = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        except ImportError:
            log.error("[Mouth] pydub not installed. Run: pip install pydub  (also needs ffmpeg)")
            return
        except Exception as e:
            log.error(f"[Mouth] MP3 decode failed: {e}")
            return

        if _stop_requested:
            return

        _speaking = True
        try:
            # blocking_mode=False lets us check _stop_requested during playback
            # We play in chunks of 0.1s so stop has low latency
            chunk_samples = int(config.SAMPLE_RATE * 0.1)
            with sd.OutputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32"
            ) as stream:
                for start in range(0, len(pcm_array), chunk_samples):
                    if _stop_requested:
                        break
                    chunk = pcm_array[start : start + chunk_samples]
                    stream.write(chunk)
        finally:
            _speaking = False

    except Exception as e:
        log.error(f"[Mouth] edge-tts playback error: {e}")
        _speaking = False


# ========================
# OFFLINE TTS — pyttsx3
# ========================

def _speak_offline_blocking(text: str) -> None:
    """
    Generate and play speech using pyttsx3 (Windows SAPI).
    Blocking — always called via asyncio.to_thread.

    pyttsx3 directly drives the Windows speech engine, so it:
      - Works with no internet
      - Has zero decode overhead
      - Sounds decent with modern Windows voices (Aria, Jenny, Zira)
    """
    global _speaking, _stop_requested

    try:
        import pyttsx3
    except ImportError:
        log.error("[Mouth] pyttsx3 not installed. Run: pip install pyttsx3")
        return

    if _stop_requested:
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", OFFLINE_RATE)

        if OFFLINE_VOICE:
            voices = engine.getProperty("voices")
            for v in voices:
                if OFFLINE_VOICE.lower() in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break

        _speaking = True
        try:
            # Save to a temp WAV file and play via sounddevice for consistent
            # output handling and stop support — avoids pyttsx3's own blocking loop
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            if _stop_requested:
                return

            import sounddevice as sd
            import wave

            with wave.open(tmp_path, "rb") as wf:
                rate        = wf.getframerate()
                sample_width = wf.getsampwidth()
                n_channels  = wf.getnchannels()
                raw_data    = wf.readframes(wf.getnframes())

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype     = dtype_map.get(sample_width, np.int16)
            audio     = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

            # Normalise based on dtype range
            audio /= float(np.iinfo(dtype).max)

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            chunk_samples = int(rate * 0.1)
            with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as stream:
                for start in range(0, len(audio), chunk_samples):
                    if _stop_requested:
                        break
                    stream.write(audio[start : start + chunk_samples])

        finally:
            _speaking = False
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        log.error(f"[Mouth] pyttsx3 speech error: {e}")
        _speaking = False


# ========================
# PUBLIC API
# ========================

async def speak(text: str) -> None:
    """
    Convert text to speech and play it through the speakers.

    Respects the active MODEL_MODE:
      Online  → edge-tts  (Microsoft Neural TTS, natural voice)
      Offline → pyttsx3   (Windows SAPI, no internet)

    If FRIDAY is already speaking when this is called, the current
    speech is stopped first (interrupt model — always responds immediately).

    Usage:
        await speak("Opening Spotify for you.")
    """
    global _stop_requested

    # Sanitise — strip markdown that sounds bad when spoken
    clean = _clean_for_speech(text)
    if not clean:
        return

    # Interrupt any ongoing speech immediately
    if _speaking:
        _stop_requested = True
        await asyncio.sleep(0.12)   # give the playback loop time to notice

    _stop_requested = False
    log.debug(f"[Mouth] Speaking ({config.MODEL_MODE}): \"{clean[:60]}\"")

    if config.MODEL_MODE == "online":
        await _speak_online(clean)
    else:
        await asyncio.to_thread(_speak_offline_blocking, clean)


async def stop_speaking() -> None:
    """
    Interrupt and stop the current speech immediately.
    Safe to call even if nothing is playing.
    """
    global _stop_requested
    _stop_requested = True
    log.debug("[Mouth] Stop requested")


def is_speaking() -> bool:
    """Returns True if speech is currently playing."""
    return _speaking


# ========================
# TEXT CLEANING
# ========================

def _clean_for_speech(text: str) -> str:
    """
    Strip markdown and other formatting that sounds bad when spoken.
    e.g. "**bold**" → "bold",  "# Heading" → "Heading"
    """
    import re

    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove inline code and code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove markdown links — keep the display text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    # Remove bullet/numbered list markers
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines into a single pause (period)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


async def check_tts_available() -> tuple[bool, str]:
    """
    Check if TTS dependencies are installed for the current mode.
    Called during startup if voice mode is enabled.
    Returns (available: bool, message: str).
    """
    if config.MODEL_MODE == "online":
        try:
            import edge_tts  # noqa: F401
            import pydub      # noqa: F401
            return True, "edge-tts ready (Microsoft Neural TTS)"
        except ImportError as e:
            missing = str(e).split("'")[1]
            return False, f"Missing: {missing}. Run: pip install edge-tts pydub"
    else:
        try:
            import pyttsx3  # noqa: F401
            return True, "pyttsx3 ready (Windows SAPI)"
        except ImportError:
            return False, "pyttsx3 not installed. Run: pip install pyttsx3"
