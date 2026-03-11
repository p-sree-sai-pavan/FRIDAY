"""
io/ears.py
==========
FRIDAY Voice Input — Microphone capture, VAD, and Speech-to-Text.

ARCHITECTURE:
  capture  → sounddevice (cross-platform mic access)
  VAD      → webrtcvad (Google's voice activity detector — lightweight, low latency)
  STT      → Online:  Groq Whisper API  (whisper-large-v3, fast + accurate)
             Offline: faster-whisper    (local Whisper on Pavan's RTX 4060)

FLOW:
  listen() is the single public coroutine.
  1. Opens mic at 16kHz (required by both webrtcvad and Whisper)
  2. Waits for speech to start (VAD detects voiced frames)
  3. Collects audio until SILENCE_CHUNKS consecutive silent frames
  4. Sends accumulated audio to STT → returns transcript string

USAGE:
  from io.ears import listen
  text = await listen()     # blocks until Pavan finishes speaking
  if text:
      result = await orchestrate(text, session)

DEPENDENCIES:
  pip install sounddevice webrtcvad numpy
  Online:  (groq already in requirements)
  Offline: pip install faster-whisper
"""

import asyncio
import io
import logging
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

log = logging.getLogger("ears")

# ========================
# AUDIO CONSTANTS
# webrtcvad only accepts: 8000, 16000, 32000, 48000 Hz
# Frame duration must be exactly 10, 20, or 30 ms
# ========================
SAMPLE_RATE       = config.SAMPLE_RATE                              # 16000 Hz
FRAME_DURATION_MS = 30                                               # 30ms frames — most reliable
FRAME_SIZE        = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)     # 480 samples
CHANNELS          = 1
SILENCE_CHUNKS    = config.SILENCE_CHUNKS                           # 50 silent frames = 1.5s silence
MIN_SPEECH_FRAMES = 5                                                # discard short noise bursts


# ========================
# HELPERS
# ========================

def _frames_to_wav_bytes(frames: list[bytes]) -> bytes:
    """Pack raw 16-bit PCM frames into an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 16-bit PCM = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


# ========================
# RECORDING  (blocking — always run in asyncio.to_thread)
# ========================

def _record_until_silence() -> bytes | None:
    """
    Opens the microphone and records until SILENCE_CHUNKS consecutive
    silent frames are detected after speech has started.

    State machine:
      IDLE      → rolling 10-frame pre-buffer (keeps start of utterance)
      RECORDING → speech detected, accumulating frames
      DONE      → silence threshold reached, return audio

    Returns raw WAV bytes, or None if no speech was captured.
    """
    try:
        import sounddevice as sd
        import webrtcvad
    except ImportError as e:
        log.error(f"[Ears] Missing dependency: {e}. Run: pip install sounddevice webrtcvad")
        return None

    # VAD aggressiveness: 0 (least), 1, 2, 3 (most aggressive filtering)
    # VAD_SENSITIVITY 0.5 → aggressiveness 1 — catches clear speech, ignores most background noise
    vad            = webrtcvad.Vad(min(3, max(0, int(config.VAD_SENSITIVITY * 3))))
    speech_frames: list[bytes] = []
    pre_buffer:    list[bytes] = []   # rolling buffer to preserve utterance start
    silence_count  = 0
    speech_count   = 0
    speech_started = False

    log.debug("[Ears] Microphone open — waiting for speech...")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=FRAME_SIZE,
        ) as stream:

            while True:
                raw_frame, _ = stream.read(FRAME_SIZE)
                pcm_bytes     = bytes(raw_frame)

                # webrtcvad is strict — frame must be exactly (FRAME_SIZE * 2) bytes
                expected = FRAME_SIZE * 2
                if len(pcm_bytes) != expected:
                    continue

                is_speech = vad.is_speech(pcm_bytes, SAMPLE_RATE)

                if not speech_started:
                    if is_speech:
                        speech_count += 1
                        pre_buffer.append(pcm_bytes)
                        if speech_count >= MIN_SPEECH_FRAMES:
                            # Confirmed speech — promote pre-buffer to speech_frames
                            speech_frames  = list(pre_buffer)
                            pre_buffer     = []
                            speech_started = True
                            log.debug("[Ears] Speech started — recording")
                    else:
                        speech_count = 0
                        # Keep rolling 10-frame pre-buffer to avoid clipping the start
                        pre_buffer.append(pcm_bytes)
                        if len(pre_buffer) > 10:
                            pre_buffer.pop(0)

                else:  # speech_started
                    speech_frames.append(pcm_bytes)

                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= SILENCE_CHUNKS:
                            log.debug(f"[Ears] Silence end — {len(speech_frames)} frames recorded")
                            break

    except Exception as e:
        log.error(f"[Ears] Microphone stream error: {e}")
        return None

    if not speech_started:
        return None

    return _frames_to_wav_bytes(speech_frames)


# ========================
# STT: ONLINE — Groq Whisper
# ========================

async def _transcribe_online(wav_bytes: bytes) -> str:
    """
    Transcribe via Groq's Whisper API (whisper-large-v3).
    Fast: typically <1s round-trip. Handles accented English well.
    """
    try:
        from groq import AsyncGroq
        client     = AsyncGroq(api_key=config.GROQ_API_KEY or "")
        audio_file = ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")

        result = await client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            language="en",
            response_format="text"
        )

        # Groq returns a plain string when response_format="text"
        text = result.strip() if isinstance(result, str) else (getattr(result, "text", "") or "").strip()
        log.info(f"[Ears] Groq Whisper → \"{text}\"")
        return text

    except Exception as e:
        log.error(f"[Ears] Groq Whisper failed: {e}")
        return ""


# ========================
# STT: OFFLINE — faster-whisper
# ========================

_fw_model = None   # loaded once, reused — GPU stays warm


async def _transcribe_offline(wav_bytes: bytes) -> str:
    """
    Transcribe locally using faster-whisper on Pavan's RTX 4060.
    Model: 'base' — 74M params, real-time on GPU, ~140MB VRAM.
    Upgrade to 'small' (244M) or 'medium' (769M) for more accuracy.
    """
    global _fw_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error("[Ears] faster-whisper not installed. Run: pip install faster-whisper")
        return ""

    try:
        if _fw_model is None:
            log.info("[Ears] Loading faster-whisper 'base' model onto GPU...")
            # float16 is optimal for NVIDIA RTX — halves VRAM vs float32 with no accuracy loss
            _fw_model = await asyncio.to_thread(
                WhisperModel, "base", device="cuda", compute_type="float16"
            )
            log.info("[Ears] faster-whisper ready")

        audio_buf = io.BytesIO(wav_bytes)

        def _run():
            segments, _ = _fw_model.transcribe(
                audio_buf,
                language="en",
                vad_filter=True,    # skip silent parts inside the clip
                beam_size=5         # higher = more accurate, slightly slower
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        text = await asyncio.to_thread(_run)
        log.info(f"[Ears] faster-whisper → \"{text}\"")
        return text

    except Exception as e:
        log.error(f"[Ears] faster-whisper failed: {e}")
        return ""


# ========================
# PUBLIC API
# ========================

async def listen() -> str:
    """
    Capture microphone input and return the transcribed text.

    Blocks until speech is detected and then silence follows.
    Returns the transcribed string, or "" if nothing was captured.

    Usage in main.py REPL:
        text = await listen()
        if text:
            result = await orchestrate(text, session)
    """
    log.debug(f"[Ears] listen() called | mode={config.MODEL_MODE}")

    # _record_until_silence is blocking — always run in a thread
    wav_bytes = await asyncio.to_thread(_record_until_silence)
    if not wav_bytes:
        return ""

    if config.MODEL_MODE == "online":
        return await _transcribe_online(wav_bytes)
    else:
        return await _transcribe_offline(wav_bytes)


async def check_mic_available() -> tuple[bool, str]:
    """
    Check if a working microphone is available.
    Called during startup if voice mode is enabled.
    Returns (available: bool, message: str).
    """
    try:
        import sounddevice as sd
        input_devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if not input_devs:
            return False, "No microphone found — check your audio settings"
        default = sd.query_devices(kind="input")
        return True, f"Microphone ready: {default['name']}"
    except ImportError:
        return False, "sounddevice not installed (pip install sounddevice)"
    except Exception as e:
        return False, f"Microphone check failed: {e}"
