"""Optional microphone input helpers for the workspace CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VoiceCaptureResult:
    """Outcome of one optional speech-to-text attempt."""

    available: bool
    text: str | None = None
    message: str | None = None


def capture_once(timeout_s: int = 5, phrase_time_limit_s: int = 30) -> VoiceCaptureResult:
    """Capture one utterance if `speech_recognition` is installed."""
    try:
        import speech_recognition as sr
    except ImportError:
        return VoiceCaptureResult(
            available=False,
            message="Voice mode requested but `speech_recognition` is not installed.",
        )

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout_s, phrase_time_limit=phrase_time_limit_s)
        text = recognizer.recognize_google(audio)
    except Exception as exc:  # noqa: BLE001
        return VoiceCaptureResult(available=True, message=f"Voice capture failed: {exc}")
    return VoiceCaptureResult(available=True, text=text)
