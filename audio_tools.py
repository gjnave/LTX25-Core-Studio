"""Small, phone-friendly audio selection helpers shared by UI and worker."""

from __future__ import annotations

import math
from pathlib import Path

import soundfile as sf


def parse_time(value, label: str, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    parts = str(value).strip().split(":")
    if len(parts) > 3:
        raise ValueError(f"{label}: use seconds, mm:ss, or hh:mm:ss.")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        else:
            leading = [int(part) for part in parts[:-1]]
            if any(part < 0 or part >= 60 for part in leading[1:]):
                raise ValueError
            if not 0 <= float(parts[-1]) < 60:
                raise ValueError
            seconds = float(parts[-1])
            for index, part in enumerate(reversed(leading), start=1):
                seconds += part * 60**index
    except ValueError as error:
        raise ValueError(f"{label}: use seconds, mm:ss, or hh:mm:ss.") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{label} must be a nonnegative time.")
    return seconds


def audio_info(path: str | Path) -> tuple[int, float]:
    with sf.SoundFile(str(path)) as source:
        return source.samplerate, len(source) / source.samplerate


def read_selection(
    path: str | Path, start, end, *, max_seconds: float | None = None,
):
    """Return selected samples as [samples, channels], sample rate, and range."""
    with sf.SoundFile(str(path)) as source:
        sample_rate = source.samplerate
        total_seconds = len(source) / sample_rate
        start_seconds = parse_time(start, "Audio start", 0.0)
        end_seconds = parse_time(end, "Audio end", total_seconds)
        if start_seconds >= total_seconds:
            raise ValueError(f"Audio start is past the file's {total_seconds:.2f}-second end.")
        if end_seconds > total_seconds + 0.02:
            raise ValueError(f"Audio end exceeds the file's {total_seconds:.2f}-second length.")
        end_seconds = min(end_seconds, total_seconds)
        if end_seconds <= start_seconds:
            raise ValueError("Audio end must be later than audio start.")
        if max_seconds is not None:
            end_seconds = min(end_seconds, start_seconds + max_seconds)
        first = round(start_seconds * sample_rate)
        last = min(len(source), round(end_seconds * sample_rate))
        if last <= first:
            raise ValueError("The selected audio range is too short.")
        source.seek(first)
        samples = source.read(last - first, dtype="float32", always_2d=True)
    return samples, sample_rate, (first / sample_rate, last / sample_rate, total_seconds)
