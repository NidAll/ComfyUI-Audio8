"""Sentence-aware long-form synthesis using the pack's ordinary synth path."""

import math
import re
import time

import torch


_SENTENCE_END = re.compile(r"[.!?…。！？؟۔…]+[\"'’”»）)\]}]*")
_CONTEXT_LIMIT_ERROR = re.compile(
    r"Text and reference use \d+ positions; the model context is \d+\."
)


def split_text(text, max_chars=240):
    """Split text into bounded pieces, preferring sentence and word boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while len(text) - start > max_chars:
        limit = start + max_chars
        sentence_ends = []
        for match in _SENTENCE_END.finditer(text, start, limit + 1):
            end = match.end()
            if end > limit:
                continue
            punctuation = match.group(0)[0]
            next_char = text[end:end + 1]
            # ASCII periods also occur in decimals and abbreviations. Treat
            # those as sentence ends only when whitespace or end follows.
            if (punctuation == "." and next_char and not next_char.isspace() and
                    not match.group(0)[1:]):
                continue
            sentence_ends.append(end)

        if sentence_ends:
            boundary = sentence_ends[-1]
        else:
            whitespace_starts = [index for index in range(start + 1, limit)
                                 if text[index].isspace()]
            boundary = whitespace_starts[-1] if whitespace_starts else limit

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = boundary
        while start < len(text) and text[start].isspace():
            start += 1

    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def _bisect_text(text):
    """Split a failed segment near its midpoint, retaining whole words where possible."""
    length = len(text)
    lower = max(1, length // 4)
    upper = min(length - 1, (3 * length) // 4)
    sentence_ends = []
    for match in _SENTENCE_END.finditer(text, 0, length):
        end = match.end()
        if not lower <= end <= upper:
            continue
        punctuation = match.group(0)[0]
        next_char = text[end:end + 1]
        if (punctuation == "." and next_char and not next_char.isspace() and
                not match.group(0)[1:]):
            continue
        sentence_ends.append(end)
    if sentence_ends:
        boundary = min(sentence_ends, key=lambda end: abs(end - length / 2))
    else:
        whitespace_starts = [index for index in range(lower, upper + 1) if text[index].isspace()]
        boundary = (min(whitespace_starts, key=lambda index: abs(index - length / 2))
                    if whitespace_starts else length // 2)
    left = text[:boundary].strip()
    right = text[boundary:].strip()
    return [part for part in (left, right) if part]


def _sample_rate(audio):
    rate = audio.get("sample_rate") if isinstance(audio, dict) else None
    if isinstance(rate, bool):
        raise ValueError("Synthesis returned an invalid sample rate.")
    try:
        numeric_rate = float(rate)
    except (TypeError, ValueError):
        raise ValueError("Synthesis returned an invalid sample rate.") from None
    if not math.isfinite(numeric_rate) or numeric_rate <= 0 or not numeric_rate.is_integer():
        raise ValueError("Synthesis returned an invalid sample rate.")
    return int(numeric_rate)


def _finished(stats):
    values = stats.get("finished") if isinstance(stats, dict) else None
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return bool(values[0])
    raise ValueError("Synthesis stats must contain one finished flag.")


def _successful_waveform(audio, expected_rate, expected_channels):
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Synthesis must return standard AUDIO with waveform and sample_rate.")
    rate = _sample_rate(audio)
    if expected_rate is not None and rate != expected_rate:
        raise ValueError("Long-form chunks returned different sample rates.")

    waveform = audio["waveform"]
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or
            waveform.shape[0] != 1 or waveform.shape[1] < 1 or waveform.shape[-1] < 1):
        raise ValueError("Each synthesis result must contain one nonempty AUDIO sample [1, channels, samples].")
    if expected_channels is not None and waveform.shape[1] != expected_channels:
        raise ValueError("Long-form chunks returned different channel counts.")

    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("Synthesis returned nonfinite audio.")
    return waveform, rate


def synthesize_long(synthesize_one, text, *, chunk_chars=240, pause_ms=100, seed=42,
                    max_new_tokens=768, callback=None, **kwargs):
    """Synthesize text in complete chunks, retrying unfinished chunks smaller."""
    if not callable(synthesize_one):
        raise TypeError("synthesize_one must be callable")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    try:
        pause_ms = float(pause_ms)
    except (TypeError, ValueError):
        raise ValueError("pause_ms must be a finite nonnegative number") from None
    if not math.isfinite(pause_ms) or pause_ms < 0:
        raise ValueError("pause_ms must be a finite nonnegative number")

    initial_chunks = split_text(text, chunk_chars)
    if not initial_chunks:
        raise ValueError("Enter nonempty text to synthesize.")

    started = time.perf_counter()
    attempt_count = 0
    discarded_attempts = 0
    expected_rate = None
    expected_channels = None
    segments = []

    def retry_smaller(chunk, cause=None):
        if len(chunk) <= 1:
            if cause is not None:
                raise RuntimeError(
                    "Audio8 context is too small for this text and reference. "
                    "Shorten the reference or simplify the text."
                ) from cause
            raise RuntimeError(
                "Audio8 could not finish a one-character text segment. "
                "Increase max_new_tokens or simplify the text."
            )
        smaller = _bisect_text(chunk)
        if len(smaller) < 2 or any(len(part) >= len(chunk) for part in smaller):
            message = "Audio8 could not split the failed text segment into smaller pieces."
            raise RuntimeError(message) from cause
        for part in smaller:
            synthesize_chunk(part)

    def synthesize_chunk(chunk):
        nonlocal attempt_count, discarded_attempts, expected_rate, expected_channels
        attempt_seed = (seed + attempt_count) % (1 << 64)
        attempt_count += 1
        try:
            audio, stats = synthesize_one([chunk], seed=attempt_seed,
                                          max_new_tokens=max_new_tokens, callback=callback, **kwargs)
        except ValueError as error:
            if _CONTEXT_LIMIT_ERROR.fullmatch(str(error)) is None:
                raise
            discarded_attempts += 1
            retry_smaller(chunk, error)
            return
        if not _finished(stats):
            discarded_attempts += 1
            retry_smaller(chunk)
            return

        waveform, rate = _successful_waveform(audio, expected_rate, expected_channels)
        expected_rate = rate
        expected_channels = waveform.shape[1]
        segments.append({"text": chunk, "waveform": waveform, "seed": attempt_seed,
                         "stats": dict(stats)})

    for chunk in initial_chunks:
        synthesize_chunk(chunk)

    pause_samples = round(pause_ms * expected_rate / 1000)
    pause = torch.zeros((1, expected_channels, pause_samples), dtype=torch.float32)
    pieces = []
    chunk_metrics = []
    samples = []
    audio_seconds = []
    sample_cursor = 0
    for index, segment in enumerate(segments):
        if index:
            pieces.append(pause)
            sample_cursor += pause_samples
        waveform = segment["waveform"]
        count = waveform.shape[-1]
        sample_start = sample_cursor
        pieces.append(waveform)
        sample_cursor += count
        samples.append(count)
        audio_seconds.append(count / expected_rate)
        chunk_metrics.append({"text": segment["text"], "sample_start": sample_start,
                              "sample_end": sample_cursor, "seed": segment["seed"],
                              "stats": segment["stats"]})

    output = {"waveform": torch.cat(pieces, dim=-1), "sample_rate": expected_rate}
    return output, {
        "chunk_count": len(segments),
        "total_seconds": time.perf_counter() - started,
        "output_seconds": sample_cursor / expected_rate,
        "output_samples": sample_cursor,
        "audio_seconds": audio_seconds,
        "samples": samples,
        "finished": [True] * len(segments),
        "chunks": chunk_metrics,
        "discarded_attempts": discarded_attempts,
    }
