import pytest

torch = pytest.importorskip("torch")

from audio8.longform import split_text, synthesize_long


def _visible(text):
    return "".join(text.split())


def test_split_text_prefers_sentence_ends_and_preserves_content():
    text = "First sentence. Second sentence! Third one? Fourth sentence."

    chunks = split_text(text, max_chars=20)

    assert chunks == ["First sentence.", "Second sentence!", "Third one?", "Fourth sentence."]
    assert all(0 < len(chunk) <= 20 for chunk in chunks)
    assert _visible("".join(chunks)) == _visible(text)


def test_split_text_handles_cjk_arabic_punctuation_and_paragraphs():
    text = "你好。世界！مرحبا؟كيف الحال۔\n\nA second paragraph ends here."

    chunks = split_text(text, max_chars=12)
    cjk_chunks = split_text("你好。世界！", max_chars=3)
    arabic_chunks = split_text("مرحبا؟كيف الحال۔", max_chars=8)

    assert all(len(chunk) <= 12 for chunk in chunks)
    assert cjk_chunks == ["你好。", "世界！"]
    assert "مرحبا؟" in arabic_chunks
    assert _visible("".join(chunks)) == _visible(text)


def test_split_text_bounds_long_unspaced_strings_and_handles_empty_text():
    text = "supercalifragilisticexpialidocious" * 3

    chunks = split_text(text, max_chars=13)

    assert len(chunks) > 3
    assert all(len(chunk) <= 13 for chunk in chunks)
    assert "".join(chunks) == text
    assert split_text(" \n\t ") == []


@pytest.mark.parametrize("max_chars", [0, -1, True, 1.5])
def test_split_text_rejects_invalid_chunk_sizes(max_chars):
    with pytest.raises(ValueError, match="positive integer"):
        split_text("text", max_chars)


def test_synthesize_long_retries_incomplete_chunks_and_keeps_only_finished_audio():
    reference = object()
    callback = object()
    calls = []

    def fake_synthesize(texts, *, seed, max_new_tokens, callback, **kwargs):
        text = texts[0]
        finished = len(text) <= 3
        calls.append({"text": text, "seed": seed, "callback": callback,
                      "max_new_tokens": max_new_tokens, "kwargs": kwargs,
                      "finished": finished})
        value = 9.0 if not finished else float(len(calls))
        waveform = torch.full((1, 1, len(text)), value)
        return {"waveform": waveform, "sample_rate": 100}, {
            "finished": [finished], "frames": [len(text)], "samples": [len(text)],
        }

    audio, stats = synthesize_long(
        fake_synthesize,
        "alpha beta!",
        chunk_chars=11,
        pause_ms=20,
        seed=(1 << 64) - 2,
        max_new_tokens=77,
        callback=callback,
        reference=reference,
        temperature=0.5,
    )

    assert [call["text"] for call in calls] == ["alpha beta!", "alpha", "al", "pha", "beta!", "be", "ta!"]
    assert [call["seed"] for call in calls] == [(1 << 64) - 2, (1 << 64) - 1, 0, 1, 2, 3, 4]
    assert all(call["callback"] is callback for call in calls)
    assert all(call["max_new_tokens"] == 77 for call in calls)
    assert all(call["kwargs"]["reference"] is reference for call in calls)
    assert all(call["kwargs"]["temperature"] == 0.5 for call in calls)

    assert audio["sample_rate"] == 100
    assert audio["waveform"].shape == (1, 1, 16)
    assert torch.equal(audio["waveform"][0, 0], torch.tensor([
        3, 3, 0, 0, 4, 4, 4, 0, 0, 6, 6, 0, 0, 7, 7, 7,
    ], dtype=torch.float32))
    assert stats["chunk_count"] == 4
    assert stats["discarded_attempts"] == 3
    assert stats["finished"] == [True] * 4
    assert stats["samples"] == [2, 3, 2, 3]
    assert stats["audio_seconds"] == [0.02, 0.03, 0.02, 0.03]
    assert [chunk["text"] for chunk in stats["chunks"]] == ["al", "pha", "be", "ta!"]
    assert [chunk["seed"] for chunk in stats["chunks"]] == [0, 1, 3, 4]
    assert [(chunk["sample_start"], chunk["sample_end"]) for chunk in stats["chunks"]] == [
        (0, 2), (4, 7), (9, 11), (13, 16),
    ]
    assert all(chunk["stats"]["finished"] == [True] for chunk in stats["chunks"])
    assert stats["total_seconds"] >= 0
    assert _visible("".join(chunk["text"] for chunk in stats["chunks"])) == _visible("alpha beta!")


def test_synthesize_long_propagates_cancellation_and_does_not_retry_one_character():
    callback = object()
    calls = []
    interruption = InterruptedError("cancelled")

    def cancelled(texts, *, callback, **kwargs):
        calls.append(callback)
        raise interruption

    with pytest.raises(InterruptedError, match="cancelled"):
        synthesize_long(cancelled, "hello", callback=callback)
    assert calls == [callback]

    calls.clear()

    def never_finishes(texts, **kwargs):
        calls.append(texts[0])
        return {"waveform": torch.ones(1, 1, 1), "sample_rate": 24000}, {"finished": [False]}

    with pytest.raises(RuntimeError, match="one-character"):
        synthesize_long(never_finishes, "x")
    assert calls == ["x"]


def test_synthesize_long_rejects_inconsistent_successful_audio():
    calls = 0

    def inconsistent(texts, **kwargs):
        nonlocal calls
        calls += 1
        rate = 24000 if calls == 1 else 48000
        return {"waveform": torch.ones(1, 1, 2), "sample_rate": rate}, {"finished": [True]}

    with pytest.raises(ValueError, match="different sample rates"):
        synthesize_long(inconsistent, "One. Two.", chunk_chars=5, pause_ms=0)


def test_synthesize_long_retries_only_identifiable_context_limit_errors():
    calls = []

    def context_limited(texts, **kwargs):
        text = texts[0]
        calls.append(text)
        if len(text) > 2:
            raise ValueError(
                f"Text and reference use {len(text) + 4} positions; the model context is 8."
            )
        return {"waveform": torch.ones(1, 1, len(text)), "sample_rate": 24000}, {"finished": [True]}

    audio, stats = synthesize_long(context_limited, "你好世界", pause_ms=0)

    assert calls == ["你好世界", "你好", "世界"]
    assert audio["waveform"].shape == (1, 1, 4)
    assert stats["discarded_attempts"] == 1
    assert _visible("".join(chunk["text"] for chunk in stats["chunks"])) == "你好世界"

    unrelated = ValueError("invalid prompt data")

    def invalid_prompt(texts, **kwargs):
        raise unrelated

    with pytest.raises(ValueError, match="invalid prompt data") as raised:
        synthesize_long(invalid_prompt, "hello")
    assert raised.value is unrelated


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (torch.ones(2, 1, 4), 24000, "one nonempty AUDIO sample"),
        (torch.tensor([[[float("inf")]]]), 24000, "nonfinite audio"),
        (torch.ones(1, 1, 4), 0, "invalid sample rate"),
    ],
)
def test_synthesize_long_validates_successful_audio(waveform, sample_rate, message):
    def invalid_audio(texts, **kwargs):
        return {"waveform": waveform, "sample_rate": sample_rate}, {"finished": [True]}

    with pytest.raises(ValueError, match=message):
        synthesize_long(invalid_audio, "hello")
