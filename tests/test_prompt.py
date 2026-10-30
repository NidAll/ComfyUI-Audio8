import pytest

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")

from audio8.prompt import build_prompt, load_tokenizer


def _tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    pieces = [
        "[UNK]",
        "<|im_start|>system",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|voice|>",
        "<|speaker:0|>",
        "convert",
        "the",
        "provided",
        "text",
        "to",
        "speech",
        "reference",
        "following:",
        "Text:",
        "Speech:",
        "words",
        "bonjour",
        "hello",
        "world",
        "Arabic",
        "مرحبا",
        "你好",
    ]
    tokenizer = Tokenizer(models.WordLevel({piece: index for index, piece in enumerate(pieces)}, unk_token="[UNK]"))
    # Split the speaker marker as its own token while keeping the chat markers
    # with their adjacent role text, as the Audio8 prompt expects.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(r"<\|speaker:\d+\|>"), behavior="isolated"),
         pre_tokenizers.WhitespaceSplit()]
    )
    return tokenizer


def _encode(tokenizer, parts):
    return [token for part in parts for token in tokenizer.encode(part, add_special_tokens=False).ids]


def test_load_tokenizer_uses_tokenizers_json_and_reference_frames_align():
    tokenizer = _tokenizer()
    tokenizer = load_tokenizer(tokenizer.to_str().encode("utf-8"))
    config = {"num_codebooks": 3, "semantic_begin_id": 1000, "pad_token_id": 0}
    references = torch.tensor([[1, 2], [11, 12], [21, 22]])
    texts = ["bonjour", "hello world"]

    prompt, mask = build_prompt(tokenizer, config, texts, references, "reference words")

    prefix = _encode(tokenizer, [
        "<|im_start|>system\n",
        "convert the provided text to speech reference to the following:\n\nText:\n",
        "<|speaker:0|>reference words",
        "\n\nSpeech:\n",
    ])
    assert tokenizer.encode("<|speaker:0|>", add_special_tokens=False).ids[0] in prefix
    assert prompt.shape == (2, 4, mask.shape[1])
    assert mask.dtype == torch.long

    for index, text in enumerate(texts):
        suffix = _encode(tokenizer, [
            "<|im_end|>\n", "<|im_start|>user\n", text,
            "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>",
        ])
        row_length = len(prefix) + references.shape[1] + len(suffix)
        left = prompt.shape[-1] - row_length
        assert torch.equal(mask[index, :left], torch.zeros(left, dtype=torch.long))
        assert torch.equal(mask[index, left:], torch.ones(row_length, dtype=torch.long))
        assert torch.equal(prompt[index, 0, left:left + len(prefix)], torch.tensor(prefix))
        frame_start = left + len(prefix)
        assert torch.equal(prompt[index, 0, frame_start:frame_start + 2], references[0] + 1000)
        assert torch.equal(prompt[index, 1:, frame_start:frame_start + 2], references)
        assert torch.equal(prompt[index, 0, -len(suffix):], torch.tensor(suffix))
        if left:
            assert torch.equal(prompt[index, 0, :left], torch.zeros(left, dtype=torch.long))
            assert torch.equal(prompt[index, 1:, :left], torch.zeros(3, left, dtype=torch.long))


def test_non_reference_prompt_left_pads_and_uses_semantic_pad_id():
    tokenizer = _tokenizer()
    config = {"num_codebooks": 2, "semantic_begin_id": 1000, "pad_token_id": 9}

    prompt, mask = build_prompt(tokenizer, config, ["bonjour", "hello world"])

    assert prompt.shape == (2, 3, mask.shape[1])
    assert mask[0, 0].item() == 0
    assert mask[1, 0].item() == 1
    left = int((mask[0] == 0).sum())
    assert torch.all(prompt[0, 0, :left] == config["pad_token_id"])
    assert torch.all(prompt[0, 1:, :left] == 0)
    assert torch.all(prompt[:, 1:] == 0)


@pytest.mark.parametrize("texts", [[], [""], [" \n  "]])
def test_empty_synthesis_text_is_rejected(texts):
    with pytest.raises(ValueError, match="nonempty text"):
        build_prompt(_tokenizer(), {"num_codebooks": 2, "semantic_begin_id": 1000,
                                    "pad_token_id": 0}, texts)


def test_reference_requires_transcript_and_codebook_major_codes():
    config = {"num_codebooks": 2, "semantic_begin_id": 1000, "pad_token_id": 0}
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="exact spoken reference transcript"):
        build_prompt(tokenizer, config, ["hello"], torch.ones(2, 1), "  ")
    with pytest.raises(ValueError, match=r"\[num_codebooks, frames\]"):
        build_prompt(tokenizer, config, ["hello"], torch.ones(1, 2), "reference words")
