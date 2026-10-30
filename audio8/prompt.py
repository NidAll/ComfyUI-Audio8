"""Upstream prompt format with a tokenizer loaded directly from embedded JSON."""

import re

import torch
from tokenizers import Tokenizer


LANGUAGES = ['Auto (text determines language)', 'Arabic', 'Cantonese', 'Chinese', 'Dutch',
             'English', 'Finnish', 'French', 'German', 'Hindi', 'Indonesian', 'Italian',
             'Japanese', 'Korean', 'Malay', 'Norwegian', 'Polish', 'Portuguese', 'Russian',
             'Spanish', 'Swedish', 'Thai', 'Turkish', 'Vietnamese']


def clean(text):
    return ' '.join(str(text).strip().split())


def build_prompt(tokenizer, config, texts, reference_codes=None, reference_text=''):
    if not texts or any(not clean(text) for text in texts):
        raise ValueError('Enter nonempty text to synthesize.')
    if reference_codes is not None and not clean(reference_text):
        raise ValueError('Voice cloning requires the exact spoken reference transcript.')
    if reference_codes is not None and (reference_codes.ndim != 2 or reference_codes.shape[0] != config['num_codebooks']):
        raise ValueError('Reference codes must have shape [num_codebooks, frames].')

    def encode_parts(parts):
        return [token for part in parts for token in tokenizer.encode(part, add_special_tokens=False).ids]

    rows = []
    for text in texts:
        target = clean(text)
        if reference_codes is None:
            prefix = encode_parts(['<|im_start|>system\n', 'convert the provided text to speech',
                                   '<|im_end|>\n', '<|im_start|>user\n', target,
                                   '<|im_end|>\n', '<|im_start|>assistant\n<|voice|>'])
            suffix = []
            frames = 0
        else:
            reference = clean(reference_text)
            if not re.search(r'<\|speaker:\d+\|>', reference):
                reference = '<|speaker:0|>' + reference
            prefix = encode_parts(['<|im_start|>system\n',
                                   'convert the provided text to speech reference to the following:\n\nText:\n',
                                   reference, '\n\nSpeech:\n'])
            suffix = encode_parts(['<|im_end|>\n', '<|im_start|>user\n', target,
                                   '<|im_end|>\n', '<|im_start|>assistant\n<|voice|>'])
            frames = reference_codes.shape[1]
        row = torch.zeros(config['num_codebooks'] + 1, len(prefix) + frames + len(suffix), dtype=torch.long)
        row[0, :len(prefix)] = torch.tensor(prefix)
        if frames:
            row[0, len(prefix):len(prefix) + frames] = reference_codes[0] + config['semantic_begin_id']
            row[1:, len(prefix):len(prefix) + frames] = reference_codes
        if suffix:
            row[0, -len(suffix):] = torch.tensor(suffix)
        rows.append(row)
    width = max(row.shape[-1] for row in rows)
    prompt = torch.zeros(len(rows), config['num_codebooks'] + 1, width, dtype=torch.long)
    prompt[:, 0] = config['pad_token_id']
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        prompt[index, :, width - row.shape[-1]:] = row
        mask[index, width - row.shape[-1]:] = 1
    return prompt, mask


def load_tokenizer(data):
    return Tokenizer.from_str(data.decode('utf-8'))
