"""Token-probability local uncertainty for PropUQ-MAS.

Both variants use length-normalized MSP:

    u = 1 - exp(mean log p(token)) = 1 - geometric-mean token probability
      = 1 - 1 / perplexity

This is NOT classic sequence-level MSP (1 - prod p(token)), which saturates
at ~1.0 for outputs with thousands of reasoning tokens.

    NormMSP         every generated token (thinking + answer)
    NormMSP-Answer  only tokens after the last </think>

In both variants the tokens of the uncertainty metadata the agent is asked to
append as XML tags (<alpha .../>, <local_uncertainty .../>) are
excluded, so eliciting adoption scores does not change the local estimate.

If the model opened <think> but never closed it (generation truncated), there
is no answer and NormMSP-Answer is set to 1.0 (maximal uncertainty).
If the output contains no think tags at all (e.g. Gemma, or thinking
disabled), the whole output is treated as the answer.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

NORM_MSP = "NormMSP"
NORM_MSP_ANSWER = "NormMSP-Answer"
TOKEN_PROB_UQ_METHODS = (NORM_MSP, NORM_MSP_ANSWER)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_METADATA_PATTERNS = (
    re.compile(r"<local_uncertainty\b[^>]*>.*?</local_uncertainty>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<(?:local_uncertainty|alpha|message_adoption)\b[^>]*/?>", re.IGNORECASE),
    # Plain-text forms such as "alpha: 0.7" are deliberately NOT excluded:
    # that pattern also matches real answer content (e.g. `alpha = 0.5` in code).
)


def _metadata_spans(text: str) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    for pattern in _METADATA_PATTERNS:
        spans.extend((m.start(), m.end()) for m in pattern.finditer(text))
    return spans


def _norm_msp(logprobs: Sequence[float]) -> Optional[float]:
    if not logprobs:
        return None
    mean_logprob = sum(logprobs) / len(logprobs)
    return max(0.0, min(1.0, 1.0 - math.exp(mean_logprob)))


def token_prob_uncertainties(
    token_ids: Sequence[int],
    token_logprobs: Sequence[Optional[float]],
    tokenizer,
) -> Dict[str, object]:
    """Compute NormMSP and NormMSP-Answer from per-token log-probabilities.

    Token pieces are decoded one by one and concatenated, and all character
    offsets (think boundary, metadata tags) are taken from that concatenated
    string, so token positions and text positions always agree.
    """
    token_ids = list(token_ids or [])
    token_logprobs = list(token_logprobs or [])
    n = min(len(token_ids), len(token_logprobs))
    token_ids, token_logprobs = token_ids[:n], token_logprobs[:n]

    empty = {
        NORM_MSP: None,
        NORM_MSP_ANSWER: None,
        "token_count_all": 0,
        "token_count_answer": 0,
        "answer_truncated": False,
        "has_think_block": False,
    }
    if n == 0:
        return empty

    pieces = tokenizer.batch_decode(
        [[int(tid)] for tid in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    starts: List[int] = []
    pos = 0
    for piece in pieces:
        starts.append(pos)
        pos += len(piece)
    ends = [s + len(p) for s, p in zip(starts, pieces)]
    text = "".join(pieces)

    spans = _metadata_spans(text)

    def is_metadata(i: int) -> bool:
        return any(starts[i] < span_end and ends[i] > span_start for span_start, span_end in spans)

    close_idx = text.rfind(THINK_CLOSE)
    has_open = THINK_OPEN in text
    has_think_block = close_idx >= 0 or has_open
    if close_idx >= 0:
        answer_start = close_idx + len(THINK_CLOSE)
        answer_truncated = False
    elif has_open:
        answer_start = len(text)  # thinking never finished: no answer tokens
        answer_truncated = True
    else:
        answer_start = 0
        answer_truncated = False

    all_lp: List[float] = []
    answer_lp: List[float] = []
    for i, lp in enumerate(token_logprobs):
        if lp is None or is_metadata(i):
            continue
        all_lp.append(float(lp))
        if starts[i] >= answer_start:
            answer_lp.append(float(lp))

    answer_u = _norm_msp(answer_lp)
    if answer_u is None and answer_truncated:
        answer_u = 1.0

    return {
        NORM_MSP: _norm_msp(all_lp),
        NORM_MSP_ANSWER: answer_u,
        "token_count_all": len(all_lp),
        "token_count_answer": len(answer_lp),
        "answer_truncated": answer_truncated,
        "has_think_block": has_think_block,
    }
