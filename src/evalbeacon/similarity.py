"""Text similarity. Default is a deterministic bag-of-words cosine; pass embed_fn for real embeddings."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]


def _bow(t: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", t.lower()))


def _cos(a, b) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity(a: str, b: str, embed_fn: EmbedFn | None = None) -> float:
    """Cosine similarity in [0, 1]. Lexical unless embed_fn is given (then vectors come from embed_fn)."""
    if embed_fn is None:
        return round(_cos(_bow(a), _bow(b)), 4)
    va, vb = embed_fn([a, b])
    return round(_cos(dict(enumerate(va)), dict(enumerate(vb))), 4)
