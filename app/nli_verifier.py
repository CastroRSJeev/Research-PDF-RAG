from __future__ import annotations
from functools import lru_cache
from typing import List

import config


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.CITATION_MODEL, max_length=512)


def _label_order(model) -> List[str]:
    cfg = getattr(getattr(model, "model", None), "config", None)
    if cfg and getattr(cfg, "id2label", None):
        return [cfg.id2label[i].title() for i in sorted(cfg.id2label)]
    return ["Contradiction", "Entailment", "Neutral"]


def entailment_score(premise: str, hypothesis: str) -> float:
    """Return the Entailment probability (0-1). Returns 0.0 if either input is empty."""
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    model = _get_model()
    scores = model.predict([(premise, hypothesis)], apply_softmax=True)
    score_list = scores[0].tolist() if hasattr(scores[0], "tolist") else list(scores[0])
    labels = _label_order(model)
    idx = next((i for i, l in enumerate(labels) if l.lower() == "entailment"), None)
    return float(score_list[idx]) if idx is not None else 0.0


def nli_label(premise: str, claim: str) -> str:
    """Return the top NLI label for (premise, claim)."""
    if not premise.strip() or not claim.strip():
        return "Neutral"
    model = _get_model()
    scores = model.predict([(premise, claim)], apply_softmax=True)
    score_list = scores[0].tolist() if hasattr(scores[0], "tolist") else list(scores[0])
    labels = _label_order(model)
    best = max(range(len(score_list)), key=score_list.__getitem__)
    return labels[best]
