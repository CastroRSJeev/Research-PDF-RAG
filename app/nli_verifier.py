from __future__ import annotations

import config

_SYSTEM = (
    "You are an NLI classifier. Given a premise and a hypothesis, "
    "respond with exactly one word: Entailment, Contradiction, or Neutral."
)


def _call_nli(premise: str, hypothesis: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=config.HF_BASE_URL, api_key=config.HF_TOKEN)
    try:
        result = client.chat.completions.create(
            model=config.HF_FIRST_MODEL,
            max_tokens=5,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Premise: {premise}\nHypothesis: {hypothesis}"},
            ],
        ).choices[0].message.content or ""
        result = result.strip().split()[0].capitalize()
        if result in ("Entailment", "Contradiction", "Neutral"):
            return result
    except Exception:
        pass
    return "Neutral"


def entailment_score(premise: str, hypothesis: str) -> float:
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    return 1.0 if _call_nli(premise, hypothesis) == "Entailment" else 0.0


def nli_label(premise: str, claim: str) -> str:
    if not premise.strip() or not claim.strip():
        return "Neutral"
    return _call_nli(premise, claim)
