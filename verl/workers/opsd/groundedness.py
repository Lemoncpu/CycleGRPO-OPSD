"""Conservative parsing and scoring for privileged caption groundedness checks."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence


_VALID_TYPES = {"appearance", "part", "count", "relation", "context", "other"}
_VALID_VERDICTS = {"supported", "contradicted", "unsupported", "uncertain"}
_VALID_OVERALL = {"supported", "partially_supported", "unsupported"}
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def disabled_groundedness(
    *,
    parse_failure_reason: str = "disabled",
    discarded_claim_reasons: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Return a neutral verdict, retaining parser diagnostics when available."""
    return {
        "enabled": False,
        "parse_ok": False,
        "parse_failure_reason": parse_failure_reason,
        "discarded_claim_reasons": dict(discarded_claim_reasons or {}),
        "overall": None,
        "claims": [],
        "groundedness_score": 0.0,
        "supported_count": 0,
        "unsupported_count": 0,
        "contradicted_count": 0,
        "uncertain_count": 0,
        "checked_claim_count": 0,
        "penalty": 0.0,
    }


def groundedness_penalty(verdict: Optional[Mapping[str, Any]]) -> float:
    """Return a neutral penalty unless a parsed, target-bearing verdict exists."""
    if not verdict or not verdict.get("enabled", False) or not verdict.get("parse_ok", False):
        return 0.0
    return max(0.0, float(verdict.get("penalty", 0.0)))


def _extract_json_object(text: str) -> Optional[Mapping[str, Any]]:
    cleaned = _JSON_FENCE.sub("", text or "").strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_groundedness_verdict(
    text: str,
    caption: str,
    *,
    max_claims: int,
    max_claim_chars: int = 160,
    unsupported_penalty: float = 0.25,
    contradicted_penalty: float = 0.75,
    min_checked_claims: int = 1,
) -> dict[str, Any]:
    """Parse a verifier response without turning uncertain/free-form output into loss.

    A claim is usable only when its text is a literal, non-empty substring of the
    original caption. Invalid claims are discarded rather than guessed or repaired.
    """
    if max_claims <= 0 or min_checked_claims <= 0:
        raise ValueError("max_claims and min_checked_claims must be positive.")
    if unsupported_penalty < 0 or contradicted_penalty < 0:
        raise ValueError("groundedness penalties must be non-negative.")
    payload = _extract_json_object(text)
    if payload is None:
        return disabled_groundedness(parse_failure_reason="no_json_object")
    overall = payload.get("overall")
    if overall not in _VALID_OVERALL:
        return disabled_groundedness(parse_failure_reason="invalid_overall")
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return disabled_groundedness(parse_failure_reason="claims_not_list")

    claims = []
    discarded_claim_reasons: dict[str, int] = {}

    def discard(reason: str) -> None:
        discarded_claim_reasons[reason] = discarded_claim_reasons.get(reason, 0) + 1

    for raw_claim in raw_claims[:max_claims]:
        if not isinstance(raw_claim, Mapping):
            discard("claim_not_mapping")
            continue
        claim_text = raw_claim.get("text")
        claim_type = raw_claim.get("type")
        verdict = raw_claim.get("verdict")
        if not isinstance(claim_text, str) or not claim_text.strip():
            discard("claim_missing_or_empty_text")
            continue
        if len(claim_text) > max_claim_chars:
            discard("claim_text_too_long")
            continue
        if claim_text not in caption:
            discard("claim_not_literal_substring")
            continue
        if claim_type not in _VALID_TYPES:
            discard("claim_invalid_type")
            continue
        if verdict not in _VALID_VERDICTS:
            discard("claim_invalid_verdict")
            continue
        claims.append({"text": claim_text, "type": claim_type, "verdict": verdict})

    if not claims:
        return disabled_groundedness(
            parse_failure_reason="no_valid_claims",
            discarded_claim_reasons=discarded_claim_reasons,
        )
    checked = sum(claim["verdict"] in {"supported", "unsupported", "contradicted"} for claim in claims)
    if checked < min_checked_claims:
        return disabled_groundedness(
            parse_failure_reason="insufficient_checked_claims",
            discarded_claim_reasons=discarded_claim_reasons,
        )
    counts = {verdict: sum(claim["verdict"] == verdict for claim in claims) for verdict in _VALID_VERDICTS}
    unsupported_rate = counts["unsupported"] / checked
    contradicted_rate = counts["contradicted"] / checked
    penalty = unsupported_penalty * unsupported_rate + contradicted_penalty * contradicted_rate
    score = max(0.0, 1.0 - unsupported_rate - contradicted_rate)
    return {
        "enabled": True,
        "parse_ok": True,
        "parse_failure_reason": None,
        "discarded_claim_reasons": discarded_claim_reasons,
        "overall": overall,
        "claims": claims,
        "groundedness_score": score,
        "supported_count": counts["supported"],
        "unsupported_count": counts["unsupported"],
        "contradicted_count": counts["contradicted"],
        "uncertain_count": counts["uncertain"],
        "checked_claim_count": checked,
        "penalty": penalty,
    }


def groundedness_token_mask(
    response_ids: Sequence[int],
    tokenizer: Any,
    claims: Sequence[Mapping[str, Any]],
) -> list[bool]:
    """Mark response token spans for unsupported/contradicted literal claims.

    Exact token-subsequence matching intentionally fails closed: if the tokenizer
    cannot align a literal claim, no token-level auxiliary weight is applied.
    """
    token_ids = [int(token_id) for token_id in response_ids]
    mask = [False] * len(token_ids)
    for claim in claims:
        if claim.get("verdict") not in {"unsupported", "contradicted"}:
            continue
        claim_ids = tokenizer.encode(str(claim.get("text", "")), add_special_tokens=False)
        if not claim_ids or len(claim_ids) > len(token_ids):
            continue
        for start in range(len(token_ids) - len(claim_ids) + 1):
            if token_ids[start : start + len(claim_ids)] == list(claim_ids):
                for index in range(start, start + len(claim_ids)):
                    mask[index] = True
                break
    return mask
