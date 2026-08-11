#!/usr/bin/env python3
"""Generate validated text-only multiple-choice QA from DAM caption manifests."""

import argparse
import concurrent.futures
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


GENERATION_PROMPT = """You create rigorous text-only multiple-choice questions for
evaluating a generated region description. The reference caption is the only evidence.

Return exactly one JSON object with this schema:
{
  "class_name": "short common-noun phrase",
  "questions": [
    {
      "type": "positive",
      "question": "question answerable only from the caption",
      "choices": [["correct choice", 1], ["distractor", 0], ["distractor", 0], ["distractor", 0]]
    },
    {
      "type": "positive",
      "question": "another question answerable only from the caption",
      "choices": [["correct choice", 1], ["distractor", 0], ["distractor", 0], ["distractor", 0]]
    }
  ]
}

Rules:
- Produce exactly two positive questions. They must test distinct, explicitly stated facts.
- Each positive question has exactly four concise, unique choices, exactly one score 1 and three score 0.
- Prefer object category, visible attributes, material, part, count, action, or a relation explicitly in the caption.
- Do not infer facts from world knowledge, the image filename, or what is merely plausible.
- You may add at most one optional negative question after the two positives. It must have type "negative", two choices [["Yes", -1], ["No", 0]], and ask whether the description states a specifically false property. It must not claim that an unmentioned fact is visually absent.
- Do not mention "reference caption", "option", "image", scoring, or these instructions in a question.
- Return JSON only. No markdown and no explanation.

Reference caption:
{caption}
"""


VALIDATION_PROMPT = """You validate text-only multiple-choice QA against a reference
caption. The caption is the only evidence. Return exactly this JSON object:
{"accepted": true_or_false, "reasons": ["short reason"]}

Accept only when all of the following are true:
1. Every positive question is uniquely answerable from an explicit caption fact.
2. No question adds an unstated visual fact or relies on common-sense inference.
3. Positive questions have four distinct choices with one score 1 and three score 0.
4. A negative question, if present, asks whether the description states a false property;
   its choices must be Yes=-1 and No=0.
5. The two positive questions test different facts and are not materially ambiguous.

Reference caption:
{caption}

Candidate QA:
{candidate_json}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        required=True,
        help="DAM caption JSONL manifest; pass once for each source.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Validated QA JSONL output.")
    parser.add_argument("--rejected-output", type=Path, help="Optional rejected-record JSONL output.")
    parser.add_argument("--dlc-qa-output", type=Path, help="Optional DLC-compatible QA JSON object.")
    parser.add_argument(
        "--class-names-output", type=Path, help="Optional DLC-compatible class-name JSON object."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--api-key", default="sk-abc123")
    parser.add_argument("--model", default="llama3.1-8b")
    parser.add_argument("--validator-model", help="Defaults to --model.")
    parser.add_argument("--max-samples", type=int, help="Deterministic random pilot/full-set size.")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--resume", action="store_true", help="Append missing IDs to an existing JSONL output.")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing companion JSON outputs.")
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object, tolerating model markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("response does not contain a JSON object")


def normalize_choices(value: Any, question_type: str) -> list[list[Any]]:
    if not isinstance(value, list):
        raise ValueError("choices must be a list")
    choices: list[list[Any]] = []
    for choice in value:
        if not isinstance(choice, (list, tuple)) or len(choice) != 2:
            raise ValueError("each choice must be [text, score]")
        text, score = choice
        if not isinstance(text, str) or not text.strip():
            raise ValueError("choice text must be non-empty")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("choice score must be numeric")
        normalized_score = int(score) if float(score).is_integer() else float(score)
        choices.append([text.strip(), normalized_score])

    labels = [choice[0].casefold() for choice in choices]
    if len(set(labels)) != len(labels):
        raise ValueError("choice text must be unique")
    if question_type == "positive":
        if len(choices) != 4 or [choice[1] for choice in choices].count(1) != 1:
            raise ValueError("positive question needs four choices and exactly one score 1")
        if any(choice[1] not in (0, 1) for choice in choices):
            raise ValueError("positive choice scores must be 0 or 1")
    elif question_type == "negative":
        if len(choices) != 2:
            raise ValueError("negative question needs Yes/No choices")
        normalized = {text.casefold(): score for text, score in choices}
        if normalized != {"yes": -1, "no": 0}:
            raise ValueError("negative choices must be Yes=-1 and No=0")
    else:
        raise ValueError("question type must be positive or negative")
    return choices


def normalize_generated_payload(payload: dict[str, Any]) -> dict[str, Any]:
    class_name = payload.get("class_name")
    questions = payload.get("questions")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be non-empty")
    if len(class_name.strip().split()) > 6:
        raise ValueError("class_name must be a short noun phrase")
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")

    normalized_questions = []
    positive_count = 0
    negative_count = 0
    question_texts = set()
    for raw_question in questions:
        if not isinstance(raw_question, dict):
            raise ValueError("question must be an object")
        question_type = raw_question.get("type")
        question = raw_question.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question text must be non-empty")
        key = question.strip().casefold()
        if key in question_texts:
            raise ValueError("question text must be unique")
        question_texts.add(key)
        if question_type == "positive":
            positive_count += 1
        elif question_type == "negative":
            negative_count += 1
        choices = normalize_choices(raw_question.get("choices"), question_type)
        normalized_questions.append(
            {"type": question_type, "question": question.strip(), "choices": choices}
        )

    if positive_count != 2 or negative_count > 1 or len(normalized_questions) != positive_count + negative_count:
        raise ValueError("QA needs exactly two positives and at most one negative")
    return {"class_name": class_name.strip(), "questions": normalized_questions}


def validate_caption_qa_questions(questions: Any) -> None:
    """Validate a persisted sidecar without requiring its optional class name."""
    normalize_generated_payload({"class_name": "placeholder", "questions": questions})


def request_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("completion content is not text")
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"completion failed after {retries} attempts: {last_error}")


def load_manifest_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                dam_source_id = row.get("dam_source_id")
                caption = row.get("caption")
                if not isinstance(dam_source_id, str) or not dam_source_id:
                    raise ValueError(f"{path}:{line_number} is missing dam_source_id")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"{path}:{line_number} is missing caption")
                if dam_source_id in seen:
                    raise ValueError(f"duplicate dam_source_id {dam_source_id!r}")
                seen.add(dam_source_id)
                rows.append(row)
    return rows


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            dam_source_id = row.get("dam_source_id")
            if not isinstance(dam_source_id, str):
                raise ValueError(f"{path}:{line_number} is missing dam_source_id")
            if dam_source_id in ids:
                raise ValueError(f"{path} has duplicate dam_source_id {dam_source_id!r}")
            ids.add(dam_source_id)
    return ids


def validate_with_llm(
    candidate: dict[str, Any], row: dict[str, Any], args: argparse.Namespace
) -> tuple[bool, list[str]]:
    prompt = VALIDATION_PROMPT.format(
        caption=row["caption"].strip(), candidate_json=json.dumps(candidate, ensure_ascii=False)
    )
    response = request_completion(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.validator_model or args.model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=250,
        timeout=args.timeout,
        retries=args.request_retries,
    )
    payload = extract_json_object(response)
    accepted = payload.get("accepted") is True
    reasons = payload.get("reasons")
    if not isinstance(reasons, list):
        reasons = ["validator did not return reasons"]
    return accepted, [str(reason) for reason in reasons]


def generate_one(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    errors = []
    for attempt in range(1, args.generation_attempts + 1):
        try:
            response = request_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=GENERATION_PROMPT.format(caption=row["caption"].strip()),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.request_retries,
            )
            candidate = normalize_generated_payload(extract_json_object(response))
            accepted, reasons = validate_with_llm(candidate, row, args)
            if not accepted:
                errors.append(f"attempt {attempt}: validator rejected: {'; '.join(reasons)}")
                continue
            result = {
                "dam_source_id": row["dam_source_id"],
                "source": row.get("source"),
                "dam_ann_id": row.get("dam_ann_id"),
                "dam_image": row.get("dam_image"),
                "image_path": row.get("image_path"),
                "caption": row["caption"].strip(),
                **candidate,
                "generation": {
                    "model": args.model,
                    "validator_model": args.validator_model or args.model,
                    "attempt": attempt,
                },
            }
            return result, None
        except Exception as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
    return None, {"dam_source_id": row["dam_source_id"], "caption": row["caption"], "errors": errors}


def validate_output_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_concurrency <= 0 or args.generation_attempts <= 0 or args.request_retries <= 0:
        raise ValueError("concurrency and attempt counts must be positive")
    if bool(args.dlc_qa_output) != bool(args.class_names_output):
        raise ValueError("--dlc-qa-output and --class-names-output must be supplied together")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to append missing rows: {args.output}")
    for path in (args.dlc_qa_output, args.class_names_output):
        if path and path.exists() and not (args.resume or args.overwrite):
            raise FileExistsError(f"companion output exists; pass --overwrite: {path}")


def write_companion_jsons(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.dlc_qa_output:
        return
    qa = {record["dam_source_id"]: record["questions"] for record in records}
    class_names = {record["dam_source_id"]: record["class_name"] for record in records}
    args.dlc_qa_output.parent.mkdir(parents=True, exist_ok=True)
    args.class_names_output.parent.mkdir(parents=True, exist_ok=True)
    args.dlc_qa_output.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    args.class_names_output.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_output_args(args)
    rows = load_manifest_rows(args.input_manifest)
    random.Random(args.seed).shuffle(rows)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    row_ids = {row["dam_source_id"] for row in rows}
    existing_ids = load_existing_ids(args.output) if args.resume else set()
    unexpected_ids = existing_ids - row_ids
    if unexpected_ids:
        raise ValueError(
            "existing output contains IDs outside this selected input set; use a separate output path "
            f"(first unexpected ID: {next(iter(unexpected_ids))!r})"
        )
    pending = [row for row in rows if row["dam_source_id"] not in existing_ids]
    rejected_path = args.rejected_output or args.output.with_suffix(".rejected.jsonl")
    if rejected_path.exists() and not args.resume:
        raise FileExistsError(f"rejected output exists; pass --resume: {rejected_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = len(rows) - len(pending)
    rejected_count = 0
    with args.output.open("a", encoding="utf-8") as accepted_file, rejected_path.open(
        "a", encoding="utf-8"
    ) as rejected_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = {executor.submit(generate_one, row, args): row for row in pending}
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result, rejected = future.result()
                if result is not None:
                    accepted_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    accepted_file.flush()
                    accepted_count += 1
                if rejected is not None:
                    rejected_file.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                    rejected_file.flush()
                    rejected_count += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"processed={completed}/{len(pending)} accepted={accepted_count} rejected={rejected_count}",
                        flush=True,
                    )

    records = []
    with args.output.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    write_companion_jsons(records, args)
    summary = {
        "input_manifests": [str(path.resolve()) for path in args.input_manifest],
        "output": str(args.output.resolve()),
        "rejected_output": str(rejected_path.resolve()),
        "requested": len(rows),
        "already_present": len(rows) - len(pending),
        "accepted": accepted_count,
        "rejected_this_run": rejected_count,
        "seed": args.seed,
        "model": args.model,
        "validator_model": args.validator_model or args.model,
        "max_concurrency": args.max_concurrency,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if accepted_count != len(rows):
        raise RuntimeError(
            f"Only generated {accepted_count}/{len(rows)} QA rows. Inspect {rejected_path} and resume."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
