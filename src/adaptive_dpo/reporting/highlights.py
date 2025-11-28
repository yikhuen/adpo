from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI


@dataclass
class Highlight:
    comparison: str
    judge: str
    model_a: str
    model_b: str
    preferred_model: str
    opposing_model: str
    choice: str
    prompt: str
    response_a: str
    response_b: str
    analysis: str
    judge_delta: str
    title: str
    salient_points: Sequence[str]


def _clip(text: Optional[str], limit: int = 480) -> str:
    if not text:
        return ""
    text = text.strip()
    if limit and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _load_jsonl(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["_record_id"] = f"{path.stem}:{line_idx}"
            record["prompt"] = record.get("prompt") or ""
            record["response_a"] = record.get("response_a") or ""
            record["response_b"] = record.get("response_b") or ""
            records.append(record)
    return records


def _build_metadata_map(metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for comparison, judge_block in metrics.items():
        for judge, entry in judge_block.items():
            key = f"{judge}__{comparison}"
            meta[key] = {
                "comparison": comparison,
                "judge": judge,
                "model_a": entry.get("model_a", "model_a"),
                "model_b": entry.get("model_b", "model_b"),
            }
    return meta


class _OpenAIClient:
    _client: Optional[OpenAI] = None

    @classmethod
    def get(cls) -> OpenAI:
        if cls._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set; cannot call OpenAI.")
            cls._client = OpenAI(api_key=api_key)
        return cls._client


def _chat_completion(messages: List[Dict[str, str]], model: str, max_output_tokens: int = 800) -> str:
    client = _OpenAIClient.get()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_output_tokens,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _chunk_records(records: Sequence[Dict[str, str]], chunk_size: int = 25) -> Iterable[Sequence[Dict[str, str]]]:
    for idx in range(0, len(records), chunk_size):
        yield records[idx : idx + chunk_size]


def _analyze_chunk(
    meta: Dict[str, str],
    records: Sequence[Dict[str, str]],
    model: str,
    max_examples: int,
) -> List[Dict[str, Any]]:
    if max_examples <= 0:
        return []
    simplified = []
    for record in records:
        simplified.append(
            {
                "record_id": record["_record_id"],
                "prompt_excerpt": _clip(record.get("prompt"), 360),
                "response_a_excerpt": _clip(record.get("response_a"), 360),
                "response_b_excerpt": _clip(record.get("response_b"), 360),
                "choice": record.get("choice", ""),
            }
        )
    user_prompt = textwrap.dedent(
        f"""
        You are reviewing RLHF pairwise judgments.
        Comparison: {meta['comparison']}
        Judge: {meta['judge']}
        Model A: {meta['model_a']}
        Model B: {meta['model_b']}

        Identify up to {max_examples} records where the reasoning is especially informative for a research report.
        Return a JSON array where each entry has:
        - record_id (string, echo one of the provided record_id values)
        - title (short label)
        - judge_delta (brief explanation of why the judge preferred one response)
        - analysis (2-3 sentence rationale referencing prompt/context)
        - salient_points (array of short bullet strings highlighting notable excerpts)

        Records:
        ```json
        {json.dumps(simplified, ensure_ascii=False)}
        ```
        Ensure the result is valid JSON with double quotes.
        """
    ).strip()
    messages = [
        {
            "role": "system",
            "content": "You are an expert RLHF analyst who writes crisp research-ready rationales.",
        },
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = _chat_completion(messages, model=model)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc
    raw = raw.strip()
    if not raw:
        return []
    # Some models wrap JSON in fences; strip them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        raise ValueError(
            "Expected JSON array from OpenAI response; "
            f"received type {type(parsed).__name__}. Set --llm-model none to skip highlights."
        )
    entries: List[Dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise ValueError(
                "OpenAI response contained a non-object entry. "
                "Set --llm-model none to skip highlights."
            )
        entries.append(entry)
    return entries


def curate_highlights(
    decisions_dir: Path,
    metrics: Dict[str, Dict[str, Any]],
    *,
    model: str = "gpt-5-mini",
    max_examples: int = 4,
) -> List[Highlight]:
    if max_examples <= 0 or not decisions_dir.exists():
        return []
    metadata_map = _build_metadata_map(metrics)
    highlights: List[Highlight] = []

    decision_files = sorted(decisions_dir.glob("*.jsonl"))
    remaining = max_examples
    for decision_file in decision_files:
        meta_key = decision_file.stem
        meta = metadata_map.get(meta_key)
        if not meta or remaining <= 0:
            continue
        records = _load_jsonl(decision_file)
        if not records:
            continue
        record_map = {record["_record_id"]: record for record in records}
        selections: List[Dict[str, Any]] = []
        chunk_size = 25
        per_chunk = max(1, min(3, remaining))
        for chunk in _chunk_records(records, chunk_size=chunk_size):
            if len(selections) >= remaining:
                break
            chunk_cap = min(per_chunk, remaining - len(selections))
            raw_highlights = _analyze_chunk(meta, chunk, model=model, max_examples=chunk_cap)
            for entry in raw_highlights:
                rid = entry.get("record_id")
                if not rid or rid not in record_map:
                    continue
                if any(h.get("record_id") == rid for h in selections):
                    continue
                entry["record_id"] = rid
                selections.append(entry)
                if len(selections) >= remaining:
                    break
            if len(selections) >= remaining:
                break

        for selection in selections:
            rid = selection["record_id"]
            record = record_map[rid]
            choice = record.get("choice", "A")
            preferred_model = meta["model_a"] if choice == "A" else meta["model_b"]
            opposing_model = meta["model_b"] if choice == "A" else meta["model_a"]
            highlights.append(
                Highlight(
                    comparison=meta["comparison"],
                    judge=meta["judge"],
                    model_a=meta["model_a"],
                    model_b=meta["model_b"],
                    preferred_model=preferred_model,
                    opposing_model=opposing_model,
                    choice=choice,
                    prompt=_clip(record.get("prompt"), 900),
                    response_a=_clip(record.get("response_a"), 900),
                    response_b=_clip(record.get("response_b"), 900),
                    analysis=selection.get("analysis", selection.get("summary", "")).strip(),
                    judge_delta=selection.get("judge_delta", "").strip(),
                    title=selection.get("title", f"{meta['judge']} – {meta['comparison']}"),
                    salient_points=selection.get("salient_points") or [],
                )
            )
            remaining -= 1
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    return highlights


def highlights_to_markdown(
    highlights: Sequence[Highlight],
    *,
    heading: str = "## LLM-Curated Highlights",
) -> List[str]:
    lines = [heading, ""]
    if not highlights:
        lines.append("_No curated highlights available (LLM step skipped)._")
        lines.append("")
        return lines

    for idx, highlight in enumerate(highlights, start=1):
        lines.append(
            f"### Highlight {idx}: {highlight.title} — Judge preferred **{highlight.preferred_model}** over **{highlight.opposing_model}**"
        )
        if highlight.judge_delta:
            lines.append(f"_Judge delta_: {highlight.judge_delta}")
        if highlight.analysis:
            lines.append(highlight.analysis)
        if highlight.salient_points:
            lines.append("")
            lines.append("Key cues:")
            for point in highlight.salient_points:
                lines.append(f"- {point}")
        lines.append("")
        lines.append("**Prompt**")
        lines.append(f"> {highlight.prompt.replace(chr(10), '<br>')}")
        lines.append("")
        lines.append(f"**Response A ({highlight.model_a})**")
        lines.append(f"> {highlight.response_a.replace(chr(10), '<br>')}")
        lines.append("")
        lines.append(f"**Response B ({highlight.model_b})**")
        lines.append(f"> {highlight.response_b.replace(chr(10), '<br>')}")
        lines.append("")
    return lines


