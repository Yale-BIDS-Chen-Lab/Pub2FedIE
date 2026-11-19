"""Strict extraction targets, prompts, parsers and multiset scoring."""
import json
from collections import Counter


def exact_occurrence_starts(text: str, mention: str) -> list[int]:
    """Return all exact, potentially overlapping mention starts."""

    if not mention:
        return []
    starts = []
    start_at = 0
    while True:
        start = text.find(mention, start_at)
        if start < 0:
            return starts
        starts.append(start)
        start_at = start + 1


def occurrence_number(text: str, mention: str, start: int) -> int:
    for number, position in enumerate(exact_occurrence_starts(text, mention), 1):
        if position == start:
            return number
    raise RuntimeError(f"Invalid span at start {start} (length {len(mention)}): no exact match")


def example_counts(
    example: dict, task_family: str, parsed: list[tuple]
) -> dict[str, int]:
    """Count exact NER spans or RE pairs, preserving duplicates."""
    if task_family == "NER":
        gold = Counter(
            (int(item["start"]), int(item["end"]), item["label"])
            for item in example["gold"]
        )
        predicted: Counter[tuple[int, int, str]] = Counter()
        unresolved = 0
        for mention, label, occurrence in parsed:
            starts = exact_occurrence_starts(example["text"], mention)
            if occurrence <= len(starts):
                start = starts[occurrence - 1]
                predicted[(start, start + len(mention), label)] += 1
            else:
                unresolved += 1
        true_positive = sum((gold & predicted).values())
        return {
            "tp": true_positive,
            "fp": sum(predicted.values()) - true_positive + unresolved,
            "fn": sum(gold.values()) - true_positive,
        }
    if task_family == "RE":
        gold = Counter((item["text"], item["label"]) for item in example["gold"])
        predicted = Counter(parsed)
        true_positive = sum((gold & predicted).values())
        return {
            "tp": true_positive,
            "fp": sum(predicted.values()) - true_positive,
            "fn": sum(gold.values()) - true_positive,
        }
    raise ValueError(task_family)


def canonical_output(task_family: str, example: dict) -> str:
    """Top-level array for NER; source-conditioned target/label pairs for RE."""

    if task_family == "NER":
        payload = [
            {
                "text": item["text"],
                "type": item["label"],
                "occurrence": occurrence_number(
                    example["text"], item["text"], int(item["start"])
                ),
            }
            for item in sorted(
                example["gold"],
                key=lambda item: (item["start"], item["end"], item["label"]),
            )
        ]
    elif task_family == "RE":
        payload = {"relations": [
            {"target": item["text"], "relation": item["label"]}
            for item in sorted(example["gold"], key=lambda item: (
                item.get("start", -1), item.get("end", -1), item["text"], item["label"]
            ))
        ]}
    else:
        raise ValueError(task_family)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def prompt_for_example(
    task_family: str, schema: list[str], example: dict
) -> str:
    """Format NER prompts or public RE prompts with a supplied or implicit source."""
    if task_family not in ("NER", "RE"):
        raise ValueError(task_family)
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    if task_family == "NER":
        return (
            "Extract every explicitly stated entity whose type belongs to the provided schema. "
            "Copy each span exactly from the text. To distinguish repeated identical surface "
            "strings, occurrence is the one-based index of that exact, case-sensitive string "
            "when scanning Text from left to right. Return only a valid top-level JSON array "
            'with exact item schema [{"text":"exact span","type":"TYPE","occurrence":1}]. '
            "Return [] when none apply. Do not wrap the array in an object or Markdown.\n"
            f"Entity schema: {schema_text}\nText: {example['text']}"
        )

    source = example["source"]
    if source is None:
        source_text = "implicit sentence-level clinical event"
    else:
        source_text = json.dumps(
            {"text": source["text"], "type": source["label"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    target_instruction = (
        "Copy target spans exactly from the text."
        if example.get("target_must_be_span", True)
        else "Return the official result-value and test-label pairs; a value may be implicit in sentence context."
    )
    return (
        "Given the source and text, extract every related target and relation label "
        f"from the provided schema. {target_instruction} Return only "
        "valid JSON with exact schema {\"relations\":[{\"target\":\"exact span\","
        "\"relation\":\"TYPE\"}]}. Return {\"relations\":[]} when no relation applies.\n"
        f"Relation schema: {schema_text}\nSource: {source_text}\nText: {example['text']}"
    )


def parse_prediction(
    raw: str, task_family: str, schema: set[str]
) -> tuple[list[tuple], bool, str | None]:
    if task_family not in ("NER", "RE"):
        raise ValueError(task_family)
    try:
        values = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        return [], False, f"json_decode_error:{error.msg}"
    ner = task_family == "NER"
    if ner:
        if not isinstance(values, list):
            return [], False, "schema_error:expected_top_level_entity_array"
        text_key, label_key = "text", "type"
        expected = {text_key, label_key, "occurrence"}
    else:
        if not isinstance(values, dict) or set(values) != {"relations"}:
            return [], False, "schema_error:expected_exact_relations_object"
        values = values["relations"]
        if not isinstance(values, list):
            return [], False, "schema_error:relations_must_be_list"
        text_key, label_key = "target", "relation"
        expected = {text_key, label_key}
    parsed = []
    for value in values:
        if not isinstance(value, dict) or set(value) != expected:
            return [], False, "schema_error:wrong_item_keys"
        if ner and (type(value["occurrence"]) is not int or value["occurrence"] < 1):
            return [], False, "schema_error:invalid_occurrence_index"
        mention, label = value[text_key], value[label_key]
        if not isinstance(mention, str) or not mention or not isinstance(label, str):
            return [], False, "schema_error:non_string_or_empty_value"
        if label not in schema:
            return [], False, "schema_error:label_outside_schema"
        parsed.append((mention, label, value["occurrence"]) if ner else (mention, label))
    return parsed, True, None


def conditional_re_prompt(task_family, schema, example):
    if task_family != "RE":
        raise ValueError("Conditional RE instruction is RE-only")
    source = example["source"]
    identity = dict(text=source["text"], type=source["label"], start=source["start"],
        end=source["end"], occurrence=source["occurrence"])
    return ("Given the marked primary source and text, extract every related target and relation label "
        "from the provided schema. Source start/end are zero-based character offsets in Text; "
        "end is exclusive. Source occurrence is the one-based exact-text occurrence in Text. "
        "These identify the supplied source when identical mentions repeat. Copy target spans exactly. "
        'Return only valid JSON with exact schema {"relations":[{"target":"exact span","relation":"TYPE"}]}. '
        'Return {"relations":[]} when none apply.\n'
        f"Relation schema: {json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Source: {json.dumps(identity, ensure_ascii=False, separators=(',', ':'))}\nText: {example['text']}")
