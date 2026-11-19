"""Parse clinical annotations from BIO tags (NER) and input/output markup (RE).

RE scoring counts exact (target text, relation) pairs, including repeated mentions.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, Sequence

from .scoring import occurrence_number

SOURCE_LABELS=frozenset(("problem","test","drug","treatment"))
RELATION_LABELS=frozenset(("bodyloc","condition","course","dosage","duration","form",
    "frequency","labvalue","negation","reference_range","route","severity","strength",
    "subject","temporal","uncertain"))
MARKER=re.compile(r'<span\s+class=[\"\']([^\"\']+)[\"\']\s*>|</span\s*>')


def parse_annotation_markup(value,allowed_labels):
    """Parse annotation spans and keep literal angle brackets in the text.

    Decode HTML references within each segment so offsets match the decoded text.
    """
    if not isinstance(value,str):raise ValueError("Markup must be a string")
    parts=[];spans=[];active=None;cursor=0;last=0
    for marker in MARKER.finditer(value):
        piece=html.unescape(value[last:marker.start()]);parts.append(piece);cursor+=len(piece)
        label=marker.group(1)
        if label is not None:
            if label not in allowed_labels:raise ValueError("Annotation label outside the frozen schema")
            if active is not None:raise ValueError("Nested annotation spans")
            active=(cursor,label)
        else:
            if active is None:raise ValueError("Unmatched annotation closing tag")
            start,label=active
            if cursor<=start:raise ValueError("Empty annotation span")
            spans.append(dict(start=start,end=cursor,label=label));active=None
        last=marker.end()
    parts.append(html.unescape(value[last:]))
    if active is not None:raise ValueError("Unclosed annotation span")
    text="".join(parts)
    if re.search(r"</?span\b",MARKER.sub("",value)):raise ValueError("Unrecognized annotation tag syntax")
    for span in spans:
        span["text"]=text[span["start"]:span["end"]]
        if not span["text"]:raise ValueError("Empty decoded annotation span")
    return text,spans


def represent_row(row,*,site,split,index,relation_labels=RELATION_LABELS):
    if not isinstance(row,dict) or not {"input","output"} <= row.keys():
        raise ValueError("RE records need input and output markup")
    text,sources=parse_annotation_markup(row["input"],SOURCE_LABELS)
    _,annotations=parse_annotation_markup(row["output"],set(relation_labels)|SOURCE_LABELS)
    if len(sources)!=1:raise ValueError("Expected exactly one marked primary source")
    source=dict(sources[0]);source["occurrence"]=occurrence_number(text,source["text"],source["start"])
    gold=[item for item in annotations if item["label"] in relation_labels]
    return dict(example_id=f"re:{site}:{split}:{index}",text=text,source=source,gold=gold)



PRIVATE_NER_LABELS = SOURCE_LABELS

def joined_token_view(tokens: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Join tokens with one ASCII space and return exact character offsets."""

    if not tokens:
        raise ValueError("tokens must be non-empty")
    if any(not isinstance(token, str) or token == "" for token in tokens):
        raise ValueError("every token must be a non-empty string")
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    return "".join(text_parts), tuple(offsets)


def strict_bio_token_spans(
    tags: Sequence[str], *, allowed_labels: Iterable[str]
) -> tuple[tuple[int, int, str], ...]:
    """Validate BIO tag sequences and return their token spans."""

    allowed = set(allowed_labels)
    if not allowed:
        raise ValueError("allowed_labels must be non-empty")
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    active_label: str | None = None

    def close(end: int) -> None:
        nonlocal start, active_label
        if start is not None and active_label is not None:
            spans.append((start, end, active_label))
        start = None
        active_label = None

    for index, tag in enumerate(tags):
        if not isinstance(tag, str):
            raise ValueError(f"tag {index} is not a string")
        if tag == "O":
            close(index)
            continue
        if "-" not in tag:
            raise ValueError(f"invalid BIO tag at token {index}")
        prefix, label = tag.split("-", 1)
        if not label or label not in allowed:
            raise ValueError(f"label outside schema at token {index}")
        if prefix == "B":
            close(index)
            start, active_label = index, label
        elif prefix == "I":
            if start is None or active_label != label:
                raise ValueError(
                    f"orphan or label-changing I tag at token {index}"
                )
        else:
            raise ValueError(f"invalid BIO prefix at token {index}")
    close(len(tags))
    return tuple(spans)


def _span_item(
    text: str,
    offsets: Sequence[tuple[int, int]],
    start_token: int,
    end_token: int,
    label: str,
) -> dict[str, object]:
    if start_token < 0 or end_token <= start_token or end_token > len(offsets):
        raise ValueError(
            f"invalid token span [{start_token},{end_token}) for {len(offsets)} tokens"
        )
    start = offsets[start_token][0]
    end = offsets[end_token - 1][1]
    surface = text[start:end]
    if not surface:
        raise RuntimeError("token-to-character span round trip failed")
    return {"start": start, "end": end, "text": surface, "label": label}


def build_private_ner_example(
    *,
    example_id: str,
    tokens: Sequence[str],
    tags: Sequence[str],
    allowed_labels: Iterable[str] = PRIVATE_NER_LABELS,
) -> dict[str, object]:
    """Build one public-contract-compatible NER example in memory."""

    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must have identical lengths")
    text, offsets = joined_token_view(tokens)
    spans = strict_bio_token_spans(tags, allowed_labels=allowed_labels)
    gold = [_span_item(text, offsets, start, end, label) for start, end, label in spans]
    return {
        "example_id": str(example_id),
        "text": text,
        "source": None,
        "gold": gold,
    }


def example(row, task, site, split, index):
    """Convert one clinical row to the example used for both training and evaluation."""
    if task == "RE":
        return represent_row(row, site=site, split=split, index=index)
    if not isinstance(row, dict) or not all(isinstance(row.get(k), list) for k in ("tokens", "tags")):
        raise ValueError("NER records need tokens and tags arrays")
    return build_private_ner_example(example_id=str(index), tokens=row["tokens"], tags=row["tags"])
