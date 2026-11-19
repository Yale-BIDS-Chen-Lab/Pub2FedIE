from __future__ import annotations
import random
import hashlib
from collections import defaultdict


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def sample_documents(documents: list[dict], count: int, seed: int, key: str) -> list[dict]:
    if len(documents) < count:
        raise RuntimeError(f"{key} has {len(documents)} documents, needs {count}")
    indices = list(range(len(documents)))
    random.Random(stable_seed(seed, key)).shuffle(indices)
    return [documents[index] for index in indices[:count]]


def partition_documents(
    documents: list[dict], support_count: int, query_count: int, seed: int, key: str
) -> tuple[list[dict], list[dict]]:
    selected = sample_documents(documents, support_count + query_count, seed, key)
    return selected[:support_count], selected[support_count:]


def group_features_by_document(features: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for feature in features:
        document_id = feature.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("feature lacks a nonempty document_id")
        grouped[document_id].append(feature)
    if not grouped:
        raise RuntimeError("cannot group an empty feature collection")
    return dict(sorted(grouped.items()))


def choose_document_then_target_class(
    grouped: dict[str, list[dict]],
    seed: int,
    prefer_positive: bool,
    *parts: str,
) -> dict:
    """Choose a document uniformly, then prefer a positive or empty target.

    If the selected document lacks the requested target class, selection falls
    back to all examples in that same document; it never switches documents or
    removes an official example from the view.
    """

    document_ids = sorted(grouped)
    if not document_ids:
        raise RuntimeError("no documents available")
    document_id = document_ids[
        stable_seed(seed, *parts, "document") % len(document_ids)
    ]
    examples = grouped[document_id]
    preferred = [
        row for row in examples if bool(row["target_has_items"]) == prefer_positive
    ]
    candidates = preferred or examples
    index = stable_seed(seed, *parts, "example") % len(candidates)
    return candidates[index]


def family_balanced_inner_steps(outer_step_zero_based: int) -> int:
    """Alternate 2/5 within each task family when task families alternate.

    Outer steps 0,2,4,... are NER and 1,3,5,... are RE.  Using the raw outer
    parity for both choices would confound NER with 2 steps and RE with 5.
    Indexing by the family-local episode (outer_step // 2) yields
    NER: 2,5,2,5,... and RE: 2,5,2,5,... .
    """

    if outer_step_zero_based < 0:
        raise ValueError("outer step must be nonnegative")
    return (2, 5)[(outer_step_zero_based // 2) % 2]


def selection_context(parts):
    if len(parts) != 3:
        raise ValueError(f"Expected a view, outer step, and sampling role: {parts}")
    domain, outer, last = parts
    outer = int(outer)
    if outer < 0:
        raise ValueError("negative outer step")
    if last in {"support", "query"}:
        role, inner = last, None
        # Per family: (2,+),(5,+),(2,-),(5,-). global//2 alone
        # would confound preferred class with the existing 2/5 inner schedule.
        prefer_positive = (outer // 4) % 2 == 0
    else:
        role, inner = "inner", int(last)
        if inner < 0:
            raise ValueError("negative inner step")
        prefer_positive = inner % 2 == 0
    return dict(domain=domain, outer_step=outer, family=("NER", "RE")[outer % 2],
                family_episode=outer // 2, role=role, inner_step=inner,
                prefer_positive=prefer_positive)


def select(grouped, seed, parts, chooser):
    context = selection_context(parts)
    result = chooser(grouped, seed, context["prefer_positive"], *parts)
    if result.get("task_family", context["family"]) != context["family"]:
        raise RuntimeError("Sampler received a document from the wrong task family")
    return result, context


def hierarchical_schedule_seed(
    *, task_family: str, site: str, initializer_seed: int, private_seed: int
) -> int:
    """Derive a schedule seed from the task, site, initialization block, and private seed."""

    payload = (
        f"client-order|{task_family}|{site}|"
        f"{int(initializer_seed)}|{int(private_seed)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def master_training_schedule(
    *, number_of_rows: int, optimizer_steps: int, seed: int, batch_size: int
) -> tuple[int, ...]:
    """Return one seeded permutation cycled to the exact matched step budget."""

    if number_of_rows <= 0:
        raise ValueError("number_of_rows must be positive")
    if optimizer_steps < 0:
        raise ValueError("optimizer_steps must be non-negative")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("effective_batch must be a positive integer")
    order = list(range(number_of_rows))
    random.Random(int(seed)).shuffle(order)
    required = optimizer_steps * batch_size
    return tuple(order[index % number_of_rows] for index in range(required))
