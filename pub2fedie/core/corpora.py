from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from .inputs import relative_to_manifest

TRAIN_INPUTS = None


def set_root(root):
    """Resolve public training inputs from a path map or a corpus directory."""
    global OFFICIAL_ROOT, E3C_NER_ROOT, E3C_RE_ROOT, BC5_ARCHIVE, NCBI_ARCHIVE, TRAIN_INPUTS
    root = Path(root).expanduser()
    TRAIN_INPUTS = None
    if root.is_file():
        spec = json.loads(root.read_text())
        resolve = lambda value: relative_to_manifest(root, value)
        try:
            BC5_ARCHIVE, NCBI_ARCHIVE, E3C_NER_ROOT = (resolve(spec[k]) for k in ("bc5cdr", "ncbi", "e3c_ner"))
            ddi = spec["ddi"]
            if not isinstance(ddi,list) or not ddi:raise ValueError("ddi must list one or more XML directories")
            TRAIN_INPUTS = dict(drugprot=tuple(resolve(spec["drugprot"][k]) for k in ("abstracts","entities","relations")),
                                biored=resolve(spec["biored"]),ddi=[resolve(d) for d in ddi],e3c_re=resolve(spec["e3c_re"]))
        except (KeyError,TypeError) as exc:
            raise ValueError(f"{root}: invalid public training path map: {exc}") from exc
        return
    if not root.is_dir():raise ValueError(f"Public data location not found: {root}")
    NCBI_ARCHIVE = root / "NCBI_corpus.zip"
    OFFICIAL_ROOT = root / "archives"
    E3C_NER_ROOT = root / "e3c" / "ner"
    E3C_RE_ROOT = root / "e3c" / "re"
    BC5_ARCHIVE = root / "BC5CDR.zip"


NER_DOMAINS = (
    "bc5cdr_disease",
    "bc5cdr_chemical",
    "ncbi_disease",
    "drugprot_entities",
    "biored_entities",
    "e3c_english_ner",
)

RE_DOMAINS = (
    "bc5cdr_cid",
    "drugprot_relations",
    "ddi2013_relations",
    "biored_relations",
    "e3c_english_re",
)




def _item(start: int, end: int, text: str, label: str) -> dict:
    if start < 0 or end <= start:
        raise ValueError(f"invalid span [{start}, {end})")
    return {"start": start, "end": end, "text": text, "label": label}


def _item_from_locations(locations: list[tuple[int, int]], text: str, label: str) -> dict:
    if not locations:
        raise ValueError("annotation has no locations")
    if len(locations) == 1:
        return _item(locations[0][0], locations[0][1], text, label)
    for start, end in locations:
        if start < 0 or end <= start:
            raise ValueError(f"invalid discontinuous location [{start}, {end})")
    return {
        "locations": [{"start": start, "end": end} for start, end in locations],
        "text": text,
        "label": label,
    }


def _validate_item(text: str, item: dict, context: str) -> None:
    if item.get("surface_validation") == "bounds_only":
        locations = item.get("locations") or [
            {"start": item["start"], "end": item["end"]}
        ]
        for location in locations:
            start, end = location["start"], location["end"]
            if start < 0 or end <= start or end > len(text) or not text[start:end].strip():
                raise RuntimeError(f"invalid offset [{start}, {end}) in {context}")
        return
    if "locations" in item:
        # A discontinuous mention can have normalized text; check each segment's bounds.
        for location in item["locations"]:
            start, end = location["start"], location["end"]
            if start < 0 or end <= start or end > len(text) or not text[start:end].strip():
                raise RuntimeError(
                    f"invalid discontinuous offset [{start}, {end}) in {context}"
                )
        return
    if "start" not in item or "end" not in item:
        return
    observed = text[item["start"] : item["end"]]
    if observed != item["text"]:
        raise RuntimeError(
            f"source span mismatch at [{item['start']}, {item['end']}) in {context}"
        )


def _locate_mentions(text: str, mentions: Iterable[tuple[str, str]]) -> list[dict]:
    used: set[tuple[int, int, str]] = set()
    items: list[dict] = []
    for mention, label in mentions:
        start_at = 0
        match: tuple[int, int, str] | None = None
        while True:
            start = text.find(mention, start_at)
            if start < 0:
                break
            candidate = (start, start + len(mention), label)
            if candidate not in used:
                match = candidate
                break
            start_at = start + 1
        if match is None:
            raise RuntimeError(f"no unused exact match for a mention of length {len(mention)}")
        used.add(match)
        items.append(_item(match[0], match[1], mention, label))
    return items


def _document(
    domain: str,
    split: str,
    document_id: str,
    task_family: str,
    schema: Iterable[str],
    examples: list[dict],
) -> dict:
    if not examples:
        raise RuntimeError(f"document has no examples: {domain}/{split}/{document_id}")
    return {
        "domain": domain,
        "split": split,
        "document_id": f"{domain}:{split}:{document_id}",
        "task_family": task_family,
        "schema": sorted(set(schema)),
        "examples": examples,
    }


def _ner_example(example_id: str, text: str, gold: list[dict]) -> dict:
    for item in gold:
        _validate_item(text, item, example_id)
    return {
        "example_id": example_id,
        "text": text,
        "source": None,
        "gold": gold,
    }


def _re_example(
    example_id: str,
    text: str,
    source: dict | None,
    gold: list[dict],
    symmetric: bool = False,
    target_must_be_span: bool = True,
) -> dict:
    if source is not None:
        _validate_item(text, source, example_id)
    for item in gold:
        _validate_item(text, item, example_id)
    return {
        "example_id": example_id,
        "text": text,
        "source": source,
        "gold": gold,
        "symmetric": symmetric,
        "target_must_be_span": target_must_be_span
        and not any("locations" in item for item in gold),
    }




def training_documents(dataset):
    """Read native public training archives in order; preserve all annotated spans."""
    if dataset=="ncbi_disease":
        with zipfile.ZipFile(NCBI_ARCHIVE) as z:
            lines=z.read("NCBI_corpus_training.txt").decode("utf-8-sig").splitlines()
        for line in lines:
            pmid,title,abstract=line.split("\t",2)
            markup=title+" "+abstract
            pos=cursor=0; parts=[]; gold=[]
            for m in re.finditer(r'<category="[^"]+">(.*?)</category>',markup):
                prefix=markup[pos:m.start()];surface=m[1];cursor+=len(prefix)
                gold.append(dict(start=cursor,end=cursor+len(surface),text=surface,entity_type="Disease"))
                parts.extend((prefix,surface));cursor+=len(surface);pos=m.end()
            yield dict(document_id=pmid,text="".join(parts)+markup[pos:],annotations=gold)
    else:
        with zipfile.ZipFile(BC5_ARCHIVE) as z:
            raw=z.read("CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt").decode("utf-8-sig")
        for doc in re.split(r"\n\s*\n",raw.replace("\r\n","\n").strip()):
            title,abstract,*lines=doc.splitlines()
            pmid,title=title.split("|t|",1); pid,abstract=abstract.split("|a|",1)
            if pid!=pmid:raise ValueError("PubTator title/abstract IDs differ")
            text=title+" "+abstract;gold=[]
            for line in lines:
                fields=line.split("\t")
                if fields[0]!=pmid:raise ValueError("PubTator annotation ID differs")
                if fields[1]=="CID":continue
                a,b=int(fields[1]),int(fields[2])
                gold.append(dict(start=a,end=b,text=text[a:b],entity_type=fields[4]))
            yield dict(document_id=pmid,text=text,annotations=gold)


def _canonical_ner(domain: str, split: str) -> list[dict]:
    if domain == "bc5cdr_disease":
        dataset = "bc5cdr"
        entity_types = {"Disease"}
    elif domain == "bc5cdr_chemical":
        dataset = "bc5cdr"
        entity_types = {"Chemical"}
    elif domain == "ncbi_disease":
        dataset = "ncbi_disease"
        entity_types = {"Disease"}
    else:
        raise KeyError(domain)
    if split!="train":raise ValueError("Only the public training split is used for preparation.")
    documents = []
    for row in training_documents(dataset):
        gold = [
            _item(
                int(annotation["start"]),
                int(annotation["end"]),
                annotation["text"],
                annotation["entity_type"],
            )
            for annotation in row["annotations"]
            if annotation["entity_type"] in entity_types
        ]
        example_id = f"{domain}:{split}:{row['document_id']}"
        documents.append(
            _document(
                domain,
                split,
                str(row["document_id"]),
                "NER",
                entity_types,
                [_ner_example(example_id, row["text"], gold)],
            )
        )
    return documents


def _drugprot_paths(split: str) -> tuple[Path, Path, Path | None]:
    if TRAIN_INPUTS is not None:return TRAIN_INPUTS["drugprot"]
    base = (
        OFFICIAL_ROOT
        / "drugprot_v1_2/verbatim/drugprot-gs-training-development"
    )
    mapping = {
        "train": (
            base / "training/drugprot_training_abstracs.tsv",
            base / "training/drugprot_training_entities.tsv",
            base / "training/drugprot_training_relations.tsv",
        ),
        "dev": (
            base / "development/drugprot_development_abstracs.tsv",
            base / "development/drugprot_development_entities.tsv",
            base / "development/drugprot_development_relations.tsv",
        ),
        "test": (
            base / "test-background/test_background_abstracts.tsv",
            base / "test-background/test_background_entities.tsv",
            None,
        ),
    }
    return mapping[split]


def _read_drugprot(split: str) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, list[dict]]]:
    abstract_path, entity_path, relation_path = _drugprot_paths(split)
    texts: dict[str, str] = {}
    with abstract_path.open(encoding="utf-8") as handle:
        for line in handle:
            pmid, title, abstract = line.rstrip("\n").split("\t", 2)
            if pmid in texts:
                raise RuntimeError(f"duplicate DrugProt PMID {pmid}")
            texts[pmid] = title + " " + abstract
    entities: dict[str, list[dict]] = defaultdict(list)
    with entity_path.open(encoding="utf-8") as handle:
        for line in handle:
            pmid, term_id, label, start, end, mention = line.rstrip("\n").split("\t")
            item = _item(int(start), int(end), mention, label)
            _validate_item(texts[pmid], item, f"DrugProt/{split}/{pmid}/{term_id}")
            item["id"] = term_id
            entities[pmid].append(item)
    relations: dict[str, list[dict]] = defaultdict(list)
    if relation_path is not None:
        with relation_path.open(encoding="utf-8") as handle:
            for line in handle:
                pmid, label, arg1, arg2 = line.rstrip("\n").split("\t")
                relations[pmid].append(
                    {
                        "label": label,
                        "source_id": arg1.removeprefix("Arg1:"),
                        "target_id": arg2.removeprefix("Arg2:"),
                    }
                )
    return texts, entities, relations


def _drugprot_ner(split: str) -> list[dict]:
    texts, entities, _ = _read_drugprot(split)
    schema = sorted({item["label"] for values in entities.values() for item in values})
    documents = []
    for pmid, text in texts.items():
        gold = [{key: value for key, value in item.items() if key != "id"} for item in entities[pmid]]
        documents.append(
            _document(
                "drugprot_entities",
                split,
                pmid,
                "NER",
                schema,
                [_ner_example(f"drugprot_entities:{split}:{pmid}", text, gold)],
            )
        )
    return documents


def _drugprot_re(split: str) -> list[dict]:
    if split == "test":
        raise RuntimeError("DrugProt test relations are not publicly labeled; use dev as query")
    texts, entities, relations = _read_drugprot(split)
    schema = sorted({relation["label"] for values in relations.values() for relation in values})
    documents = []
    for pmid, text in texts.items():
        by_id = {item["id"]: item for item in entities[pmid]}
        outgoing: dict[str, list[dict]] = defaultdict(list)
        for relation in relations[pmid]:
            if relation["source_id"] not in by_id or relation["target_id"] not in by_id:
                raise RuntimeError(f"DrugProt relation refers to an unknown entity in {pmid}")
            target = by_id[relation["target_id"]]
            outgoing[relation["source_id"]].append(
                _item(target["start"], target["end"], target["text"], relation["label"])
            )
        examples = []
        for source in entities[pmid]:
            if source["label"] != "CHEMICAL":
                continue
            source_view = {key: value for key, value in source.items() if key != "id"}
            examples.append(
                _re_example(
                    f"drugprot_relations:{split}:{pmid}:{source['id']}",
                    text,
                    source_view,
                    outgoing[source["id"]],
                )
            )
        # A handful of official documents contain no CHEMICAL source mention,
        # so they define no source-conditioned RE example. Keep the source
        # files untouched and expose only the task-defined examples.
        if examples:
            documents.append(
                _document(
                    "drugprot_relations",
                    split,
                    pmid,
                    "RE",
                    schema,
                    examples,
                )
            )
    return documents


def _biored_path(split: str) -> Path:
    if TRAIN_INPUTS is not None:return TRAIN_INPUTS["biored"]
    filename = {"train": "Train", "dev": "Dev", "test": "Test"}[split]
    return OFFICIAL_ROOT / f"biored/verbatim/BioRED/{filename}.BioC.JSON"


def _bioc_json_text(document: dict, context: str) -> str:
    passages = document["passages"]
    end = max(int(passage["offset"]) + len(passage["text"]) for passage in passages)
    characters = [" "] * end
    occupied = [False] * end
    for passage in passages:
        offset = int(passage["offset"])
        for index, character in enumerate(passage["text"], start=offset):
            if occupied[index] and characters[index] != character:
                raise RuntimeError(f"overlapping BioC passages disagree in {context}")
            characters[index] = character
            occupied[index] = True
    return "".join(characters)


def _read_biored(split: str) -> list[dict]:
    path = _biored_path(split)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["documents"]


def _biored_annotations(document: dict, text: str, context: str) -> list[dict]:
    items = []
    for passage in document["passages"]:
        for annotation in passage.get("annotations", []):
            locations = annotation["locations"]
            if len(locations) != 1:
                raise RuntimeError(f"discontinuous BioRED annotation in {context}")
            location = locations[0]
            item = _item(
                int(location["offset"]),
                int(location["offset"]) + int(location["length"]),
                annotation["text"],
                annotation["infons"]["type"],
            )
            _validate_item(text, item, context)
            item["id"] = annotation["id"]
            item["concept_id"] = annotation["infons"].get("identifier", "")
            items.append(item)
    return items


def _biored_ner(split: str) -> list[dict]:
    rows = _read_biored(split)
    documents = []
    all_schema: set[str] = set()
    staged = []
    for row in rows:
        text = _bioc_json_text(row, f"BioRED/{split}/{row['id']}")
        annotations = _biored_annotations(row, text, f"BioRED/{split}/{row['id']}")
        gold = [
            {key: value for key, value in item.items() if key not in {"id", "concept_id"}}
            for item in annotations
        ]
        all_schema.update(item["label"] for item in gold)
        staged.append((row["id"], text, gold))
    for document_id, text, gold in staged:
        documents.append(
            _document(
                "biored_entities",
                split,
                str(document_id),
                "NER",
                all_schema,
                [_ner_example(f"biored_entities:{split}:{document_id}", text, gold)],
            )
        )
    return documents


def _biored_re(split: str) -> list[dict]:
    rows = _read_biored(split)
    schema = sorted(
        {
            relation["infons"]["type"]
            for row in rows
            for relation in row.get("relations", [])
        }
    )
    documents = []
    for row in rows:
        text = _bioc_json_text(row, f"BioRED/{split}/{row['id']}")
        annotations = _biored_annotations(row, text, f"BioRED/{split}/{row['id']}")
        by_concept: dict[str, list[dict]] = defaultdict(list)
        for item in annotations:
            # One official BioRED mention can carry multiple identifiers.
            # Comma separates alternatives; pipe is part of official sequence-
            # variant identifiers such as c|INS|5382|C.
            for concept_id in item["concept_id"].split(","):
                concept_id = concept_id.strip()
                if concept_id:
                    by_concept[concept_id].append(item)
        outgoing: dict[str, list[dict]] = defaultdict(list)
        for relation in row.get("relations", []):
            infons = relation["infons"]
            source_id = infons["entity1"]
            target_id = infons["entity2"]
            if source_id not in by_concept or target_id not in by_concept:
                raise RuntimeError(f"BioRED dangling relation in document {row['id']}")
            for source in by_concept[source_id]:
                for target in by_concept[target_id]:
                    outgoing[source["id"]].append(
                        _item(
                            target["start"],
                            target["end"],
                            target["text"],
                            infons["type"],
                        )
                    )
        examples = []
        for source in annotations:
            source_view = {
                key: value
                for key, value in source.items()
                if key not in {"id", "concept_id"}
            }
            examples.append(
                _re_example(
                    f"biored_relations:{split}:{row['id']}:{source['id']}",
                    text,
                    source_view,
                    outgoing[source["id"]],
                )
            )
        documents.append(
            _document(
                "biored_relations",
                split,
                str(row["id"]),
                "RE",
                schema,
                examples,
            )
        )
    return documents


def _bc5_member(split: str) -> str:
    split_name = {"train": "Training", "dev": "Development", "test": "Test"}[split]
    return f"CDR_Data/CDR.Corpus.v010516/CDR_{split_name}Set.BioC.xml"


def _xml_infons(element: ET.Element) -> dict[str, str]:
    return {infon.attrib["key"]: infon.text or "" for infon in element.findall("infon")}


def _bc5_re(split: str) -> list[dict]:
    member = _bc5_member(split)
    with zipfile.ZipFile(BC5_ARCHIVE) as archive:
        root = ET.fromstring(archive.read(member))
    documents = []
    for row in root.findall("document"):
        document_id = row.findtext("id") or ""
        passages = row.findall("passage")
        end = max(
            int(passage.findtext("offset") or "0") + len(passage.findtext("text") or "")
            for passage in passages
        )
        characters = [" "] * end
        annotations = []
        for passage in passages:
            offset = int(passage.findtext("offset") or "0")
            passage_text = passage.findtext("text") or ""
            characters[offset : offset + len(passage_text)] = list(passage_text)
            for annotation in passage.findall("annotation"):
                infons = _xml_infons(annotation)
                locations = annotation.findall("location")
                if not locations:
                    raise RuntimeError(f"BC5 annotation without location in {document_id}")
                item = _item_from_locations(
                    [
                        (
                            int(location.attrib["offset"]),
                            int(location.attrib["offset"])
                            + int(location.attrib["length"]),
                        )
                        for location in locations
                    ],
                    annotation.findtext("text") or "",
                    infons["type"],
                )
                # Some BC5 offsets disagree with the annotated text; keep both and check bounds.
                item["surface_validation"] = "bounds_only"
                item["id"] = annotation.attrib["id"]
                item["concept_id"] = infons.get("MESH", "")
                annotations.append(item)
        text = "".join(characters)
        for item in annotations:
            _validate_item(text, item, f"BC5/{split}/{document_id}")
        by_concept: dict[str, list[dict]] = defaultdict(list)
        for item in annotations:
            for concept_id in item["concept_id"].split("|"):
                by_concept[concept_id].append(item)
        outgoing: dict[str, list[dict]] = defaultdict(list)
        for relation in row.findall("relation"):
            infons = _xml_infons(relation)
            if infons.get("relation") != "CID":
                raise RuntimeError(f"unexpected BC5 relation type in {document_id}")
            for source in by_concept.get(infons.get("Chemical", ""), []):
                for target in by_concept.get(infons.get("Disease", ""), []):
                    outgoing[source["id"]].append(
                        {
                            key: value
                            for key, value in target.items()
                            if key not in {"label", "id", "concept_id"}
                        }
                        | {"label": "CID"}
                    )
        examples = []
        for source in annotations:
            if source["label"] != "Chemical":
                continue
            source_view = {
                key: value
                for key, value in source.items()
                if key not in {"id", "concept_id"}
            }
            examples.append(
                _re_example(
                    f"bc5cdr_cid:{split}:{document_id}:{source['id']}",
                    text,
                    source_view,
                    outgoing[source["id"]],
                )
            )
        documents.append(
            _document(
                "bc5cdr_cid",
                split,
                document_id,
                "RE",
                ["CID"],
                examples,
            )
        )
    return documents


def _ddi_roots(split: str) -> list[Path]:
    if TRAIN_INPUTS is not None:return TRAIN_INPUTS["ddi"]
    base = OFFICIAL_ROOT / "ddi2013/verbatim/DDICorpus"
    if split == "train":
        return [base / "Train/DrugBank", base / "Train/MedLine"]
    if split == "test":
        test = base / "Test/Test for DDI Extraction task"
        return [test / "DrugBank", test / "MedLine"]
    raise RuntimeError("DDI2013 has official train/test only")


def _parse_ddi_offset(raw: str, context: str) -> list[tuple[int, int]]:
    locations = []
    if not isinstance(raw, str):
        raise RuntimeError(f"invalid DDI offset in {context}")
    for segment in raw.split(";"):
        try:
            start_raw, end_raw = segment.split("-", 1)
            locations.append((int(start_raw), int(end_raw) + 1))
        except ValueError:
            raise RuntimeError(f"invalid DDI offset in {context}") from None
    return locations


def _ddi_re(split: str) -> list[dict]:
    paths = sorted(path for root in _ddi_roots(split) for path in root.glob("*.xml"))
    schema: set[str] = set()
    staged = []
    for path in paths:
        root = ET.parse(path).getroot()
        document_id = root.attrib["id"]
        examples = []
        for sentence in root.findall("sentence"):
            sentence_id = sentence.attrib["id"]
            text = sentence.attrib["text"]
            entities = {}
            for element in sentence.findall("entity"):
                locations = _parse_ddi_offset(
                    element.attrib["charOffset"], f"{path}:{element.attrib['id']}"
                )
                item = _item_from_locations(
                    locations, element.attrib["text"], element.attrib["type"]
                )
                _validate_item(text, item, f"{path}:{element.attrib['id']}")
                entities[element.attrib["id"]] = item
            outgoing: dict[str, list[dict]] = defaultdict(list)
            for pair in sentence.findall("pair"):
                if pair.attrib.get("ddi") != "true":
                    continue
                # One positive DDI pair has no annotated subtype.
                label = pair.attrib.get("type", "UNSPECIFIED_DDI")
                schema.add(label)
                left_id, right_id = pair.attrib["e1"], pair.attrib["e2"]
                if left_id not in entities or right_id not in entities:
                    raise RuntimeError(f"DDI pair refers to an unknown entity in {path}")
                left, right = entities[left_id], entities[right_id]
                outgoing[left_id].append(
                    {key: value for key, value in right.items() if key != "label"}
                    | {"label": label}
                )
                outgoing[right_id].append(
                    {key: value for key, value in left.items() if key != "label"}
                    | {"label": label}
                )
            for source_id, source in entities.items():
                examples.append(
                    _re_example(
                        f"ddi2013_relations:{split}:{sentence_id}:{source_id}",
                        text,
                        source,
                        outgoing[source_id],
                        symmetric=True,
                    )
                )
        if examples:
            staged.append((document_id, examples))
    return [
        _document(
            "ddi2013_relations",
            split,
            document_id,
            "RE",
            schema,
            examples,
        )
        for document_id, examples in staged
    ]


def _e3c_ner(split: str) -> list[dict]:
    from datasets import load_from_disk

    split_name = {"train": "train", "dev": "validation", "test": "test"}[split]
    dataset = load_from_disk(str(E3C_NER_ROOT))[split_name]
    schema = sorted(
        {
            entity["type"]
            for row in dataset
            for entity in row["entities"]
        }
    )
    documents = []
    for index, row in enumerate(dataset):
        gold = _locate_mentions(
            row["text"],
            [(entity["entity_text"], entity["type"]) for entity in row["entities"]],
        )
        documents.append(
            _document(
                "e3c_english_ner",
                split,
                str(index),
                "NER",
                schema,
                [_ner_example(f"e3c_english_ner:{split}:{index}", row["text"], gold)],
            )
        )
    return documents


def _e3c_re(split: str) -> list[dict]:
    filename = {"train": "en_train.json", "dev": "en_val.json", "test": "en_test.json"}[split]
    path = TRAIN_INPUTS["e3c_re"] if TRAIN_INPUTS is not None else E3C_RE_ROOT / filename
    rows = json.loads(path.read_text(encoding="utf-8"))
    schema = sorted({relation[1] for row in rows for relation in row["relations"]})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        # An E3C result may be implicit in the sentence; keep the annotated value without offsets.
        gold = []
        for value, label in row["relations"]:
            start = row["text"].find(value)
            if start >= 0:
                gold.append(_item(start, start + len(value), value, label))
            else:
                gold.append({"text": value, "label": label})
        grouped[str(row["text_id"])].append(
            _re_example(
                f"e3c_english_re:{split}:{row['text_id']}:{row['sentence_id']}",
                row["text"],
                None,
                gold,
                target_must_be_span=False,
            )
        )
    return [
        _document(
            "e3c_english_re",
            split,
            text_id,
            "RE",
            schema,
            examples,
        )
        for text_id, examples in grouped.items()
    ]


def load_public_domain(domain: str, split: str) -> list[dict]:
    if TRAIN_INPUTS is not None and split!="train":raise ValueError("The public path map describes training inputs only")
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid split: {split}")
    if domain in {"bc5cdr_disease", "bc5cdr_chemical", "ncbi_disease"}:
        return _canonical_ner(domain, split)
    if domain == "drugprot_entities":
        return _drugprot_ner(split)
    if domain == "biored_entities":
        return _biored_ner(split)
    if domain == "e3c_english_ner":
        return _e3c_ner(split)
    if domain == "bc5cdr_cid":
        return _bc5_re(split)
    if domain == "drugprot_relations":
        return _drugprot_re(split)
    if domain == "ddi2013_relations":
        return _ddi_re(split)
    if domain == "biored_relations":
        return _biored_re(split)
    if domain == "e3c_english_re":
        return _e3c_re(split)
    raise KeyError(domain)
