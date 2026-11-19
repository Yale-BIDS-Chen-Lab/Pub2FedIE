"""Resolve input files from a path map or data directory."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "study.json").read_text())


def relative_to_manifest(manifest, value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest}: expected a file or directory path")
    return Path(manifest).parent / Path(value).expanduser()


def clinical_files(location, task, split):
    """Resolve a path map or a <task>/<split>/<site>.json directory."""
    root = Path(location).expanduser()
    other_split = "test" if split == "train" else "train"
    if root.is_dir():
        files = {p.stem: p for p in sorted((root/task/split).glob("*.json"))}
        other_paths = (root/task/other_split).glob("*.json")
    else:
        spec = json.loads(root.read_text())
        if not isinstance(spec, dict) or not isinstance(spec.get(task), dict):
            raise ValueError(f"{root}: missing {task} mapping")
        entries = spec[task].get(split)
        if not isinstance(entries, dict):
            raise ValueError(f"{root}: expected site-to-file mapping under {task}/{split}")
        files = {site: relative_to_manifest(root, path) for site, path in entries.items()}
        other = spec[task].get(other_split, {})
        if not isinstance(other,dict):
            raise ValueError(f"{root}: expected a site-to-file mapping for each split")
        other_paths = (relative_to_manifest(root, p) for p in other.values())
    if {p.resolve() for p in files.values()}.intersection(p.resolve() for p in other_paths):
        raise ValueError("The same clinical file cannot serve as both training and test data")
    if not files:
        raise ValueError(f"{root}: no inputs for {task}/{split}")
    for site in files:
        if not re.fullmatch(r"[\w-][\w.-]*", site) or site.lower() in {"source", "global"}:
            raise ValueError(f"Invalid site identifier: {site!r}")
    if split == "train" and any(s.casefold() == CFG["external"].casefold() for s in files):
        raise ValueError(f"{CFG['external']} is held out for evaluation")
    return files


def training_sites(record):
    sites = record.get("training_sites")
    if (not isinstance(sites, list) or not sites or not all(isinstance(s, str) and s for s in sites)
            or len(set(sites)) != len(sites)):
        raise ValueError("Missing or invalid training site list")
    return sorted(sites)


def split_digest(rows, task):
    fields = ("tokens", "tags") if task == "NER" else ("input", "output")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row[k] for k in fields], ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def records(location, task, site, split):
    paths = clinical_files(location, task, split)
    if site not in paths:
        raise ValueError(f"No {task}/{split} input for site {site}")
    rows = json.loads(paths[site].read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{task}/{split}/{site}: expected a nonempty array of records")
    return rows
