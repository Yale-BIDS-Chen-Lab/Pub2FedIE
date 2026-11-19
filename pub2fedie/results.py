"""Summarize one arm or contrast two paired sets of evaluation counts (CPU only)."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import t

from .core.inputs import CFG, training_sites

DEFAULT_BOOTSTRAP_SEED = CFG["bootstrap_seed"]


def load_arm(folder, sites, label=None):
    """Compute site-level F1 for one method, budget, round schedule, and target."""
    scores, sizes = {}, {}
    identity = None
    infer_sites = sites is None
    cohort = None
    for path in sorted(Path(folder).rglob("counts.json")):
        rec = json.loads(path.read_text())
        site = rec["site"]
        if not infer_sites and site not in sites:
            continue
        current = training_sites(rec)
        if cohort is not None and cohort != current:
            raise ValueError(f"{path}: inconsistent source cohort")
        cohort = current
        if infer_sites:
            sites = cohort
            if site not in sites:
                continue
        if rec["rounds"] < 1 or rec.get("round_number") != rec["rounds"]:
            raise ValueError(f"{path}: expected a final private checkpoint")
        target = rec["target"]
        if target not in ("global", site) or (target != "global" and site not in cohort):
            raise ValueError(f"{path}: invalid evaluation target")
        arm = (rec["model"], rec["task"], rec["method"], rec["steps"], rec["rounds"],
               "global" if target == "global" else "own-local", rec.get("rule"))
        if identity is not None and arm != identity:
            raise ValueError(f"{folder}: multiple configurations; provide one per directory")
        identity = arm
        key = (rec["block"], rec["seed"], site)
        if any(type(x) is not int or x<0 for x in key[:2]) or key in scores:
            raise ValueError(f"{path}: invalid or duplicate block/seed/site")
        if label is not None and label not in rec["by_label"]:
            raise ValueError(f"{path}: unknown label {label}")
        counts = rec["counts"] if label is None else rec["by_label"][label]
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        if any(type(x) is not int or x < 0 for x in (tp, fp, fn)):
            raise ValueError(f"{path}: expected nonnegative integer counts")
        if type(rec["rows"]) is not int or rec["rows"] <= 0:
            raise ValueError(f"{path}: empty or invalid test split")
        digest = rec.get("split_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}: missing or invalid test split fingerprint")
        signature=(rec["rows"],digest)
        if sizes.setdefault(site, signature) != signature:
            raise ValueError(f"{path}: inconsistent test split")
        scores[key] = 200 * tp / max(1, 2 * tp + fp + fn)
    if not scores:raise ValueError(f"{folder}: no matching evaluation endpoints")
    pairs={key[:2] for key in scores}
    if missing := {(*pair,site) for pair in pairs for site in sites} - scores.keys():
        raise ValueError(f"{folder}: missing {len(missing)} endpoints; first: {min(missing)}")
    return identity, sizes, {pair:np.mean([scores[pair+(site,)] for site in sites]) for pair in pairs}


def hierarchical_interval(values, seed):
    """Resample blocks, then runs within them; works for F1 or a paired difference."""
    values = [np.asarray(v) for v in values]
    blocks = len(values)
    mean = float(np.mean([v.mean() for v in values]))
    if sum(map(len,values))==1:return mean, None
    rng = np.random.default_rng(seed)
    outer = rng.integers(0, blocks, (20_000, blocks))
    if len({len(v) for v in values})==1:
        repeats=len(values[0])
        inner=rng.integers(0,repeats,(20_000,blocks,repeats))
        draws=np.asarray(values)[outer[:,:,None],inner].mean(axis=(1,2))
    else:
        sampled=np.empty(outer.shape)
        for block,v in enumerate(values):
            mask=outer==block
            sampled[mask]=v[rng.integers(len(v),size=(mask.sum(),len(v)))].mean(axis=1)
        draws=sampled.mean(axis=1)
    return mean, np.quantile(draws,[0.025,0.975]).tolist()


def paired_interval(left, right, seed=DEFAULT_BOOTSTRAP_SEED):
    return hierarchical_interval([np.subtract(a,b) for a,b in zip(left,right,strict=True)],seed)


def compare(left_dir, right_dir=None, domain="Source", label=None, bootstrap_seed=DEFAULT_BOOTSTRAP_SEED, sites=None):
    if sites is not None and (domain!="Source" or not sites or len(set(sites))!=len(sites)):
        raise ValueError("--sites needs distinct source sites and --domain Source")
    selected=sites if domain=="Source" else [domain]
    left, ls, la = load_arm(left_dir, selected, label)
    groups={}
    for pair in sorted(la):groups.setdefault(pair[0],[]).append(pair)
    fields = ("backbone", "task", "method", "steps", "rounds", "target", "aggregation_rule")

    def summarize(identity, scores):
        values=[np.array([scores[pair] for pair in pairs]) for pairs in groups.values()]
        mean,ci=hierarchical_interval(values,bootstrap_seed)
        return values,dict(zip(fields,identity),f1=mean,f1_95_ci=ci)

    a,first=summarize(left,la)
    result=dict(domain=domain,label=label,bootstrap_seed=bootstrap_seed,sites=sorted(ls),
                blocks=len(groups),left=first)
    if right_dir is None:
        return dict(result,runs=len(la))
    right, rs, rb = load_arm(right_dir, selected, label)
    if left[:2] != right[:2] or ls != rs:
        raise ValueError("Arms must use the same backbone, task and test splits")
    if la.keys()!=rb.keys():raise ValueError("Unpaired initialization block/private seed: missing runs in one arm")
    b,second=summarize(right,rb)
    delta, ci = paired_interval(a, b, bootstrap_seed)
    block_means=np.array([(x-y).mean() for x,y in zip(a,b,strict=True)])
    df=len(block_means)-1
    block_ci=None
    if df:
        half=float(t.ppf(0.975,df)*block_means.std(ddof=1)/np.sqrt(df+1))
        block_ci=[delta-half,delta+half]
    return dict(result,right=second,paired_runs=len(la),left_minus_right_f1_points=delta,
                paired_95_ci=ci,block_mean_differences=dict(zip(groups,block_means.tolist())),
                positive_blocks=int((block_means>0).sum()),block_t_df=df or None,block_t_95_ci=block_ci)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="directory of counts for the first arm")
    parser.add_argument("--right", type=Path, help="optional comparator for a paired difference")
    parser.add_argument("--domain", default="Source",help="Source or one evaluation site ID")
    parser.add_argument("--sites",nargs="+",help="explicit sites to average for Source")
    parser.add_argument("--label", help="optional single NER entity or RE modifier")
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    try:
        result = compare(args.left, args.right, args.domain, args.label, args.bootstrap_seed, args.sites)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
