"""Train federated client adapters, or reaggregate saved one-round SFT-Cont clients."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from .core import adapters as ada
from .core.clinical import example
from .core.inputs import clinical_files, records, split_digest, training_sites
from .core.model import (
    CFG, SIMULATED_CONTINUATION, activate, configure_torch_reproducibility, feature, load, load_state, loss, output_directory,
)
from .core.sampling import hierarchical_schedule_seed, master_training_schedule

PUBLIC_PREPARATION = {
    "SFT-Cont": "SFT", "SFT-Res": "SFT", "Meta-Cont": "Meta", "Meta-Res": "Meta",
    SIMULATED_CONTINUATION: SIMULATED_CONTINUATION,
}


def seed(args,site):
    return hierarchical_schedule_seed(task_family=args.task,site=site,initializer_seed=args.block,private_seed=args.seed)


def training_orders(data, args, steps, batch_size):
    return {s: master_training_schedule(number_of_rows=len(rows), optimizer_steps=steps,
                                        seed=seed(args, s), batch_size=batch_size)
            for s, rows in data.items()}


def validated_panel(data,orders,task,indices):
    """Validate held-out training rows for each site's validation panel."""
    if not isinstance(indices,dict) or set(indices)!=set(data):raise ValueError("Panel must specify every training site")
    def text_hash(s,i):
        ex=example(data[s][i],task,s,"train",i)
        return hashlib.sha256(ex["text"].encode()).digest()
    seen={text_hash(s,i) for s in data for i in set(orders[s])}
    panel=[]
    for s in data:
        if not isinstance(indices[s],list) or not indices[s]:raise ValueError("Empty or invalid validation panel")
        for i in indices[s]:
            if type(i)!=int or not 0<=i<len(data[s]):raise ValueError("Invalid train-side panel index")
            h=text_hash(s,i)
            if h in seen:raise ValueError("Exposed or repeated panel text")
            seen.add(h);panel.append(example(data[s][i],task,s,"train",i))
    return panel


def make_panel(data,args,*,batch_size,fixed_clients=False):
    # A longer excluded training prefix lets multiple budgets share a panel.
    hashes={s:[hashlib.sha256(example(row,args.task,s,"train",i)["text"].encode()).digest()
               for i,row in enumerate(data[s])] for s in data}
    exposed=set()
    size=getattr(args,"panel_size",None)
    if size is None:size=CFG["clinical"]["validation_rows_per_site"]
    exclude=getattr(args,"panel_exclude_steps",None)
    if size<1 or (exclude is not None and exclude<args.steps):
        raise ValueError("Use a positive panel size and exclude at least the actual training steps")
    for s,order in training_orders(data,args,exclude or args.steps,batch_size).items():
        exposed.update(hashes[s][i] for i in order)
    panel_key="relation-validation" if args.task=="RE" and not fixed_clients else "validation"
    indices={}
    for s in data:
        ranked=sorted(range(len(data[s])),key=lambda i:hashlib.sha256(
            f"{panel_key}|{args.task}|{args.block}|{args.seed}|{s}|{i}".encode()).digest())
        indices[s]=[]
        for i in ranked:
            h=hashes[s][i]
            if h not in exposed:
                indices[s].append(i);exposed.add(h)
            if len(indices[s])==size:break
        if len(indices[s])!=size:raise ValueError("Insufficient unique unexposed training records for the requested validation panel")
    return indices


def check_schedule(args):
    if args.steps<=0 or args.rounds<=0 or args.steps%args.rounds:
        raise ValueError("Total steps must be positive and evenly divisible by positive rounds")
    if min(args.block,args.seed)<0:raise ValueError("Seeds must be nonnegative")


def panel_loss(model,features,pad):
    # Validation forward passes use fixed chunks, independent of the optimizer batch.
    return sum(float(loss(model,features[i:i+8],pad))*len(features[i:i+8])
               for i in range(0,len(features),8))/len(features)


def train(args):
    check_schedule(args)
    preparation=PUBLIC_PREPARATION.get(args.method)
    prepared=preparation is not None;res=args.method.endswith("-Res");fed=args.method=="Fed-MedLoRA+"
    snapshots=sorted(set(getattr(args,"save_steps",None) or []))
    if snapshots and (args.rounds!=1 or fed or any(not 0<b<args.steps for b in snapshots)):
        raise ValueError("Snapshots must be positive earlier steps on a non-Fed 1R trajectory")
    if bool(args.public)!=prepared or (args.panel and not fed):raise ValueError("Public initialization and panel must match method")
    sites=sorted(clinical_files(args.clinical,args.task,"train"))
    data={s:records(args.clinical,args.task,s,"train") for s in sites}
    batch_size=CFG["clinical"]["effective_batch"]
    orders=training_orders(data,args,args.steps,batch_size)
    panel=[]
    if fed:
        indices=(json.loads(Path(args.panel).read_text()) if args.panel else
                 make_panel(data,args,batch_size=batch_size))
        panel=validated_panel(data,orders,args.task,indices)
    out=output_directory(args.out)
    configure_torch_reproducibility(seed(args,"GLOBAL")%2**31)
    slots=["client","global"] if prepared and not res else ["public","client","global"]
    m,tok=load(args.model,args.snapshot,slots); schema=CFG["schemas"][args.task]
    active=["public","client"] if res else ["client"]
    if prepared:
        public=torch.load(args.public,weights_only=True,map_location="cpu")
        if (public["model"],public["block"],public["preparation"])!=(args.model,args.block,preparation):
            raise ValueError("Public adapter does not match the model, block, and preparation")
        load_state(m,"public" if res else "client",public["state"],"meta")
    if not prepared or res:load_state(m,"client",ada.snapshot_adapter(m,"global"),"global")
    initial=ada.snapshot_adapter(m,"client")
    val=[feature(tok,args.model,args.task,schema,ex) for ex in panel]
    meta=dict(model=args.model,task=args.task,method=args.method,steps=args.steps,rounds=args.rounds,
              block=args.block,seed=args.seed,training_sites=sites,effective_batch=batch_size)
    train_hashes = ({s: split_digest(data[s], args.task) for s in sites}
                    if args.method == "SFT-Cont" and args.rounds == 1 else {})
    def save(state,target,round_number,budget=None):
        folder=out if budget is None else out/f"step{budget}"
        if budget is not None:folder.mkdir(exist_ok=True)
        identity=dict(meta,steps=budget or args.steps)
        if target in train_hashes:
            identity["training_split_sha256"] = train_hashes[target]
        torch.save(dict(**identity,target=target,round_number=round_number,state=state,
                        frozen_public=ada.snapshot_adapter(m,"public") if res else None),
                   folder/f"r{round_number}_{target}.pt")
    per=args.steps//args.rounds
    for rd in range(args.rounds):
        factors=[];states=[];losses=[]
        for s in sites:
            ada.restore_adapter(m,"client",initial);params=activate(m,active,["client"]);m.train()
            opt=torch.optim.AdamW(params,lr=CFG["clinical"]["lr"],weight_decay=CFG["weight_decay"])
            for j in range(rd*per,(rd+1)*per):
                opt.zero_grad(set_to_none=True)
                for i in orders[s][j*batch_size:(j+1)*batch_size]:
                    ex=example(data[s][i],args.task,s,"train",i)
                    (loss(m,[feature(tok,args.model,args.task,schema,ex)],tok.pad_token_id)/batch_size).backward()
                torch.nn.utils.clip_grad_norm_(params,CFG["clip"],error_if_nonfinite=True);opt.step()
                if j+1 in snapshots:save(ada.snapshot_adapter(m,"client"),s,rd+1,j+1)
            state=ada.snapshot_adapter(m,"client");save(state,s,rd+1)
            if fed:
                states.append(state);m.eval()
                with torch.no_grad():
                    losses.append(panel_loss(m,val,tok.pad_token_id))
            else:factors.append(ada.collect_adapter_low_rank(m,"client"))
        if fed:
            weights=torch.softmax(-torch.tensor(losses,dtype=torch.float64),dim=0)
            initial={k:sum(w*v[k].double() for w,v in zip(weights,states)).float() for k in states[0]}
        else:
            avg=ada.aggregate_adapter_updates(factors,CFG["rank"],ada.adapter_scalings(m,"client"))
            ada.load_adapter_factors(m,"client",avg);initial=ada.snapshot_adapter(m,"client")
        save(initial,"global",rd+1)
    # Aggregate saved client snapshots after training for evaluation at each budget.
    for b in snapshots:
        components=[]
        for s in sites:
            state=torch.load(out/f"step{b}"/f"r1_{s}.pt",weights_only=True,map_location="cpu")["state"]
            ada.restore_adapter(m,"client",state)
            components.append(ada.collect_adapter_low_rank(m,"client"))
        avg=ada.aggregate_adapter_updates(components,CFG["rank"],ada.adapter_scalings(m,"client"))
        ada.load_adapter_factors(m,"client",avg)
        save(ada.snapshot_adapter(m,"client"),"global",1,b)


def reaggregate(args):
    """Apply the requested aggregation rules to the same saved client adapters."""
    clients = Path(args.reaggregate)
    by_site = {p.stem[3:]: torch.load(p, weights_only=True, map_location="cpu")
               for p in sorted(clients.glob("r1_*.pt")) if p.stem != "r1_global"}
    if not by_site:
        raise ValueError("Missing first-round client checkpoints")
    sites = training_sites(next(iter(by_site.values())))
    if set(by_site) != set(sites):
        raise ValueError("Client checkpoints do not cover exactly the recorded training sites")
    cells = [by_site[s] for s in sites]
    fields = ("model", "task", "method", "rounds", "steps", "block", "seed", "round_number", "effective_batch")
    info = {k: cells[0][k] for k in fields}
    if any(any(c[k] != v for k, v in info.items()) or c["target"] != site
           or training_sites(c) != sites for c, site in zip(cells, sites)):
        raise ValueError("Mismatched client identities")
    if (info["method"], info["rounds"], info["round_number"]) != ("SFT-Cont", 1, 1) or any(c["frozen_public"] is not None for c in cells):
        raise ValueError("The fixed-client ablation requires continued one-round SFT adapters")
    run = SimpleNamespace(**info, panel_size=args.panel_size, panel_exclude_steps=args.panel_exclude_steps)
    if sorted(clinical_files(args.clinical, run.task, "train")) != sites:
        raise ValueError("Training site IDs and client checkpoints do not match")

    rules = sorted(set(args.rules or "ABCD"))
    equal = [1 / len(sites)] * len(sites)
    if {"B", "D"}.intersection(rules):
        data = {s: records(args.clinical, run.task, s, "train") for s in sites}
        if any(split_digest(data[s], run.task) != by_site[s].get("training_split_sha256") for s in sites):
            raise ValueError("Validation weighting requires the training rows in their original order")
        orders = training_orders(data, run, run.steps, run.effective_batch)
        ids = (json.loads(Path(args.panel).read_text()) if args.panel else
               make_panel(data, run, batch_size=run.effective_batch, fixed_clients=True))
        panel = validated_panel(data, orders, run.task, ids)
        configure_torch_reproducibility(seed(run, "GLOBAL") % 2**31)
        model, tok = load(run.model, args.snapshot, ["client"])
        activate(model, ["client"], [])
        model.eval()
        features = [feature(tok, run.model, run.task, CFG["schemas"][run.task], ex) for ex in panel]
        losses = []
        with torch.no_grad():
            for c in cells:
                load_state(model, "client", c["state"], "client")
                losses.append(panel_loss(model, features, tok.pad_token_id))
        validation = torch.softmax(-torch.tensor(losses, dtype=torch.float64), dim=0)

    out = output_directory(args.out)
    states = [c["state"] for c in cells]
    for rule in rules:
        state = ada.combine_client_states(states, validation if rule in "BD" else equal,
                                          "effective" if rule in "AB" else "factor",
                                          rank=CFG["rank"], scaling=CFG["alpha"] / CFG["rank"])
        torch.save(dict(**info, state=state, frozen_public=None, rule=rule, target="global",
                        training_sites=sites), out/f"{rule}.pt")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clinical",required=True)
    p.add_argument("--snapshot",help="local backbone (training or validation-weighted rules B/D)")
    p.add_argument("--out",required=True)
    p.add_argument("--panel-size",type=int,help="validation examples per site for automatic Fed panels")
    p.add_argument("--panel-exclude-steps",type=int,help="optional longer common training prefix to exclude")
    p.add_argument("--panel",help="optional JSON with held-out training indices per site")
    fit=p.add_argument_group("training")
    fit.add_argument("--model",choices=CFG["models"])
    fit.add_argument("--task",choices=CFG["schemas"])
    fit.add_argument("--method",choices=[*PUBLIC_PREPARATION,"No-Public","Fed-MedLoRA+"])
    for k in ("steps","rounds","block","seed"):
        fit.add_argument("--"+k,type=int)
    fit.add_argument("--public")
    fit.add_argument("--save-steps",type=int,nargs="+",help="optional intermediate budgets on a one-round trajectory")
    fixed=p.add_argument_group("reuse saved clients (no training)")
    fixed.add_argument("--reaggregate",metavar="DIR",help="directory containing the one-round SFT-Cont client checkpoints")
    fixed.add_argument("--rules",choices=list("ABCD"),nargs="+",help="aggregation rules (default: A B C D)")
    args=p.parse_args(argv)
    required=("model","task","method","steps","rounds","block","seed")
    if args.reaggregate is not None:
        extra=[k for k in (*required,"public","save_steps") if getattr(args,k) is not None]
        if extra:
            p.error("training options do not apply to --reaggregate: "+", ".join("--"+k.replace("_","-") for k in extra))
        if set(args.rules or "ABCD").intersection("BD") and not args.snapshot:
            p.error("--snapshot is required for validation-weighted rules B and D")
        return reaggregate(args)
    if args.rules is not None:
        p.error("--rules requires --reaggregate")
    missing=[k for k in (*required,"snapshot") if getattr(args,k) is None]
    if missing:
        p.error("training requires "+", ".join("--"+k for k in missing))
    return train(args)


if __name__=="__main__":
    main()
