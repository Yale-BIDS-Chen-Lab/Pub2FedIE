"""Evaluate greedy predictions with exact matching and write overall and per-label counts."""
import argparse
from collections import Counter
import json
import torch

from .core import scoring
from .core.clinical import example
from .core.inputs import records, split_digest, training_sites
from .core.model import CFG, activate, chat_prompt, configure_torch_reproducibility, load, load_state, output_directory


def score_one(ex,task,schema,raw):
    parsed,valid,_=scoring.parse_prediction(raw,task,set(schema))
    all_counts=scoring.example_counts(ex,task,parsed)
    all_counts["fp"]+=int(not valid)
    typed={label:scoring.example_counts(
        dict(ex,gold=[g for g in ex["gold"] if g["label"]==label]),
        task,[p for p in parsed if p[1]==label]) for label in schema}
    return all_counts,typed,valid


def evaluate(args):
    if args.checkpoint:
        checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        sites = training_sites(checkpoint)
        if checkpoint["target"] != "global" and checkpoint["target"] not in sites:
            raise ValueError("Own-local evaluation requires a recorded training site")
    else:
        public=torch.load(args.public_only,map_location="cpu",weights_only=True) if args.public_only else None
        checkpoint=dict(model=public["model"] if public else args.model,task=args.task,method=public["preparation"] if public else "Backbone only",
                        steps=0,rounds=0,target="global",block=public["block"] if public else None,
                        state=public["state"] if public else None,frozen_public=None)
    if checkpoint["target"] not in ("global",args.site):raise ValueError("Own-local evaluation is restricted to its source")
    model,task=checkpoint["model"],checkpoint["task"];schema=CFG["schemas"][task]
    configure_torch_reproducibility(getattr(args,"eval_seed",42))
    res=checkpoint["frozen_public"] is not None
    active=["public","client"] if res else ["client"]
    m,tok=load(model,args.snapshot,active);tok.padding_side="left"
    if checkpoint["state"] is not None:
        load_state(m,"client",checkpoint["state"],"meta" if args.public_only else "client")
    if res:load_state(m,"public",checkpoint["frozen_public"],"public")
    activate(m,active,[]);m.eval()
    rows=records(args.clinical,task,args.site,"test")
    out=output_directory(args.out);counts=Counter();typed={label:Counter() for label in schema};invalid=0
    for i in range(0,len(rows),CFG["eval_batch"]):
        examples=[example(row,task,args.site,"test",j) for j,row in enumerate(rows[i:i+CFG["eval_batch"]],i)]
        prompts=[chat_prompt(tok,model,task,schema,e) for e in examples]
        x=tok(prompts,padding=True,add_special_tokens=False,return_tensors="pt")
        if x.input_ids.shape[1] > CFG["max_input_tokens"]:
            raise ValueError("Evaluation prompt exceeds the configured token limit.")
        x=x.to(next(m.parameters()).device)
        with torch.inference_mode():
            y=m.generate(**x,do_sample=False,max_new_tokens=CFG["max_new_tokens"],use_cache=True,pad_token_id=tok.pad_token_id)
        texts=tok.batch_decode(y[:,x.input_ids.shape[1]:],skip_special_tokens=True)
        for e,raw in zip(examples,texts,strict=True):
            c,t,ok=score_one(e,task,schema,raw.strip());counts.update(c);invalid+=not ok
            for label in schema:typed[label].update(t[label])
    meta={k:checkpoint[k] for k in ("model","task","method","steps","rounds","target","block","seed","round_number","rule","training_sites") if k in checkpoint}
    (out/"counts.json").write_text(json.dumps(dict(**meta,site=args.site,rows=len(rows),invalid=invalid,
        split_sha256=split_digest(rows, task),counts=dict(counts),by_label=typed),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("snapshot","clinical","out"):p.add_argument("--"+k,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint");group.add_argument("--public-only");group.add_argument("--backbone-only",action="store_true")
    p.add_argument("--site",required=True)
    p.add_argument("--model",choices=CFG["models"],default="Qwen",help="backbone-only evaluation")
    p.add_argument("--task",choices=CFG["schemas"],default="NER",help="public-only or backbone-only evaluation")
    p.add_argument("--eval-seed",type=int,default=42)
    evaluate(p.parse_args())
