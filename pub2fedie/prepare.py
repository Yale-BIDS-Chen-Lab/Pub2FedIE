"""Prepare a public SFT or first-order aggregation-aware Meta adapter."""
import argparse
import random
from contextlib import contextmanager, nullcontext
import torch

from .core import adapters as ada, corpora, sampling
from .core.model import (
    CFG, SIMULATED_CONTINUATION, activate, configure_torch_reproducibility, feature, load, loss, output_directory,
)


def pools(tok,name,seed):
    p=CFG["public"]; out={}
    for view in corpora.NER_DOMAINS+corpora.RE_DOMAINS:
        halves=sampling.partition_documents(corpora.load_public_domain(view,"train"),p["support_units"],
                                            p["query_units"],seed,f"public-documents:{view}")
        out[view]={}
        for role,docs in zip(("support","query"),halves):
            rows=[]
            for d in docs:
                for ex in d["examples"]:
                    f=feature(tok,name,d["task_family"],d["schema"],ex,clinical=False)
                    rows.append(dict(f,document_id=d["document_id"],task_family=d["task_family"],target_has_items=bool(ex["gold"])))
            out[view][role]=sampling.group_features_by_document(rows)
    return out


def fit(args):
    updates=args.updates if getattr(args,"updates",None) is not None else CFG["public"]["updates"]
    if args.block<0 or updates<=0:raise ValueError("Use a nonnegative seed and a positive update count")
    corpora.set_root(args.corpora)
    out=output_directory(args.out)
    configure_torch_reproducibility(args.block)
    continued=args.preparation==SIMULATED_CONTINUATION
    m,tok=load(args.model,args.snapshot,["meta"] if continued else ["meta","residual","agg"])
    data=pools(tok,args.model,args.block); p=CFG["public"]
    zero=ada.snapshot_adapter(m,"residual") if args.preparation=="Meta" else None
    trainable=activate(m,["meta"],["meta"])
    outer=torch.optim.AdamW(trainable,lr=p["lr"],weight_decay=CFG["weight_decay"])
    rng=random.Random(sampling.stable_seed(args.block,"public-view-order"))
    for t in range(updates):
        views=rng.sample(sorted((corpora.NER_DOMAINS,corpora.RE_DOMAINS)[t%2]),p["views_per_update"])
        def sample(view,role,key):
            vals,_=sampling.select(data[view][role],args.block,(view,str(t),key),sampling.choose_document_then_target_class)
            return loss(m,[vals],tok.pad_token_id)
        outer.zero_grad(set_to_none=True);m.train()
        if args.preparation=="SFT":
            activate(m,["meta"],["meta"])
            torch.stack([sample(v,"support","support") for v in views]).mean().backward()
        else:
            slot="meta" if continued else "residual"
            start=ada.snapshot_adapter(m,"meta") if continued else zero
            clients=[]
            for v in views:
                ada.restore_adapter(m,slot,start)
                params=activate(m,["meta"] if continued else ["meta","residual"],[slot])
                inner=torch.optim.AdamW(params,lr=p["inner_lr"],weight_decay=CFG["weight_decay"])
                for j in range(sampling.family_balanced_inner_steps(t)):
                    inner.zero_grad(set_to_none=True); sample(v,"support",str(j)).backward()
                    torch.nn.utils.clip_grad_norm_(params,CFG["clip"],error_if_nonfinite=True);inner.step()
                clients.append(ada.snapshot_adapter(m,slot) if continued else ada.collect_adapter_low_rank(m,slot))
            if continued:
                ada.restore_adapter(m,"meta",start)
                outer.zero_grad(set_to_none=True)  # Discard the last client's support gradient.
                query_path=effective_forward(m,continued_matrices(dict(ada.adapter_parameter_items(m,"meta")),
                    clients,ada.adapter_scalings(m,"meta"),CFG["rank"]))
            else:
                factors=ada.aggregate_adapter_updates(clients,CFG["rank"],ada.adapter_scalings(m,"agg"))
                ada.load_adapter_factors(m,"agg",factors)
                activate(m,["meta","agg"],["meta"])
                query_path=nullcontext()
            with query_path:
                m.eval()
                with torch.no_grad():worst=max(views,key=lambda v:float(sample(v,"query","query")))
                m.train()
                for i,v in enumerate(views):
                    weight=1/len(views)+p["worst_weight"]*(v==worst)
                    (sample(v,"query","query")*weight).backward(retain_graph=continued and i<len(views)-1)
            activate(m,["meta"],["meta"])
            for v in views:(sample(v,"query","query")*p["initial_weight"]/len(views)).backward()
        torch.nn.utils.clip_grad_norm_(trainable,CFG["clip"],error_if_nonfinite=True);outer.step()
    torch.save(dict(model=args.model,preparation=args.preparation,block=args.block,
                    state=ada.snapshot_adapter(m,"meta")),out/"public.pt")


# Differentiable rank projection for Meta with simulated continuation.

class RankProjection(torch.autograd.Function):
    """Truncated SVD of an effective matrix, differentiated as a matrix, not as LoRA factors."""

    @staticmethod
    def forward(ctx, matrix, u, singular, v, rank):
        if not 0 < rank <= min(matrix.shape) or singular.numel() < rank:
            raise ValueError("Invalid projection rank")
        if not torch.isfinite(singular).all():
            raise RuntimeError("Nonfinite projection spectrum")
        if rank < min(matrix.shape):
            kept = singular[rank - 1].square()
            tail = singular[rank].square() if rank < singular.numel() else kept * 0
            if not (kept > 0 and (kept - tail) / kept > 1e-10):
                raise RuntimeError("Rank cutoff is not differentiably separated")
        ctx.save_for_backward(u, singular, v)
        ctx.rank, ctx.shape, ctx.input_dtype = rank, matrix.shape, matrix.dtype
        return ((u[:, :rank] * singular[:rank]) @ v[:, :rank].T).to(matrix.dtype)

    @staticmethod
    def backward(ctx, grad):
        u, singular, v = ctx.saved_tensors
        r = ctx.rank
        if r == min(ctx.shape):
            return grad, None, None, None, None
        g = grad.to(u.dtype)
        ur, vr, us, vs = u[:, :r], v[:, :r], u[:, r:], v[:, r:]
        urg, gvr = ur.T @ g, g @ vr
        output = ur @ urg + gvr @ vr.T - ur @ (urg @ vr) @ vr.T
        if us.shape[1]:
            a, b = singular[:r, None], singular[None, r:]
            denominator = a.square() - b.square()
            d, c = b.square() / denominator, a * b / denominator
            grs, gsr = urg @ vs, us.T @ gvr
            output = output + ur @ (d * grs + c * gsr.T) @ vs.T
            output = output + us @ (d.T * gsr + c.T * grs.T) @ vr.T
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite rank-projection gradient")
        return output.to(ctx.input_dtype), None, None, None, None


def continued_matrices(parameters, local_states, scalings, rank):
    """Average continued effective matrices; hold only the inner optimizer displacement fixed."""
    def factor(key, state):
        initial = parameters[key]
        return initial + (state[key].to(initial) - initial).detach()

    matrices = {}
    for name, scale in sorted(scalings.items()):
        ka, kb = name + ".lora_A.meta.weight", name + ".lora_B.meta.weight"
        factors = [(factor(kb, state), factor(ka, state)) for state in local_states]
        weight = scale / len(factors)
        mean = sum((b @ a) * weight for b, a in factors)
        # Obtain the spectrum from the small QR cores, including discarded directions.
        with torch.no_grad():
            left = torch.cat([b.double() * weight**0.5 for b, _ in factors], dim=1)
            right = torch.cat([a.double() * weight**0.5 for _, a in factors], dim=0)
            ql, rl = torch.linalg.qr(left, mode="reduced")
            qr, rr = torch.linalg.qr(right.T, mode="reduced")
            u, s, vh = torch.linalg.svd(rl @ rr.T, full_matrices=False)
            u, v = ql @ u, qr @ vh.T
        matrices[name] = RankProjection.apply(mean, u, s, v, rank)
    return matrices


@contextmanager
def effective_forward(model, matrices):
    """Apply each projected update once and keep it connected through backward/recomputation."""
    originals = []
    try:
        for name, delta in matrices.items():
            layer = model.get_submodule(name)
            originals.append((layer, layer.forward))

            def forward(x, *args, _layer=layer, _delta=delta, **kwargs):
                base = _layer.base_layer(x, *args, **kwargs)
                inputs = _layer.lora_dropout["meta"](x.to(_delta.dtype))
                return (base + torch.nn.functional.linear(inputs, _delta)).to(base.dtype)

            layer.forward = forward
        yield
    finally:
        for layer, original in originals:
            layer.forward = original


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg,choices in [("model",CFG["models"]),("preparation",["SFT","Meta",SIMULATED_CONTINUATION])]:
        parser.add_argument("--"+arg,required=True,choices=choices)
    parser.add_argument("--block",type=int,required=True)
    parser.add_argument("--updates",type=int,help="number of public optimizer updates")
    for flag in ("snapshot","corpora","out"):parser.add_argument("--"+flag,required=True)
    fit(parser.parse_args())
