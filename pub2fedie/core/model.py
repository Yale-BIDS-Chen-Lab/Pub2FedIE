"""Offline backbones, named adapters, feature masking and shared CLI conventions."""
import os
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import adapters as ada
from . import scoring as score
from .inputs import CFG, ROOT

SIMULATED_CONTINUATION = "Meta with simulated continuation"


def configure_torch_reproducibility(seed: int) -> None:
    """Seed the run and enable the study's strict PyTorch settings."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "strict deterministic CUDA runs require "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before Python starts"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load(name, path, slots):
    if not torch.cuda.is_available():raise RuntimeError("Full fitting or generation needs CUDA.")
    if not Path(path).is_dir():raise ValueError("Model snapshot directory not found.")
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
    tok.pad_token=tok.pad_token or tok.eos_token
    kw=dict(local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True)
    if name=="Llama":kw["attn_implementation"]="sdpa"
    try: model=AutoModelForCausalLM.from_pretrained(path,**kw)
    except ValueError:
        if name!="Qwen":raise
        from transformers import Qwen3_5ForConditionalGeneration
        model=Qwen3_5ForConditionalGeneration.from_pretrained(path,**kw)
    model=model.to("cuda");model.config.use_cache=False
    if name=="Qwen":
        from .qwen import install_qwen35_deterministic_prefix_sum
        install_qwen35_deterministic_prefix_sum(model)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    recipe=CFG["models"][name];targets=[]
    for key,layer in model.named_modules():
        m=re.fullmatch(r"model\.(?:language_model\.)?layers\.(\d+)\.self_attn\.(\w+)",key)
        if m and int(m[1]) in recipe["layers"] and m[2] in CFG["projections"] and isinstance(layer,torch.nn.Linear):
            targets.append(key)
    if len(targets)!=len(recipe["layers"])*4:raise ValueError("Unexpected backbone projection layout")
    conf=LoraConfig(r=CFG["rank"],lora_alpha=CFG["alpha"],lora_dropout=CFG["dropout"],
                    target_modules=sorted(targets),task_type=TaskType.CAUSAL_LM,bias="none")
    model=get_peft_model(model,conf,adapter_name=slots[0])
    for slot in slots[1:]:model.add_adapter(slot,conf)
    for slot in slots:
        p=ada.adapter_parameter_items(model,slot)
        if sum(t.numel() for _,t in p)!=recipe["parameters"]:raise ValueError("Adapter parameter count differs from the configured count.")
        for _,t in p:t.data=t.data.float()
    return model,tok


def chat_prompt(tok, name, task, schema, ex, clinical=True):
    prompt=(score.conditional_re_prompt if clinical and task=="RE" else score.prompt_for_example)(task,schema,ex)
    return tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,
                                   **({"enable_thinking":False} if name=="Qwen" else {}))


def feature(tok, name, task, schema, ex, clinical=True):
    chat=chat_prompt(tok,name,task,schema,ex,clinical=clinical)
    target=score.canonical_output(task,ex)
    inp=tok(chat,add_special_tokens=False)["input_ids"]
    tgt=tok(target,add_special_tokens=False)["input_ids"]+[tok.eos_token_id]
    if len(inp)+len(tgt)>CFG["max_input_tokens"]:raise ValueError("Prompt and target exceed the configured token limit.")
    return dict(input_ids=inp+tgt,labels=[-100]*len(inp)+tgt)


def loss(model, features, pad):
    device=next(model.parameters()).device
    width=max(len(f["input_ids"]) for f in features)
    def packed(k,pad_value):
        return torch.tensor([f[k]+[pad_value]*(width-len(f[k])) for f in features],device=device)
    mask=torch.tensor([[1]*len(f["input_ids"])+[0]*(width-len(f["input_ids"])) for f in features],device=device)
    return model(input_ids=packed("input_ids",pad),attention_mask=mask,labels=packed("labels",-100),use_cache=False).loss


def activate(model, slots, trainable):
    ada.set_active_adapters(model,slots)
    return ada.set_trainable_adapters(model,set(trainable))


def load_state(model, slot, state, original):
    ada.restore_adapter(model,slot,{k.replace("."+original+".","."+slot+"."):v for k,v in state.items()})


def output_directory(path):
    path=Path(path).resolve()
    if path==ROOT or ROOT in path.parents:raise ValueError("Choose an output directory outside the repository.")
    path.mkdir(parents=True,exist_ok=False)
    return path
