from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def adapter_parameter_items(model, adapter_name: str):
    marker = f".{adapter_name}."
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name and ("lora_A" in name or "lora_B" in name)
    ]


def snapshot_adapter(model, adapter_name: str) -> dict[str, torch.Tensor]:
    items = adapter_parameter_items(model, adapter_name)
    if not items:
        raise RuntimeError(f"no LoRA parameters found for adapter {adapter_name}")
    return {
        name: parameter.detach().cpu().clone() for name, parameter in items
    }


def restore_adapter(
    model, adapter_name: str, state: dict[str, torch.Tensor]
) -> None:
    current = dict(adapter_parameter_items(model, adapter_name))
    if set(current) != set(state):
        raise RuntimeError(
            f"adapter state mismatch for {adapter_name}: "
            f"current_only={sorted(set(current) - set(state))[:5]} "
            f"state_only={sorted(set(state) - set(current))[:5]}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))


def set_trainable_adapters(model, trainable: set[str]) -> list[torch.nn.Parameter]:
    selected: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        enabled = is_lora and any(f".{adapter}." in name for adapter in trainable)
        parameter.requires_grad = enabled
        if enabled:
            selected.append(parameter)
    if trainable and not selected:
        raise RuntimeError(f"no parameters enabled for adapters {sorted(trainable)}")
    return selected


def set_active_adapters(model, adapter_names: list[str]) -> None:
    if not adapter_names:
        raise ValueError("at least one adapter must be active")
    model.base_model.set_adapter(adapter_names, inference_mode=False)


def collect_adapter_low_rank(
    model, adapter_name: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    updates: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None or scaling is None:
            continue
        if adapter_name not in lora_a or adapter_name not in lora_b:
            continue
        updates[module_name] = (
            lora_b[adapter_name].weight.detach().float().cpu().clone(),
            lora_a[adapter_name].weight.detach().float().cpu().clone(),
            float(scaling[adapter_name]),
        )
    if not updates:
        raise RuntimeError(f"no effective LoRA updates found for {adapter_name}")
    return updates


def adapter_scalings(model, adapter_name: str) -> dict[str, float]:
    result = {}
    for module_name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            result[module_name] = float(scaling[adapter_name])
    if not result:
        raise RuntimeError(f"no adapter scaling found for {adapter_name}")
    return result


def truncated_product_svd(
    left: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(
            f"incompatible low-rank product: left={tuple(left.shape)} "
            f"right={tuple(right.shape)}"
        )
    if rank <= 0:
        raise ValueError("rank must be positive")
    q_left, r_left = torch.linalg.qr(left.float(), mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T.float(), mode="reduced")
    core = r_left @ r_right.T
    u_core, singular_values, vh_core = torch.linalg.svd(
        core, full_matrices=False
    )
    retained_rank = min(rank, singular_values.numel())
    u = q_left @ u_core[:, :retained_rank]
    vh = vh_core[:retained_rank] @ q_right.T
    kept = singular_values[:retained_rank]
    return u, kept, vh


def factorize_mean_effective_update(
    components: Iterable[tuple[torch.Tensor, torch.Tensor, float]],
    rank: int,
    target_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    components = list(components)
    if not components:
        raise ValueError("at least one component is required")
    if target_scaling <= 0:
        raise ValueError("target_scaling must be positive")
    count = len(components)
    left_blocks = []
    right_blocks = []
    for b_matrix, a_matrix, scaling in components:
        if scaling <= 0:
            raise ValueError("component scaling must be positive")
        weight = math.sqrt(scaling / count)
        left_blocks.append(b_matrix.float() * weight)
        right_blocks.append(a_matrix.float() * weight)
    left = torch.cat(left_blocks, dim=1)
    right = torch.cat(right_blocks, dim=0)
    u, singular_values, vh = truncated_product_svd(
        left, right, rank
    )
    retained_rank = singular_values.numel()
    root = torch.sqrt(singular_values.clamp_min(0) / target_scaling)
    b_factor = u * root.unsqueeze(0)
    a_factor = root.unsqueeze(1) * vh
    if retained_rank < rank:
        b_factor = torch.nn.functional.pad(
            b_factor, (0, rank - retained_rank)
        )
        a_factor = torch.nn.functional.pad(
            a_factor, (0, 0, 0, rank - retained_rank)
        )
    return b_factor, a_factor


def aggregate_adapter_updates(
    domain_updates: list[dict[str, tuple[torch.Tensor, torch.Tensor, float]]],
    rank: int,
    target_scalings: dict[str, float],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not domain_updates:
        raise ValueError("domain_updates cannot be empty")
    module_names = set(domain_updates[0])
    for index, update in enumerate(domain_updates[1:], start=1):
        if set(update) != module_names:
            raise RuntimeError(f"domain update module mismatch at index {index}")
    if set(target_scalings) != module_names:
        raise RuntimeError("target scaling modules do not match domain updates")
    factors = {}
    for module_name in sorted(module_names):
        factors[module_name] = factorize_mean_effective_update(
            [update[module_name] for update in domain_updates],
            rank,
            target_scalings[module_name],
        )
    return factors


def load_adapter_factors(
    model,
    adapter_name: str,
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    seen = set()
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if module_name not in factors:
                continue
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                raise RuntimeError(f"target module is not LoRA-enabled: {module_name}")
            b_factor, a_factor = factors[module_name]
            lora_a[adapter_name].weight.copy_(
                a_factor.to(
                    lora_a[adapter_name].weight.device,
                    dtype=lora_a[adapter_name].weight.dtype,
                )
            )
            lora_b[adapter_name].weight.copy_(
                b_factor.to(
                    lora_b[adapter_name].weight.device,
                    dtype=lora_b[adapter_name].weight.dtype,
                )
            )
            seen.add(module_name)
    if seen != set(factors):
        raise RuntimeError(f"failed to load factors for {sorted(set(factors) - seen)}")


def combine_client_states(states, weights, representation, *, rank, scaling):
    """Combine saved LoRA states by averaging factors or effective updates."""
    if not states or len(states) != len(weights) or any(s.keys() != states[0].keys() for s in states):
        raise ValueError("Expected one weight per client and matching LoRA factors")
    w = torch.as_tensor(weights, dtype=torch.float64)
    if not torch.isfinite(w).all() or not (w > 0).all() or abs(float(w.sum()) - 1) > 1e-10:
        raise ValueError("Weights must be positive and sum to one")
    if representation == "factor":
        return {k: sum(p * s[k].double() for p, s in zip(w, states)).float() for k in states[0]}
    if representation != "effective":
        raise ValueError("Unknown aggregation rule")
    result = {}
    for a in (k for k in states[0] if ".lora_A." in k):
        b = a.replace(".lora_A.", ".lora_B.")
        result[b], result[a] = factorize_mean_effective_update(
            [(s[b], s[a], scaling * len(states) * float(p)) for s, p in zip(states, w)],
            rank, scaling,
        )
    if result.keys() != states[0].keys():
        raise ValueError("Unpaired factors")
    return result




