"""Vectorized PyTorch oracle for the one-layer tiny causal MoE model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .spec import FixtureSpec, TensorSpec


@dataclass(frozen=True)
class RouteTrace:
    router_scores: Tensor
    selected_experts: Tensor
    selected_weights: Tensor


@dataclass(frozen=True)
class OracleOutput:
    logits: Tensor
    routes: RouteTrace


def stable_top_k(scores: Tensor, k: int) -> Tensor:
    """Return IDs ordered by descending score, then ascending expert ID."""

    if scores.ndim != 2:
        raise ValueError("stable_top_k expects a rank-2 score tensor")
    if not 0 < k <= scores.shape[1]:
        raise ValueError("k must be between one and the number of experts")
    if not bool(torch.all(torch.isfinite(scores))):
        raise ValueError("routing scores must be finite")
    rows = [
        sorted(range(scores.shape[1]), key=lambda expert: (-float(row[expert]), expert))[:k]
        for row in scores.detach().cpu()
    ]
    return torch.tensor(rows, dtype=torch.int64, device=scores.device)


class TinyMoEOracle:
    """A no-grad reference assembled entirely from formula-generated tensors."""

    def __init__(self, spec: FixtureSpec) -> None:
        self.spec = spec
        self.hidden_size = spec.model["hidden_size"]
        self.num_heads = spec.model["num_heads"]
        self.head_size = self.hidden_size // self.num_heads
        self.num_experts = spec.model["num_experts"]
        self.top_k = spec.model["top_k"]
        self.context_length = spec.model["context_length"]
        self.rms_epsilon = math.ldexp(1.0, spec.numeric["rms_norm_epsilon_power_of_two"])
        self.attention_scale = math.ldexp(1.0, spec.numeric["attention_scale_power_of_two"])
        self.weights = {tensor.role: self._make_tensor(tensor) for tensor in spec.tensors}

    def _make_tensor(self, tensor: TensorSpec) -> Tensor:
        count = math.prod(tensor.shape)
        flat_index = torch.arange(count, dtype=torch.int64)
        recipe = self.spec.recipe
        numerator = (
            recipe["multiplier_tensor"] * (tensor.tensor_id + 1)
            + recipe["multiplier_index"] * (flat_index + 1)
        ) % recipe["modulus"] - recipe["center"]
        values = numerator.to(torch.float32)
        if tensor.tensor_id in recipe["normalization_tensor_ids"]:
            values = recipe["normalization_base"] + torch.ldexp(
                values,
                torch.tensor(-recipe["normalization_denominator_power_of_two"]),
            )
        else:
            values = torch.ldexp(
                values,
                torch.tensor(-recipe["ordinary_denominator_power_of_two"]),
            )
        return values.reshape(tensor.shape).contiguous()

    def _rms_norm(self, values: Tensor, weight_role: str) -> Tensor:
        mean_square = values.square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(mean_square + self.rms_epsilon) * self.weights[weight_role]

    def _project(self, values: Tensor, weight_role: str) -> Tensor:
        return values @ self.weights[weight_role].transpose(0, 1)

    @torch.no_grad()
    def __call__(self, token_ids: list[int] | Tensor) -> OracleOutput:
        ids = torch.as_tensor(token_ids, dtype=torch.int64)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("token IDs must be a non-empty rank-1 sequence")
        if ids.numel() > self.context_length:
            raise ValueError("token sequence exceeds the context length")
        vocab_size = self.spec.model["vocab_size"]
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= vocab_size)):
            raise ValueError("token ID is outside the vocabulary")

        residual = self.weights["token_embedding"].index_select(0, ids)
        normalized = self._rms_norm(residual, "layers.0.attn_norm")

        sequence_length = ids.numel()
        projections = []
        for role in ("layers.0.attn_q", "layers.0.attn_k", "layers.0.attn_v"):
            projected = self._project(normalized, role)
            projections.append(
                projected.reshape(sequence_length, self.num_heads, self.head_size).transpose(0, 1)
            )
        query, key, value = projections
        attention_scores = (query @ key.transpose(1, 2)) * self.attention_scale
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
        )
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))
        attention_probabilities = torch.softmax(attention_scores, dim=-1)
        context = attention_probabilities @ value
        context = context.transpose(0, 1).contiguous().reshape(sequence_length, self.hidden_size)
        residual = residual + self._project(context, "layers.0.attn_out")

        normalized = self._rms_norm(residual, "layers.0.ffn_norm")
        router_scores = self._project(normalized, "layers.0.router")
        selected_experts = stable_top_k(router_scores, self.top_k)
        selected_scores = router_scores.gather(1, selected_experts)
        selected_weights = torch.softmax(selected_scores, dim=-1)

        expert_outputs = []
        for expert_id in range(self.num_experts):
            prefix = f"layers.0.experts.{expert_id}"
            gate = F.silu(self._project(normalized, f"{prefix}.gate"))
            up = self._project(normalized, f"{prefix}.up")
            expert_outputs.append(self._project(gate * up, f"{prefix}.down"))
        all_expert_outputs = torch.stack(expert_outputs, dim=1)
        positions = torch.arange(sequence_length).unsqueeze(1)
        chosen_outputs = all_expert_outputs[positions, selected_experts]
        mixture = (chosen_outputs * selected_weights.unsqueeze(-1)).sum(dim=1)
        residual = residual + mixture

        final = self._rms_norm(residual, "final_norm")
        logits = self._project(final, "lm_head")
        return OracleOutput(
            logits=logits,
            routes=RouteTrace(
                router_scores=router_scores,
                selected_experts=selected_experts,
                selected_weights=selected_weights,
            ),
        )
