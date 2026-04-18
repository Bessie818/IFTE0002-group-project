import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class AttentionArtifacts:
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    attention_scores: torch.Tensor
    attention_weights: torch.Tensor


class CreditCardSelfAttention(nn.Module):
    """
    Beginner-friendly single-head self-attention for tabular tokens.

    Input shape:
        x -> [batch_size, seq_len, d_model]

    Output shape:
        context -> [batch_size, seq_len, attention_dim]
    """

    def __init__(self, d_model: int, attention_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.attention_dim = attention_dim or d_model

        self.query_projection = nn.Linear(d_model, self.attention_dim)
        self.key_projection = nn.Linear(d_model, self.attention_dim)
        self.value_projection = nn.Linear(d_model, self.attention_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_artifacts: bool = False,
    ):
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.attention_dim)

        if attention_mask is not None:
            mask = self._expand_mask(attention_mask, scores)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, values)

        if return_artifacts:
            artifacts = AttentionArtifacts(
                queries=queries,
                keys=keys,
                values=values,
                attention_scores=scores,
                attention_weights=attention_weights,
            )
            return context, artifacts

        return context

    def _expand_mask(
        self, attention_mask: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Accepts:
            [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        Returns:
            boolean mask broadcastable to attention scores
        """
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        if attention_mask.dim() == 2:
            return attention_mask.unsqueeze(1).expand(-1, scores.size(1), -1)

        if attention_mask.dim() == 3:
            return attention_mask

        raise ValueError(
            "attention_mask must have shape [batch_size, seq_len] "
            "or [batch_size, seq_len, seq_len]."
        )


class CreditCardMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None, return_artifacts=False):
        batch_size, seq_len, _ = x.shape

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(context)

        if return_artifacts:
            avg_attn = attn_weights.mean(dim=1)
            return output, type("Artifacts", (), {"attention_weights": avg_attn})()

        return output
