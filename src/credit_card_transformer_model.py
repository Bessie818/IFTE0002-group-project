from typing import Optional

import torch
from torch import nn

from transformer_attention import CreditCardSelfAttention


class CreditCardTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_dim: Optional[int] = None,
        ffn_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        attention_dim = attention_dim or d_model

        self.attention = CreditCardSelfAttention(
            d_model=d_model,
            attention_dim=attention_dim,
        )
        self.attention_output = nn.Linear(attention_dim, d_model)
        self.norm_after_attention = nn.LayerNorm(d_model)
        self.norm_after_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        attention_result = self.attention(
            x,
            attention_mask=attention_mask,
            return_artifacts=return_attention,
        )
        if return_attention:
            attended_x, artifacts = attention_result
        else:
            attended_x = attention_result
            artifacts = None

        attended_x = self.attention_output(attended_x)
        x = self.norm_after_attention(x + self.dropout(attended_x))

        ffn_output = self.feed_forward(x)
        x = self.norm_after_ffn(x + self.dropout(ffn_output))

        if return_attention:
            return x, artifacts
        return x


class CreditCardTransformerClassifier(nn.Module):
    """
    Minimal end-to-end model:
        token ids -> embedding -> stacked transformer blocks -> [CLS] -> classifier
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        pad_token_id: int,
        d_model: int = 32,
        attention_dim: Optional[int] = None,
        ffn_hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 6,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(sequence_length, d_model)

        self.blocks = nn.ModuleList([
            CreditCardTransformerBlock(
                d_model=d_model,
                attention_dim=attention_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        all_artifacts = []

        for block in self.blocks:
            block_result = block(
                x,
                attention_mask=attention_mask,
                return_attention=return_attention,
            )
            if return_attention:
                x, artifacts = block_result
                all_artifacts.append(artifacts)
            else:
                x = block_result

        cls_representation = x[:, 0, :]
        logits = self.classifier(cls_representation).squeeze(-1)

        if return_attention:
            return logits, all_artifacts
        return logits
