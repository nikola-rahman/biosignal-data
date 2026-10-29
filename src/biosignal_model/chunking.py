"""Where the global tokens are (docs/architecture.md section 4): the chunker interface.

The v4 model: local stage -> `boundaries(local features, visible only)` -> global model ->
`upsample` + residual (H-Net type). A chunker places the boundaries of the global tokens from
the local features of the visible samples; the global model works on the tokens; `upsample`
gives every sample the feature of its token, and the local features are added back as a
residual.

`FixedChunker` is the first implementation: one grid of `token_s` seconds for every channel
and rate (the packer's token table, biosignal_model.channel_batch). A learned chunker (H-Net
type) is the end state; it implements the same `boundaries` and returns its own token table
(and the token of every sample), so nothing after it changes.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from biosignal_model.channel_batch import ChannelInputs, TokenTable


class Chunker(Protocol):
    def boundaries(
        self, features: torch.Tensor, visible: torch.Tensor, x: ChannelInputs
    ) -> TokenTable:
        """The global tokens of the batch: `features` [S, width] from the local stage,
        `visible` [S] the samples the chunker may read (observed and not hidden); a learned
        chunker must not place a boundary from a hidden sample."""
        ...


class FixedChunker(nn.Module):
    """Tokens of `token_s` seconds on one grid from each example's start (no weights)."""

    def __init__(self, token_s: float) -> None:
        super().__init__()
        self.token_s = token_s

    def boundaries(
        self, features: torch.Tensor, visible: torch.Tensor, x: ChannelInputs
    ) -> TokenTable:
        t = x.tokens
        if t.t0.numel() and not torch.allclose(
            t.t1 - t.t0, torch.full_like(t.t0, self.token_s), rtol=0, atol=1e-9
        ):
            raise ValueError("the batch's tokens are not of this chunker's token_s")
        return t


def upsample(token_features: torch.Tensor, tokens: TokenTable) -> torch.Tensor:
    """[S, width]: each sample gets the feature of its token ([T, width])."""
    return token_features[tokens.sample_token]
