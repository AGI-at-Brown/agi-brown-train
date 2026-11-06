"""Shared document and conversation formats.
"""
import jax
from typing import NotRequired, TypedDict

class TrainBatch(TypedDict):
    observed_ids: jax.Array     # int32 [B, S]
    target_ids: jax.Array       # int32 [B, S]
    # to mask out some tokens that shouldn't contribute to the loss 
    # (probably won't need it unless we're doing a very specific format or something)
    loss_mask: NotRequired[jax.Array]        # float32 [B, S]