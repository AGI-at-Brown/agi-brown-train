"""Shared document and conversation formats.
"""
import jax
from typing import NotRequired, TypedDict
from dataclasses import dataclass, field

class TrainBatch(TypedDict):
    observed_ids: jax.Array     # int32 [B, S]
    target_ids: jax.Array       # int32 [B, S]
    # to mask out some tokens that shouldn't contribute to the loss 
    # (probably won't need it unless we're doing a very specific format or something)
    loss_mask: NotRequired[jax.Array]        # float32 [B, S]

# store the training dataset
# Document would be something like 1 webpage for Common Crawl
@dataclass(frozen=True, kw_only=True)
class Document:
    id: str
    source: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class CommonCrawlDocument(Document):
    url: str
    timestamp: str
    language: str | None = None