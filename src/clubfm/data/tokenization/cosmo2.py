"""Load the Cosmo2 tokenizer from Hugging Face Hub."""

from tokenizers import Tokenizer

COSMO2_TOKENIZER_ID = "HuggingFaceTB/cosmo2-tokenizer"


def load_cosmo2_tokenizer(*, revision: str = "main") -> Tokenizer:
  return Tokenizer.from_pretrained(COSMO2_TOKENIZER_ID, revision=revision)
