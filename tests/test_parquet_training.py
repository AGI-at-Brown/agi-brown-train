import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from clubfm.models.qwen_flash_next import Config
from clubfm.training.trainer import TrainConfig, get_dataset


@pytest.fixture
def train_cfg(tmp_path):
  vocab = {"<eos>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
  tokenizer = Tokenizer(WordLevel(vocab))
  tokenizer.pre_tokenizer = WhitespaceSplit()
  return TrainConfig(
    epochs=1,
    lr=0.001,
    vocab=None,
    num_train_steps=3,
    dataset_path=str(tmp_path / "train.parquet"),
    batch_size=2,
    weight_cfg=Config(
      B=2, L=4, D=16, F=32, N=2, K=1, H=8, V=6, T=3, rope_theta=10000.0, eos_token_id=0
    ),
    tokenizer=tokenizer,
  )


def test_parquet_training_packs_documents_across_row_groups(train_cfg):
  pq.write_table(
    pa.table({"text": ["a b", None, "  ", "c d e a", "b"]}),
    train_cfg.dataset_path,
    row_group_size=2,
  )

  [batch] = list(get_dataset(train_cfg))

  np.testing.assert_array_equal(batch["observed_ids"], [[1, 2, 0], [3, 4, 5]])
  np.testing.assert_array_equal(batch["target_ids"], [[2, 0, 3], [4, 5, 1]])


def test_parquet_training_respects_step_limit(train_cfg):
  pq.write_table(pa.table({"text": ["a b c d"] * 10}), train_cfg.dataset_path)

  assert len(list(get_dataset(train_cfg.replace(num_train_steps=1)))) == 1


def test_parquet_training_requires_text_column(train_cfg):
  pq.write_table(pa.table({"title": ["a b"]}), train_cfg.dataset_path)

  with pytest.raises(ValueError, match="'text' column"):
    next(get_dataset(train_cfg))


def test_fixed_length_text_training_preserves_row_content(train_cfg, tmp_path):
  path = tmp_path / "train.txt"
  path.write_text("a b c d\n", encoding="utf-8")

  [batch] = list(get_dataset(train_cfg.replace(dataset_path=str(path), num_train_steps=1)))

  np.testing.assert_array_equal(batch["observed_ids"], [[1, 2, 3], [1, 2, 3]])
  np.testing.assert_array_equal(batch["target_ids"], [[2, 3, 4], [2, 3, 4]])
