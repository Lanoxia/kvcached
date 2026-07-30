# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from importlib.machinery import ModuleSpec

import pytest

_torch_mock = types.ModuleType("torch")
_torch_mock.__spec__ = ModuleSpec("torch", loader=None)
sys.modules.setdefault("torch", _torch_mock)

from kvcached.integration.sglang.compat import legacy_get_num_new_pages


class TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_extend_counts_pages_added_after_prefix():
    assert legacy_get_num_new_pages(
        seq_lens=TensorLike([17, 32, 49]),
        page_size=16,
        prefix_lens=TensorLike([16, 17, 32]),
    ) == 3


def test_decode_counts_sequences_that_cross_page_boundaries():
    assert legacy_get_num_new_pages(
        seq_lens=TensorLike([1, 16, 17, 32, 33]),
        page_size=16,
        decode=True,
    ) == 3


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="page_size"):
        legacy_get_num_new_pages([1], 0, [0])

    with pytest.raises(ValueError, match="same length"):
        legacy_get_num_new_pages([1, 2], 16, [0])

    with pytest.raises(ValueError, match="prefix length"):
        legacy_get_num_new_pages([15], 16, [16])

    with pytest.raises(ValueError, match="prefix_lens is required"):
        legacy_get_num_new_pages([1], 16)


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(1, 1), (16, 0), (17, 1), (32, 0), (33, 1)],
)
def test_decode_boundary_behavior(seq_len, expected):
    assert legacy_get_num_new_pages([seq_len], 16, decode=True) == expected
