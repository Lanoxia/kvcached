# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for supported SGLang releases."""

from __future__ import annotations

from typing import Any, List, Optional


def _as_int_list(values: Any) -> List[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def legacy_get_num_new_pages(
    seq_lens: Any,
    page_size: int,
    prefix_lens: Optional[Any] = None,
    decode: bool = False,
) -> int:
    """Match SGLang's helper on releases where it is not publicly exported."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    sequence_lengths = _as_int_list(seq_lens)
    if prefix_lens is None or decode:
        if not decode:
            raise ValueError("prefix_lens is required outside decode")
        return sum(length % page_size == 1 for length in sequence_lengths)

    prefix_lengths = _as_int_list(prefix_lens)
    if len(sequence_lengths) != len(prefix_lengths):
        raise ValueError("seq_lens and prefix_lens must have the same length")
    if any(
        prefix_length > sequence_length
        for sequence_length, prefix_length in zip(sequence_lengths, prefix_lengths)
    ):
        raise ValueError("prefix length cannot exceed sequence length")

    return sum(
        (sequence_length + page_size - 1) // page_size
        - (prefix_length + page_size - 1) // page_size
        for sequence_length, prefix_length in zip(sequence_lengths, prefix_lengths)
    )
