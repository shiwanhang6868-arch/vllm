# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise numerical acceptance rules without fake vLLM model/output objects."""

import pytest

from .pp_fixtures import assert_pp_logprobs_close


def test_near_tie_compares_shared_history_before_accepting_greedy_divergence():
    assert_pp_logprobs_close(
        [{0: -0.7, 1: -0.7005}, {0: -0.2, 1: -1.7}],
        [{0: -0.7005, 1: -0.7}, {0: -4.0, 1: -0.01}],
        reference_tokens=[0, 0],
        actual_tokens=[1, 1],
    )


def test_matching_greedy_tokens_do_not_hide_numerical_errors():
    with pytest.raises(AssertionError):
        assert_pp_logprobs_close(
            [{0: -0.7, 1: -0.8}],
            [{0: -0.1, 1: -2.0}],
            reference_tokens=[0],
            actual_tokens=[0],
        )


def test_fixed_prompt_history_checks_all_positions():
    with pytest.raises(AssertionError):
        assert_pp_logprobs_close(
            [None, {0: -0.7, 1: -0.8}, {0: -0.2, 1: -1.7}],
            [None, {0: -0.7, 1: -0.8}, {0: -3.0, 1: -0.1}],
        )
