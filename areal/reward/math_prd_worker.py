# SPDX-License-Identifier: Apache-2.0

"""Subprocess worker for the standalone PRD-style math reward."""

from __future__ import annotations


def verify_semantic_boxed(
    ground_truth_boxed: str,
    answer_boxed: str,
    parse_timeout: int = 5,
    verify_timeout: int = 5,
) -> bool:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    from sympy import Basic, Sum, preorder_traversal
    from sympy.core.function import AppliedUndef

    extraction_config = (LatexExtractionConfig(), ExprExtractionConfig())
    gold_extractions = parse(
        ground_truth_boxed,
        extraction_config,
        parsing_timeout=parse_timeout,
        raise_on_error=True,
    )
    answer_extractions = parse(
        answer_boxed,
        extraction_config,
        parsing_timeout=parse_timeout,
        raise_on_error=True,
    )
    if not gold_extractions or not answer_extractions:
        return False
    if _constant_gold_vs_nonclosed_sum_pred(
        gold_extractions,
        answer_extractions,
        Basic,
        Sum,
        preorder_traversal,
        AppliedUndef,
    ):
        return False
    return bool(
        verify(
            gold_extractions,
            answer_extractions,
            timeout_seconds=verify_timeout,
            raise_on_error=True,
        )
    )


def _constant_gold_vs_nonclosed_sum_pred(
    gold_extractions,
    answer_extractions,
    sympy_basic_type,
    sympy_sum_type,
    preorder_traversal,
    applied_undef_type,
) -> bool:
    has_concrete_numeric_gold = any(
        _is_concrete_numeric_extraction(item, sympy_basic_type)
        for item in gold_extractions
    )
    if not has_concrete_numeric_gold:
        return False
    return any(
        _is_nonclosed_sum_extraction(
            item,
            sympy_basic_type,
            sympy_sum_type,
            preorder_traversal,
            applied_undef_type,
        )
        for item in answer_extractions
    )


def _is_concrete_numeric_extraction(value, sympy_basic_type) -> bool:
    return (
        isinstance(value, sympy_basic_type)
        and bool(value.is_number)
        and not value.free_symbols
    )


def _is_nonclosed_sum_extraction(
    value,
    sympy_basic_type,
    sympy_sum_type,
    preorder_traversal,
    applied_undef_type,
) -> bool:
    if not isinstance(value, sympy_basic_type):
        return False
    nodes = list(preorder_traversal(value))
    contains_sum = any(isinstance(node, sympy_sum_type) for node in nodes)
    if not contains_sum:
        return False
    contains_applied_undefined_function = any(
        isinstance(node, applied_undef_type) for node in nodes
    )
    return bool(value.free_symbols) or contains_applied_undefined_function
