"""Unit tests for the extraction-evaluation example script (Example 16).

The script is a numbered example file, not an importable package module, so it
is loaded by path. Covers the two matcher relaxations added for the 2026-07-08
benchmark review: P7 (squashed-form containment for OCR word-glue) and P9
(a long shared verbatim span scores as strong identity in structural alignment).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "examples"
    / "scripts"
    / "16_extraction_evaluation.py"
)


@pytest.fixture(scope="module")
def evalmod():
    spec = importlib.util.spec_from_file_location("extraction_evaluation", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_relaxed_match_pairs_ocr_glued_name(evalmod):
    """'Garden work' (GT) vs 'Gardenwork' (extracted) pair via squashed form,
    which canonical underscore-joined containment misses."""
    gt = [("Item", ("GARDEN_WORK",))]
    got = [("Item", ("GARDENWORK",))]
    pairs = evalmod.relaxed_match(gt, got)
    assert pairs == [(("Item", ("GARDEN_WORK",)), ("Item", ("GARDENWORK",)))]


def test_relaxed_match_respects_digit_signature_guard(evalmod):
    """Squashed containment must not merge ids that differ in a digit run."""
    gt = [("X", ("ITEM_1",))]
    got = [("X", ("ITEM_2",))]
    assert evalmod.relaxed_match(gt, got) == []


def test_relaxed_match_requires_unique_candidate(evalmod):
    """Two equally-matching produced ids leave the GT id unpaired (no guessing)."""
    gt = [("Party", ("ROBERT_SCHNEIDER_AG",))]
    got = [("Party", ("ROBERTSCHNEIDERAG",)), ("Party", ("ROBERT_SCHNEIDER_AG",))]
    # The exact-canonical one is matched in the strict pass upstream; here both
    # are candidates for the relaxed pass, so relaxed must abstain.
    assert evalmod.relaxed_match(gt, got) == []


def test_pair_similarity_long_shared_span_scores_strong(evalmod):
    """A >=40-char shared verbatim span alone reaches the 1.0 alignment threshold
    (lets sparse-id exclusion clauses pair)."""
    texte = "les dommages resultant d'un defaut d'entretien notoire du batiment"
    gt_node = {"__class__": "Exclusion", "texte": texte}
    got_node = {"__class__": "Exclusion", "texte": texte + " assure"}
    assert evalmod._pair_similarity(gt_node, got_node) >= 1.0


def test_pair_similarity_short_containment_stays_weak(evalmod):
    """A short shared category word stays at the weak 0.5 (below threshold)."""
    gt_node = {"__class__": "Exclusion", "resume": "usure"}
    got_node = {"__class__": "Exclusion", "resume": "usure vitre"}
    assert evalmod._pair_similarity(gt_node, got_node) == 0.5
