"""Unit tests for src/utils.py core metric functions."""

import math
import sys
from pathlib import Path

import pytest

# Allow importing from src/ without package install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import compute_fsi, compute_sequence_identity, compute_site_recovery, truncate_sequence


# ---------------------------------------------------------------------------
# truncate_sequence
# ---------------------------------------------------------------------------


class TestTruncateSequence:
    def test_short_sequence_unchanged(self):
        seq = "ACDEFG"
        assert truncate_sequence(seq) == seq

    def test_exact_max_length_unchanged(self):
        seq = "A" * 1022
        assert truncate_sequence(seq) == seq

    def test_longer_sequence_truncated(self):
        seq = "A" * 2000
        result = truncate_sequence(seq)
        assert len(result) == 1022

    def test_custom_max_length(self):
        seq = "ACDEFGHIJK"
        assert truncate_sequence(seq, max_length=5) == "ACDEF"

    def test_empty_sequence(self):
        assert truncate_sequence("") == ""


# ---------------------------------------------------------------------------
# compute_sequence_identity
# ---------------------------------------------------------------------------


class TestComputeSequenceIdentity:
    def test_identical_sequences(self):
        assert compute_sequence_identity("ACDEF", "ACDEF") == pytest.approx(1.0)

    def test_no_matches(self):
        assert compute_sequence_identity("AAAAA", "CCCCC") == pytest.approx(0.0)

    def test_partial_match(self):
        # 2 out of 4 positions match
        assert compute_sequence_identity("ACDE", "ACFF") == pytest.approx(0.5)

    def test_mismatched_lengths_uses_shorter(self):
        # "ACE" vs "ACE" for first 3 chars → identity on first 3
        result = compute_sequence_identity("ACEFG", "ACE")
        assert result == pytest.approx(1.0)

    def test_empty_sequences_returns_zero(self):
        assert compute_sequence_identity("", "") == pytest.approx(0.0)

    def test_single_char_match(self):
        assert compute_sequence_identity("A", "A") == pytest.approx(1.0)

    def test_single_char_mismatch(self):
        assert compute_sequence_identity("A", "C") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_site_recovery
# ---------------------------------------------------------------------------


class TestComputeSiteRecovery:
    def test_perfect_recovery(self):
        seq = "ACDEF"
        func_r, overall_r = compute_site_recovery(seq, seq, functional_sites=[1, 3, 5])
        assert func_r == pytest.approx(1.0)
        assert overall_r == pytest.approx(1.0)

    def test_no_recovery(self):
        designed = "GGGGG"
        wildtype = "ACDEF"
        func_r, overall_r = compute_site_recovery(designed, wildtype, functional_sites=[1, 2, 3])
        assert func_r == pytest.approx(0.0)
        assert overall_r == pytest.approx(0.0)

    def test_functional_sites_only_recovered(self):
        # Position 0 (0-indexed) is wrong; functional sites [2,3,4] (1-indexed) all match
        designed = "XCDEF"
        wildtype = "ACDEF"
        func_r, overall_r = compute_site_recovery(designed, wildtype, functional_sites=[2, 3, 4])
        assert func_r == pytest.approx(1.0)   # C==C, D==D, E==E
        assert overall_r == pytest.approx(4 / 5)  # X!=A, rest match → FSI = 1.25

    def test_out_of_range_functional_sites_ignored(self):
        designed = "ACDE"
        wildtype = "ACDE"
        # site 10 is out of range for length-4 seq → should be ignored
        func_r, overall_r = compute_site_recovery(designed, wildtype, functional_sites=[1, 10])
        assert func_r == pytest.approx(1.0)
        assert overall_r == pytest.approx(1.0)

    def test_empty_functional_sites(self):
        # No functional sites → func_r = 0.0 by convention
        designed = "ACDE"
        wildtype = "ACDE"
        func_r, overall_r = compute_site_recovery(designed, wildtype, functional_sites=[])
        assert func_r == pytest.approx(0.0)
        assert overall_r == pytest.approx(1.0)

    def test_1indexed_conversion(self):
        # Position 1 (1-indexed) → index 0 (0-indexed)
        designed = "XCDEF"
        wildtype = "ACDEF"
        func_r, _ = compute_site_recovery(designed, wildtype, functional_sites=[1])
        assert func_r == pytest.approx(0.0)  # X != A

        func_r, _ = compute_site_recovery(designed, wildtype, functional_sites=[2])
        assert func_r == pytest.approx(1.0)  # C == C


# ---------------------------------------------------------------------------
# compute_fsi
# ---------------------------------------------------------------------------


class TestComputeFSI:
    def test_fsi_greater_than_one(self):
        # Functional recovery 0.9, overall recovery 0.3 → FSI = 3.0
        fsi = compute_fsi(functional_recovery=0.9, overall_recovery=0.3)
        assert fsi == pytest.approx(3.0)

    def test_fsi_equal_to_one(self):
        fsi = compute_fsi(functional_recovery=0.5, overall_recovery=0.5)
        assert fsi == pytest.approx(1.0)

    def test_fsi_less_than_one(self):
        fsi = compute_fsi(functional_recovery=0.2, overall_recovery=0.8)
        assert fsi == pytest.approx(0.25)

    def test_zero_overall_recovery_with_functional(self):
        # Pathological case: model recovers functional sites despite zero overall recovery
        fsi = compute_fsi(functional_recovery=0.5, overall_recovery=0.0)
        assert fsi == math.inf

    def test_zero_overall_recovery_no_functional(self):
        # Both zero → FSI defined as 1.0 by convention
        fsi = compute_fsi(functional_recovery=0.0, overall_recovery=0.0)
        assert fsi == pytest.approx(1.0)

    def test_zero_functional_nonzero_overall(self):
        fsi = compute_fsi(functional_recovery=0.0, overall_recovery=0.5)
        assert fsi == pytest.approx(0.0)

    def test_bonta_expected_range(self):
        # BoNT-A (3BTA): mean FSI ≈ 3.07 in published results
        # Verify our formula produces reasonable values for that regime
        fsi = compute_fsi(functional_recovery=0.9, overall_recovery=0.293)
        assert fsi > 2.5

    def test_anthrax_expected_zero(self):
        # Anthrax PA (1ACC): FSI ≈ 0.00 — phi-clamp never recovered
        fsi = compute_fsi(functional_recovery=0.0, overall_recovery=0.35)
        assert fsi == pytest.approx(0.0)
