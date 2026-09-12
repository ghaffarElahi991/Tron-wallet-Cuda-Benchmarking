import json
import sys
import tempfile
import unittest
from pathlib import Path

from tron_bench import (
    BASE58_SET,
    PatternError,
    estimate,
    match_probability,
    measure_trials,
    parse_fixed_pattern,
    quantile_attempts,
    summarize,
)


class PatternTests(unittest.TestCase):
    def test_case_insensitive_letter_expands(self):
        self.assertEqual(parse_fixed_pattern("a", True), [frozenset(("a", "A"))])

    def test_base58_singleton_case_classes(self):
        self.assertEqual(parse_fixed_pattern("I", True), [frozenset(("i",))])
        self.assertEqual(parse_fixed_pattern("l", True), [frozenset(("L",))])
        self.assertEqual(parse_fixed_pattern("O", True), [frozenset(("o",))])

    def test_wildcard_and_class(self):
        parsed = parse_fixed_pattern("?[a1]", True)
        self.assertEqual(parsed[0], BASE58_SET)
        self.assertEqual(parsed[1], frozenset(("a", "A", "1")))

    def test_rejects_variable_width_regex(self):
        with self.assertRaises(PatternError):
            parse_fixed_pattern("ab.*", True)


class ProbabilityTests(unittest.TestCase):
    def test_empty_pattern_matches_every_tron_address(self):
        favorable, total, _ = match_probability("", "", True)
        self.assertEqual(favorable, total)

    def test_lowercase_cannot_be_second_character_case_sensitive(self):
        favorable, _, _ = match_probability("a", "", False)
        self.assertEqual(favorable, 0)

    def test_ignore_case_maps_second_character_to_uppercase(self):
        lower_favorable, total, _ = match_probability("a", "", True)
        upper_favorable, upper_total, _ = match_probability("A", "", False)
        self.assertEqual(total, upper_total)
        self.assertEqual(lower_favorable, upper_favorable)

    def test_wildcard_after_fixed_t_matches_whole_space(self):
        favorable, total, _ = match_probability("?", "", True)
        self.assertEqual(favorable, total)

    def test_rate_must_be_positive(self):
        with self.assertRaises(ValueError):
            estimate("abcd", "uvwxyz", True, 0, 1, 30)

    def test_quantile(self):
        self.assertEqual(quantile_attempts(1.0, 0.5), 1)
        self.assertGreaterEqual(quantile_attempts(0.1, 0.95), 1)


class SummaryTests(unittest.TestCase):
    def test_jsonl_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            rows = [
                {"found": True, "elapsed_seconds": 1.0, "candidates": 10},
                {"found": True, "elapsed_seconds": 3.0, "candidates": 30},
                {"found": False, "elapsed_seconds": 9.0},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = summarize([path])
            self.assertEqual(result["successful_trials"], 2)
            self.assertEqual(result["mean_seconds"], 2.0)
            self.assertEqual(result["aggregate_rate_per_second"], 10.0)

    def test_measure_does_not_capture_child_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measured.jsonl"
            rows = measure_trials(
                command=[sys.executable, "-c", "print('private material must be discarded')"],
                output=output,
                trials=2,
                warmups=0,
                timeout_seconds=2,
                success_exit_code=0,
                pattern_label="test",
                gpu_label="none",
            )
            self.assertTrue(all(row["found"] for row in rows))
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("private material", text)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
