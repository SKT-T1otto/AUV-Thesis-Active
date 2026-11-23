"""Zero-step checks of route success bounds against attainable extrema."""

import unittest
from itertools import permutations, product
from math import fsum, inf, nan

from chapter3_bser.handoff_value.bounds import NO_HANDOFF, success_difference


ABS_TOL = 1e-12
BINS = (NO_HANDOFF, "near", "far", "uncertain")
A = (0.10, 0.45, 0.15, 0.30)
B = (0.10, 0.25, 0.35, 0.30)
LOWER = (0.0, 0.65, 0.10, 0.0)
UPPER = (0.0, 0.80, 0.30, 1.0)
GAPS = (0.0, 0.0, 0.0, 0.10)


class HandoffBoundsTests(unittest.TestCase):
    def assert_interval(self, actual, lower, upper):
        self.assertAlmostEqual(actual.lower, lower, delta=ABS_TOL)
        self.assertAlmostEqual(actual.upper, upper, delta=ABS_TOL)
        self.assertLessEqual(-1.0, actual.lower)
        self.assertLessEqual(actual.lower, actual.upper)
        self.assertLessEqual(actual.upper, 1.0)

    def test_256_vertex_extrema(self):
        count = 0
        for a, b, lo, hi, gap in product(
            (0, 0.2, 0.5, 1), (0, 0.2, 0.5, 1), (0, 0.2), (0.5, 1), (0, 0.125, 0.5, 1)
        ):
            with self.subTest(a=a, b=b, lo=lo, hi=hi, gap=gap):
                width = min(hi - lo, gap)
                vertices = (
                    (lo, lo), (hi, hi), (lo, lo + width), (lo + width, lo),
                    (hi, hi - width), (hi - width, hi),
                )
                attainable = [a * x - b * y for x, y in vertices]
                actual = success_difference(
                    (1 - a, a), (1 - b, b), (0, lo), (0, hi), (0, gap),
                    bin_ids=(NO_HANDOFF, "handoff"),
                )
                self.assert_interval(actual, min(attainable), max(attainable))
            count += 1
        self.assertEqual(count, 256)

    def test_three_four_bin_examples(self):
        for gap, expected in ((0, (0.07, 0.14)), (0.10, (0.04, 0.17)), (0.25, (-0.005, 0.215))):
            with self.subTest(gap=gap):
                actual = success_difference(A, B, LOWER, UPPER, (0, 0, 0, gap), bin_ids=BINS)
                self.assert_interval(actual, *expected)

    def test_swapping_routes_reflects_the_interval(self):
        for gap, tv_a, tv_b in product((0.0, 0.1, 1.0), (0.0, 0.03, 0.83), (0.0, 0.09, 0.52)):
            with self.subTest(gap=gap, tv_a=tv_a, tv_b=tv_b):
                gaps = (0, gap, gap, gap)
                ab = success_difference(A, B, LOWER, UPPER, gaps, bin_ids=BINS, tv_a=tv_a, tv_b=tv_b)
                ba = success_difference(B, A, LOWER, UPPER, gaps, bin_ids=BINS, tv_a=tv_b, tv_b=tv_a)
                self.assert_interval(ba, -ab.upper, -ab.lower)

    def test_increasing_either_tv_radius_only_widens_bounds(self):
        def bounds(tv_a, tv_b):
            return success_difference(A, B, LOWER, UPPER, GAPS, bin_ids=BINS, tv_a=tv_a, tv_b=tv_b)

        for tv_a, tv_b in product((0.0, 0.03, 0.20, 0.85, 1.0), repeat=2):
            with self.subTest(tv_a=tv_a, tv_b=tv_b):
                base = bounds(tv_a, tv_b)
                self.assert_interval(base, max(-1, 0.04 - tv_a - tv_b), min(1, 0.17 + tv_a + tv_b))
                for wider in (bounds(min(1, tv_a + 0.2), tv_b), bounds(tv_a, min(1, tv_b + 0.2))):
                    self.assertLessEqual(wider.lower, base.lower)
                    self.assertGreaterEqual(wider.upper, base.upper)

    def test_full_bin_width_recovers_separate_route_intervals(self):
        gaps = tuple(hi - lo for lo, hi in zip(LOWER, UPPER))
        for (a, b), (tv_a, tv_b) in product(((A, B), (B, A), (A, A)), ((0, 0), (0.02, 0.03))):
            with self.subTest(a=a, b=b, tv_a=tv_a, tv_b=tv_b):
                a_low = fsum(p * lo for p, lo in zip(a, LOWER))
                a_high = fsum(p * hi for p, hi in zip(a, UPPER))
                b_low = fsum(p * lo for p, lo in zip(b, LOWER))
                b_high = fsum(p * hi for p, hi in zip(b, UPPER))
                actual = success_difference(a, b, LOWER, UPPER, gaps, bin_ids=BINS, tv_a=tv_a, tv_b=tv_b)
                self.assert_interval(actual, (a_low - tv_a) - (b_high + tv_b), (a_high + tv_a) - (b_low - tv_b))

    def test_bin_order_invariance_including_no_handoff(self):
        for order in permutations(range(len(BINS))):
            with self.subTest(order=order):
                ids = tuple(BINS[i] for i in order)
                columns = [tuple(values[i] for i in order) for values in (A, B, LOWER, UPPER, GAPS)]
                self.assert_interval(success_difference(*columns, bin_ids=ids), 0.04, 0.17)

    def test_no_handoff_and_zero_mass_bins(self):
        self.assert_interval(success_difference((1,), (1,), (0,), (0,), (0,), bin_ids=(NO_HANDOFF,)), 0, 0)
        mass = (1, 0, 0, 0)
        self.assert_interval(success_difference(mass, mass, LOWER, UPPER, GAPS, bin_ids=BINS), 0, 0)

    def test_same_masses_with_shared_and_distinct_success_rates(self):
        shared = success_difference(A, A, LOWER, UPPER, (0, 0, 0, 0), bin_ids=BINS)
        partial = success_difference(A, A, LOWER, UPPER, GAPS, bin_ids=BINS)
        self.assert_interval(shared, 0, 0)
        self.assert_interval(partial, -0.03, 0.03)

    def test_missing_duplicate_or_nonzero_no_handoff(self):
        # The first ordinary bin can also have value [0, 0].
        for ids in ((), ("zero_success_handoff", "near", "far", "uncertain")):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "no_handoff"):
                success_difference(A, B, LOWER, UPPER, GAPS, bin_ids=ids)
        for ids in ((NO_HANDOFF, NO_HANDOFF, "far", "uncertain"), (NO_HANDOFF, "near", "near", "far")):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "unique"):
                success_difference(A, B, LOWER, UPPER, GAPS, bin_ids=ids)
        with self.assertRaisesRegex(ValueError, r"no_handoff.*\[0, 0\]"):
            success_difference(A, B, LOWER, (0.01, *UPPER[1:]), GAPS, bin_ids=BINS)

    def test_bad_probability_sum_for_each_route(self):
        for route, a, b in (("mass_a", (0.09, *A[1:]), B), ("mass_b", A, (0.11, *B[1:]))):
            with self.subTest(route=route), self.assertRaisesRegex(ValueError, route + " must sum to 1"):
                success_difference(a, b, LOWER, UPPER, GAPS, bin_ids=BINS)

    def test_inverted_interval(self):
        with self.assertRaisesRegex(ValueError, "inverted interval.*near"):
            success_difference(A, B, (0, 0.81, 0.1, 0), UPPER, GAPS, bin_ids=BINS)

    def test_dimension_and_invalid_values_name_the_field(self):
        inputs = dict(mass_a=A, mass_b=B, lower=LOWER, upper=UPPER, within_bin_gap=GAPS)
        for field, vector in inputs.items():
            with self.subTest(field=field, issue="length"), self.assertRaisesRegex(ValueError, field):
                success_difference(**{**inputs, field: vector[:-1]}, bin_ids=BINS)
            for bad in (nan, -inf, inf, -0.01, 1.01):
                changed = (vector[0], bad, *vector[2:])
                with self.subTest(field=field, bad=bad), self.assertRaisesRegex(ValueError, field + r"\[near\]"):
                    success_difference(**{**inputs, field: changed}, bin_ids=BINS)
        for field in ("tv_a", "tv_b"):
            for bad in (nan, -inf, inf, -0.01, 1.01):
                with self.subTest(field=field, bad=bad), self.assertRaisesRegex(ValueError, field):
                    success_difference(**inputs, bin_ids=BINS, **{field: bad})

    def test_tolerated_mass_roundoff_preserves_order_at_extremes(self):
        a, b = (0.0, 0.5, 0.5 + 5e-13), (1.0, 0.0, 0.0)
        exact = (0, 1, 1)
        for first, second, expected in ((a, b, 1.0), (b, a, -1.0)):
            with self.subTest(expected=expected):
                actual = success_difference(first, second, exact, exact, (0, 0, 0), bin_ids=BINS[:3])
                self.assertEqual((actual.lower, actual.upper), (expected, expected))


if __name__ == "__main__":
    unittest.main()
