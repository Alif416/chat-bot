import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rate_limiter import RateLimiter


class TestRateLimiter(unittest.TestCase):
    def test_allows_under_limit(self):
        rl = RateLimiter(max_calls=5, window_seconds=60)
        for _ in range(5):
            self.assertTrue(rl.allow())

    def test_blocks_at_limit(self):
        rl = RateLimiter(max_calls=3, window_seconds=60)
        for _ in range(3):
            rl.allow()
        self.assertFalse(rl.allow())

    def test_remaining_decrements(self):
        rl = RateLimiter(max_calls=5, window_seconds=60)
        self.assertEqual(rl.remaining, 5)
        rl.allow()
        self.assertEqual(rl.remaining, 4)
        rl.allow()
        self.assertEqual(rl.remaining, 3)

    def test_reset_in_zero_when_empty(self):
        rl = RateLimiter(max_calls=5, window_seconds=60)
        self.assertEqual(rl.reset_in, 0.0)

    def test_reset_in_positive_after_call(self):
        rl = RateLimiter(max_calls=5, window_seconds=60)
        rl.allow()
        self.assertGreater(rl.reset_in, 0.0)
        self.assertLessEqual(rl.reset_in, 60.0)

    def test_window_expiry_allows_again(self):
        rl = RateLimiter(max_calls=2, window_seconds=1)
        rl.allow()
        rl.allow()
        self.assertFalse(rl.allow())
        time.sleep(1.05)
        self.assertTrue(rl.allow())


if __name__ == "__main__":
    unittest.main()
