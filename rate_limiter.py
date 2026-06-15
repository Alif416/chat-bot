import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: int) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: deque[float] = deque()

    def allow(self) -> bool:
        now = time.time()
        while self._calls and self._calls[0] < now - self.window:
            self._calls.popleft()
        if len(self._calls) < self.max_calls:
            self._calls.append(now)
            return True
        return False

    @property
    def remaining(self) -> int:
        now = time.time()
        while self._calls and self._calls[0] < now - self.window:
            self._calls.popleft()
        return self.max_calls - len(self._calls)

    @property
    def reset_in(self) -> float:
        if not self._calls:
            return 0.0
        return max(0.0, self.window - (time.time() - self._calls[0]))
