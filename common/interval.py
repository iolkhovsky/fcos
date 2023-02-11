from dataclasses import dataclass
from enum import Enum
import time


class IntervalType(Enum):
    step = 1
    epoch = 2
    min = 3


@dataclass
class Interval:
    value: int
    type: IntervalType
    quantity_on_hand: int = 0

    @staticmethod
    def from_config(config) -> float:
        return Interval(
            type=IntervalType[config['type']],
            value=int(config['value'])
        )


class IntervalManager:
    def __init__(self, interval):
        self._last_value = 0
        if interval is not None:
            assert isinstance(interval, Interval)
            if interval.type == IntervalType.min:
                self._last_value = time.time()
        self._interval = interval

    def check(self, step, epoch):
        if self._interval is None:
            return False

        if self._interval.type == IntervalType.step:
            if step >= self._last_value + self._interval.value:
                self._last_value = step
                return True
            else:
                return False
        elif self._interval.type == IntervalType.epoch:
            if epoch >= self._last_value + self._interval.value:
                self._last_value = epoch
                return True
            else:
                return False
        elif self._interval.type == IntervalType.min:
            tstamp = time.time()
            if tstamp >= self._last_value + self._interval.value * 60.:
                self._last_value = tstamp
                return True
            else:
                return False
