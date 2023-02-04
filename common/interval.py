from dataclasses import dataclass
from enum import Enum

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
