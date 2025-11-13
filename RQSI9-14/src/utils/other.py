# region imports
from AlgorithmImports import *
# endregion
def shift_to_anchor(high: float, low: float, anchor: float) -> tuple[float, float]:
    current_avg = (high + low) / 2.0
    shift = anchor - current_avg
    new_high = high + shift
    new_low = low + shift
    return new_high, new_low
