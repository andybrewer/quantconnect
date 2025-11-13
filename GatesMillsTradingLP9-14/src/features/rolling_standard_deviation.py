# region imports
from AlgorithmImports import *
# endregion
import numpy as np


def rolling_standard_deviation_feature(
    prices: list[float], window_size: int
) -> float | None:
    """
    Rolling Standard Deviation feature is the calculation of the
    Standard Deviation of the X most recent datapoints.
    Given the size of a window (number of consecutive datapoints)
    the Standard Deviation of the most recent datapoints is calculated,
    ignoring the ones outside the window.
    Returns None if the number of datapoints is smaller than the window size

    :param prices: list[float]
    :param window_size: int
    :return: float
    """

    number_of_price_points: int = len(prices)
    if number_of_price_points < window_size:
        print(
            "WARNING! The number of price points "
            f"submitted ({number_of_price_points}) is "
            "too small to calculate a Standard Deviation. "
            "Returning None"
        )
        return None

    prices_to_consider = prices[:]

    if number_of_price_points > window_size:
        prices_to_consider = prices_to_consider[number_of_price_points - window_size :]

    assert len(prices_to_consider) == window_size

    result = np.std(prices_to_consider)

    return result
