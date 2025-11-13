# region imports
from AlgorithmImports import *
# endregion
from enum import Enum


class OvernightGapClassification(str, Enum):
    POTENTIAL_HIGH_VOLATILITY = "potential_high_volatility"
    INCONCLUSIVE = "inconclusive"


def classify_dx_gap_magnitude(value: float, d: int) -> OvernightGapClassification:
    # the logic is that if the open of the day is a lot smaller than the close of the previous day
    # it is an indicator of potential high volatility
    if d == 0:
        if value < -0.0035 or value >= 0.0015:
            return OvernightGapClassification.POTENTIAL_HIGH_VOLATILITY
    elif d == 1:
        if value < -0.0035 or value >= 0.002:
            return OvernightGapClassification.POTENTIAL_HIGH_VOLATILITY
    else:
        raise ValueError(f"Unexpected d valueL {d}")
    return OvernightGapClassification.INCONCLUSIVE


def dx_gap_magnitude_feature(
    open_price_of_current_day: float, close_price_of_previous_day: float
) -> float:
    """
    Gap Magnitude feature is the coeficient of growth between the market close and the market open (gap)

    :param open_price_of_current_day: float
    :param close_price_of_previous_day: float
    :return: float
    """

    result = (
        open_price_of_current_day - close_price_of_previous_day
    ) / close_price_of_previous_day

    return result


def gap_magnitude_2_day_average_feature(
    open_price_of_current_day: float,
    close_price_of_previous_day: float,
    open_price_of_previous_day: float,
    close_price_of_day_before_yesterday: float,
) -> float:
    """
    Gap Magnitude 2 Day Average feature is the average between today's Gap Magnitude and the previous day's

    :param open_price_of_current_day: float,
    :param close_price_of_previous_day: float,
    :param open_price_of_previous_day: float,
    :param close_price_of_day_before_yesterday: float,
    :return: float
    """

    this_day_gap_magnitude = dx_gap_magnitude_feature(
        open_price_of_current_day, close_price_of_previous_day
    )
    previous_day_gap_magnitude = dx_gap_magnitude_feature(
        open_price_of_previous_day, close_price_of_day_before_yesterday
    )

    result = (this_day_gap_magnitude + previous_day_gap_magnitude) / 2

    return result
