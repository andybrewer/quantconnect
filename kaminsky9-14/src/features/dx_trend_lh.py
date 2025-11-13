# region imports
from AlgorithmImports import *
# endregion
def dx_trend_lh_feature(
    lowest_price: float, highest_price: float, close_price: float
) -> float:
    """
    D-X Trend LH (Low/High) feature is the biggest change in price from the highest
    to the close OR the lowest to close (which one is bigger in absolute value)

    :param lowest_price: float
    :param highest_price: float
    :param close_price_of_day: float
    :return: float
    """

    assert lowest_price <= highest_price

    low_to_close_gap = close_price - lowest_price
    high_to_close_gap = highest_price - close_price

    assert low_to_close_gap >= 0
    assert high_to_close_gap >= 0

    low_coeficient = (close_price - lowest_price) / lowest_price
    high_coeficient = (close_price - highest_price) / highest_price

    if low_to_close_gap > high_to_close_gap:
        return low_coeficient
    else:
        return high_coeficient

    # TODO: handle cases where both absolute values are the same
