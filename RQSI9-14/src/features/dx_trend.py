# region imports
from AlgorithmImports import *
# endregion
def dx_trend_feature(open_price: float, close_price: float) -> float:
    """
    D-X Trend feature is the coeficient of growth of the close price compared to the open price

    :param open_price: float
    :param close_price: float
    :return: float
    """

    result = (close_price - open_price) / open_price

    return result


def get_closing_price_from_knowing_trend_and_open_price(
    open_price: float, trend: float
) -> float:

    result = (trend * open_price) + open_price

    return result
