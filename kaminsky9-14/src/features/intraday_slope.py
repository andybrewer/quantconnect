# region imports
from AlgorithmImports import *
# endregion
from enum import Enum
from pydantic import BaseModel

from .dx_trend import dx_trend_feature


class IntradayTrendClassification(str, Enum):
    UPWARD = "upward"
    DOWNWARD = "downward"
    NOISE = "noise"


class IntradayTrend(BaseModel):
    slope: float
    classification: IntradayTrendClassification
    percentile: float | None = None


def dx_intraday_trend_slope(
    open_value: float, current_close_value: float
) -> IntradayTrend:

    slope = dx_trend_feature(open_value, current_close_value)
    classification = IntradayTrendClassification.NOISE

    return IntradayTrend(slope=slope, classification=classification)
