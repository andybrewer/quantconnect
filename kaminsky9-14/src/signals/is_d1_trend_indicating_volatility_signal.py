# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime

from ..features.intraday_slope import IntradayTrend, IntradayTrendClassification
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsD1TrendIndicatingVolatilitySignal(Signal):

    def decide(self, d1_intraday_trend: IntradayTrend | None) -> SignalDecision:
        if d1_intraday_trend is None:
            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                degree_of_certainty=1,
                relevant_data={},
            )

        is_high_volatility = (
            d1_intraday_trend.classification == IntradayTrendClassification.UPWARD
            and d1_intraday_trend.slope >= 0.01
        ) or (
            d1_intraday_trend.classification == IntradayTrendClassification.DOWNWARD
            and d1_intraday_trend.slope <= -0.01
        )

        return SignalDecision(
            decision_type=(
                SignalDecisionType.POSITIVE
                if is_high_volatility
                else SignalDecisionType.NEGATIVE
            ),
            degree_of_certainty=1,
            relevant_data={
                "D1 intraday trend slope": d1_intraday_trend.slope,
                "D1 intraday trend classification": d1_intraday_trend.classification.value,
            },
        )
