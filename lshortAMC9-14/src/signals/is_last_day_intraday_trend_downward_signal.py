# region imports
from AlgorithmImports import *
# endregion
from ..features.intraday_slope import IntradayTrendClassification
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsLastDayIntradayTrendDirectionDownwardSignal(Signal):

    def decide(
        self, intraday_trend_classification: IntradayTrendClassification | None
    ) -> SignalDecision:

        return SignalDecision(
            decision_type=(
                SignalDecisionType.POSITIVE
                if intraday_trend_classification == IntradayTrendClassification.DOWNWARD
                else SignalDecisionType.NEGATIVE
            ),
            degree_of_certainty=1,
            relevant_data={
                "last_day_intraday_trend_classification": intraday_trend_classification
            },
        )
