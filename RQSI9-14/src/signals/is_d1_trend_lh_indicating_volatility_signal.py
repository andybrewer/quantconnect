# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime

from ..features.intraday_slope import IntradayTrend, IntradayTrendClassification
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsD1TrendLHAndPercentileIndicatingVolatilitySignal(Signal):
    def decide(
        self, 
        d1_trend_lh: float, 
        d1_trend_percentile: float, 
        d2_trend_lh: float | None = None
    ) -> SignalDecision:
        # If d1_trend_percentile is missing, return a negative decision immediately.
        if d1_trend_percentile is None:
            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                degree_of_certainty=1,
                relevant_data={
                    "D1 intraday trend lh": d1_trend_lh,
                    "D1 intraday trend percentile": d1_trend_percentile,
                    "D2 intraday trend lh": d2_trend_lh,
                },
            )

        # Determine high volatility based on thresholds.
        if d2_trend_lh is None:
            is_high_volatility = abs(d1_trend_lh) >= 0.009
        else:
            is_high_volatility = abs(d1_trend_lh) >= 0.009 or abs(d2_trend_lh) >= 0.009

        # Always return a SignalDecision.
        return SignalDecision(
            decision_type=SignalDecisionType.POSITIVE if is_high_volatility else SignalDecisionType.NEGATIVE,
            degree_of_certainty=1,
            relevant_data={
                "D1 intraday trend lh": d1_trend_lh,
                "D1 intraday trend percentile": d1_trend_percentile,
                "D2 intraday trend lh": d2_trend_lh,
            },
        )
