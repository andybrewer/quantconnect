# region imports
from AlgorithmImports import *
# endregion
from .signals_common import Signal, SignalDecision, SignalDecisionType
from ..features.opening_range import OpeningRange
from ..models.models import MyCandle


class IsCurrentCandleClosingSufficientlyAwayFromOpeningRangeSignal(Signal):
    def decide(
        self,
        opening_range: OpeningRange | None,
        percentage: float,
        current_candle: MyCandle,
    ) -> SignalDecision:
        if opening_range is None:
            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                relevant_data={},
                degree_of_certainty=1,
            )
        opening_range_thickness = abs(opening_range.high - opening_range.low) * (
            percentage / 100
        )

        if (
            current_candle.close <= opening_range.low - opening_range_thickness
            or current_candle.close >= opening_range.high + opening_range_thickness
        ):
            signal_decision_type = SignalDecisionType.POSITIVE
        else:
            signal_decision_type = SignalDecisionType.NEGATIVE

        return SignalDecision(
            decision_type=signal_decision_type,
            relevant_data={
                "current_candle.close": current_candle.close,
                "opening_range_thickness": opening_range_thickness,
                "opening_range_below_low_limit": opening_range.low
                - opening_range_thickness,
                "opening_range_above_high_limit": opening_range.high
                + opening_range_thickness,
            },
            degree_of_certainty=1,
        )
