# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime, timedelta
from ..features.consolidation_box import (
    BoxBreakInsideClassification,
    BoxBreakOpeningRangeClassification,
    ConsolidationBoxFactory,
)
from .signals_common import Signal, SignalDecision, SignalDecisionType


class ABoxJustClosedHigherOrLowerSignal(Signal):
    def __init__(
        self,
        name: str,
        looking_for_high: bool,
        looking_for_low: bool,
    ) -> None:
        super().__init__(name=name)
        assert looking_for_high + looking_for_low == 1
        self.looking_for_high = looking_for_high
        self.looking_for_low = looking_for_low

    def decide(
        self,
        current_datetime: datetime,
        consolidation_box_factories: list[ConsolidationBoxFactory],
    ) -> SignalDecision:

        box_break_inside_goal = (
            BoxBreakInsideClassification.UP
            if self.looking_for_high
            else BoxBreakInsideClassification.DOWN
        )
        box_break_according_to_opening_range_goal = (
            BoxBreakOpeningRangeClassification.ABOVE
            if self.looking_for_high
            else BoxBreakOpeningRangeClassification.BELLOW
        )

        for box_factory in consolidation_box_factories:
            for cons_box in box_factory.closed_consolidation_boxes:
                if (
                    current_datetime
                    - timedelta(minutes=box_factory.minutes_to_consider_a_hod_or_lod_after_close)
                    <= cons_box.end_moment
                    and cons_box.box_break_inside == box_break_inside_goal
                    and cons_box.box_break_according_to_opening_range == box_break_according_to_opening_range_goal
                ):
                    return SignalDecision(
                        decision_type=SignalDecisionType.POSITIVE_BUY if self.looking_for_high else SignalDecisionType.POSITIVE_SELL,
                        degree_of_certainty=1,
                        relevant_data={"consolidation_box": cons_box.model_dump()},
                    )

        return SignalDecision(
            decision_type=SignalDecisionType.NOTHING,
            degree_of_certainty=1,
            relevant_data={},
        )
